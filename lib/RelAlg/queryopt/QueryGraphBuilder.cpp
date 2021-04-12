#include "mlir/Dialect/RelAlg/queryopt/QueryGraphBuilder.h"

namespace mlir::relalg {
static node_set getNodeSetFromClass(llvm::EquivalenceClasses<size_t>& classes, size_t val, size_t numNodes) {
   node_set res(numNodes);
   auto eqclass = classes.findLeader(val);
   for (auto me = classes.member_end(); eqclass != me; ++eqclass) {
      res.set(*eqclass);
   }
   return res;
}
static node_set getNodeSetFromClasses(llvm::EquivalenceClasses<size_t>& classes, const node_set& s) {
   node_set res(s.size());
   for (auto pos : s) {
      auto eqclass = classes.findLeader(pos);
      for (auto me = classes.member_end(); eqclass != me; ++eqclass) {
         res.set(*eqclass);
      }
   }
   return res;
}
static bool isConnected(llvm::EquivalenceClasses<size_t>& connections, node_set& s) {
   assert(s.any());
   size_t firstClass = s.size();
   bool connected = true;
   for (auto pos : s) {
      if (firstClass == s.size()) {
         firstClass = connections.getLeaderValue(pos);
      } else {
         if (firstClass != connections.getLeaderValue(pos)) {
            connected = false;
         }
      }
   }
   return connected;
}
size_t countCreatingOperators(Operator op, std::unordered_set<mlir::Operation*>& alreadyOptimized) {
   size_t res = 0;
   auto children = op.getChildren();
   auto used = op.getUsedAttributes();
   auto created = op.getCreatedAttributes();
   if (alreadyOptimized.count(op.getOperation())) {
      res += 1;
      return res;
   }
   for (auto child : children) {
      res += countCreatingOperators(child, alreadyOptimized);
   }
   res += !created.empty();
   return res;
}
QueryGraphBuilder::NodeResolver mlir::relalg::QueryGraphBuilder::populateQueryGraph(Operator op) {
   auto children = op.getChildren();
   auto used = op.getUsedAttributes();
   auto created = op.getCreatedAttributes();
   NodeResolver resolver(qg);
   if (already_optimized.count(op.getOperation())) {
      size_t newNode = addNode(op);
      for (auto* attr : op.getAvailableAttributes()) {
         resolver.add(attr, newNode);
      }
      return resolver;
   }
   for (auto child : children) {
      resolver.merge(populateQueryGraph(child));
   }
   if (mlir::isa<mlir::relalg::CrossProductOp>(op.getOperation())) {
      //do not construct crossproducts in the querygraph
   } else if (mlir::relalg::detail::isJoin(op.getOperation()) && !mlir::isa<mlir::relalg::InnerJoinOp>(op.getOperation())) {
      //add join edges into the query graph
      node_set tes = calcTES(op, resolver);
      node_set leftTes = calcT(children[0], resolver) & tes;
      node_set rightTes = calcT(children[1], resolver) & tes;
      qg.addEdge(expand(leftTes), expand(rightTes), empty_node, op, mlir::relalg::QueryGraph::EdgeType::REAL);
      if (!created.empty()) {
         size_t newNode = addNode(op);
         for (auto* attr : op.getCreatedAttributes()) {
            resolver.add(attr, newNode);
         }
         qg.nodes[newNode].dependencies = expand(tes);
         qg.addEdge(expand(tes), node_set::single(num_nodes, newNode), empty_node, op, mlir::relalg::QueryGraph::EdgeType::IGNORE);
      }
   } else if (!created.empty()) {
      //add node for operators that create attributes
      size_t newNode = addNode(op);
      for (auto* attr : op.getCreatedAttributes()) {
         resolver.add(attr, newNode);
      }
      if (children.size() == 1) {
         //if operator has one child e.g. aggregation/renaming/map
         // -> create "implicit" hyperedge
         node_set tes = calcTES(op, resolver);
         qg.nodes[newNode].dependencies = expand(tes);
         qg.addEdge(expand(tes), node_set::single(num_nodes, newNode), empty_node, op, mlir::relalg::QueryGraph::EdgeType::IMPLICIT);
      }
   } else if (mlir::isa<mlir::relalg::SelectionOp>(op.getOperation()) || mlir::isa<mlir::relalg::InnerJoinOp>(op.getOperation())) {
      node_set ses = calcSES(op, resolver);
      if (ses.count() == 1) {
         //if selection is only based on one node -> add selection to node
         auto nodeId = ses.find_first();
         qg.nodes[nodeId].additional_predicates.push_back(op);
      } else {
         auto first = ses.find_first();
         llvm::EquivalenceClasses<size_t> cannotBeSeperated;
         for (auto pos : ses) {
            cannotBeSeperated.insert(pos);
            for (auto dep : qg.nodes[pos].dependencies) {
               cannotBeSeperated.unionSets(pos, dep);
            }
         }

         for (Operator subop : op.getAllSubOperators()) {
            if (subop != op) {
               auto subopTes = calcTES(subop, resolver);
               if (ses.intersects(subopTes) && !canPushSelection(ses, subop, resolver)) {
                  auto representant = (subopTes & ses).find_first();
                  for (auto pos : (subopTes & ses)) {
                     cannotBeSeperated.unionSets(representant, pos);
                  }
               }
            }
         }
         if (cannotBeSeperated.getNumClasses() == 1) {
            qg.addEdge(getNodeSetFromClass(cannotBeSeperated, first, num_nodes), empty_node, empty_node, op, mlir::relalg::QueryGraph::EdgeType::REAL);
         } else {
            node_set decisions = empty_node;
            for (const auto& a : cannotBeSeperated) {
               if (a.isLeader()) {
                  decisions.set(a.getData());
               }
            }
            decisions.iterateSubsets([&](node_set left) {
               node_set right = decisions & ~left;
               if (left < right) {
                  left = getNodeSetFromClasses(cannotBeSeperated, left);
                  right = getNodeSetFromClasses(cannotBeSeperated, right);

                  qg.addEdge(left, right, empty_node, op, mlir::relalg::QueryGraph::EdgeType::REAL);
               }
            });
         }
      }
   } else {
      assert(false && " should not happen");
   }
   return resolver;
}
void QueryGraphBuilder::ensureConnected() {
   llvm::EquivalenceClasses<size_t> alreadyConnected;
   for (size_t i = 0; i < qg.getNodes().size(); i++) {
      alreadyConnected.insert(i);
   }

   std::list<size_t> edgesToProcess;
   for (size_t i = 0; i < qg.getEdges().size(); i++) {
      if (qg.getEdges()[i].left.any() && qg.getEdges()[i].right.any()) {
         edgesToProcess.push_back(i);
      }
   }
   for (size_t i = 0; i < qg.getEdges().size(); i++) {
      std::list<size_t> newList;
      for (auto edgeid : edgesToProcess) {
         auto& edge = qg.getEdges()[edgeid];
         if (isConnected(alreadyConnected, edge.left) && isConnected(alreadyConnected, edge.right)) {
            alreadyConnected.unionSets(edge.left.find_first(), edge.right.find_first());
         } else {
            newList.push_back(edgeid);
         }
      }

      std::swap(edgesToProcess, newList);
   }
   for (const auto& a : alreadyConnected) {
      if (a.isLeader()) {
         for (const auto& b : alreadyConnected) {
            if (b.isLeader()) {
               if (a.getData() != b.getData()) {
                  //std::cout << a.getData() << " vs " << b.getData() << "\n";
                  node_set left = getNodeSetFromClass(alreadyConnected, a.getData(), num_nodes);
                  if (expand(left) != left) {
                     continue;
                  }
                  node_set right = getNodeSetFromClass(alreadyConnected, b.getData(), num_nodes);
                  if (expand(right) != right) {
                     continue;
                  }
                  //std::cout << "would lead to edge (" << left << "," << right << ")\n";
                  bool connectingEdgeExists = false;
                  for (auto& edge : qg.getEdges()) {
                     if ((left.intersects(edge.left) && right.intersects(edge.right)) || (left.intersects(edge.right) && right.intersects(edge.left))) {
                        if (edge.op && !mlir::isa<mlir::relalg::SelectionOp>(edge.op.getOperation()) && !mlir::isa<mlir::relalg::InnerJoinOp>(edge.op.getOperation())) {
                           //std::cout << "but prohibited by edge (" << edge.left << "," << edge.right << ")\n";
                           connectingEdgeExists = true;
                        }
                        break;
                     }
                     if (left == edge.left || right == edge.right || left == edge.right || right == edge.left) {
                        if (edge.op && !mlir::isa<mlir::relalg::SelectionOp>(edge.op.getOperation()) && !mlir::isa<mlir::relalg::InnerJoinOp>(edge.op.getOperation())) {
                           //std::cout << "but prohibited by edge (" << edge.left << "," << edge.right << ")\n";
                           connectingEdgeExists = true;
                        }
                        break;
                     }
                  }
                  if (!connectingEdgeExists) {
                     qg.addEdge(left, right, node_set(num_nodes), Operator(), QueryGraph::EdgeType::REAL);
                  }
               }
            }
         }
      }
   }
}
node_set QueryGraphBuilder::calcTES(Operator b, NodeResolver& resolver) {
   if (TESs.count(b.getOperation())) {
      return TESs[b.getOperation()];
   } else {
      node_set tes = calcSES(b, resolver);
      auto children = b.getChildren();
      if (children.size() == 2) {
         auto [b_left, b_right] = normalizeChildren(b);
         for (auto a : b_left.getAllSubOperators()) {
            if (a.getChildren().size() == 2) {
               auto [a_left, a_right] = normalizeChildren(a);
               if (!detail::binaryOperatorIs(detail::assoc, a, b)) {
                  tes |= calcT(a_left, resolver);
               }
               if (!detail::binaryOperatorIs(detail::lAsscom, a, b)) {
                  tes |= calcT(a_right, resolver);
               }
            } else {
               if (mlir::isa<mlir::relalg::AggregationOp>(a.getOperation())) {
                  tes |= calcT(a, resolver);
               }
            }
         }
         for (auto a : b_right.getAllSubOperators()) {
            if (a.getChildren().size() == 2) {
               auto [a_left, a_right] = normalizeChildren(a);
               if (!detail::binaryOperatorIs(detail::assoc, b, a)) {
                  tes |= calcT(a_right, resolver);
               }
               if (!detail::binaryOperatorIs(detail::rAsscom, b, a)) {
                  tes |= calcT(a_left, resolver);
               }
            } else {
               if (mlir::isa<mlir::relalg::AggregationOp>(a.getOperation())) {
                  tes |= calcT(a, resolver);
               }
            }
         }

      } else if (children.size() == 1) {
         auto onlyChild = children[0];
         if (mlir::isa<mlir::relalg::AggregationOp>(b.getOperation())) {
            tes |= calcT(onlyChild, resolver);
         }
         if (auto renameop = mlir::dyn_cast_or_null<mlir::relalg::RenamingOp>(b.getOperation())) {
            for (auto a : onlyChild.getAllSubOperators()) {
               if (a.getUsedAttributes().intersects(renameop.getUsedAttributes()) || a.getCreatedAttributes().intersects(renameop.getCreatedAttributes())) {
                  tes |= calcT(onlyChild, resolver);
               }
            }
         }
      }
      TESs[b.getOperation()] = tes;
      return tes;
   }
}
node_set QueryGraphBuilder::calcSES(Operator op, NodeResolver& resolver) const {
   node_set res = node_set(num_nodes);
   for (auto* attr : op.getUsedAttributes()) {
      res.set(resolver.resolve(attr));
   }
   return res;
}

QueryGraphBuilder::QueryGraphBuilder(Operator root, std::unordered_set<mlir::Operation*>& alreadyOptimized) : root(root),
                                                                                                              already_optimized(alreadyOptimized),
                                                                                                              num_nodes(countCreatingOperators(root, alreadyOptimized)),
                                                                                                              qg(num_nodes, alreadyOptimized),
                                                                                                              empty_node(num_nodes) {}

} // namespace mlir::relalg