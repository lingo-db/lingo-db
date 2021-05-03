#include "mlir/Dialect/RelAlg/queryopt/QueryGraphBuilder.h"
#include <list>

namespace mlir::relalg {
static NodeSet getNodeSetFromClass(llvm::EquivalenceClasses<size_t>& classes, size_t val, size_t numNodes) {
   NodeSet res(numNodes);
   auto eqclass = classes.findLeader(val);
   for (auto me = classes.member_end(); eqclass != me; ++eqclass) {
      res.set(*eqclass);
   }
   return res;
}
static NodeSet getNodeSetFromClasses(llvm::EquivalenceClasses<size_t>& classes, const NodeSet& s) {
   NodeSet res(s.size());
   for (auto pos : s) {
      auto eqclass = classes.findLeader(pos);
      for (auto me = classes.member_end(); eqclass != me; ++eqclass) {
         res.set(*eqclass);
      }
   }
   return res;
}
static bool isConnected(llvm::EquivalenceClasses<size_t>& connections, NodeSet& s) {
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
size_t countCreatingOperators(Operator op, llvm::SmallPtrSet<mlir::Operation*,12>& alreadyOptimized) {
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
   if (alreadyOptimized.count(op.getOperation())) {
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
      NodeSet tes = calcTES(op, resolver);
      NodeSet leftTes = calcT(children[0], resolver) & tes;
      NodeSet rightTes = calcT(children[1], resolver) & tes;
      qg.addEdge(expand(leftTes), expand(rightTes), emptyNode, op, mlir::relalg::QueryGraph::EdgeType::REAL);
      if (!created.empty()) {
         size_t newNode = addNode(op);
         for (auto* attr : op.getCreatedAttributes()) {
            resolver.add(attr, newNode);
         }
         qg.nodes[newNode].dependencies = expand(tes);
         qg.addEdge(expand(tes), NodeSet::single(numNodes, newNode), emptyNode, op, mlir::relalg::QueryGraph::EdgeType::IGNORE);
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
         NodeSet tes = calcTES(op, resolver);
         qg.nodes[newNode].dependencies = expand(tes);
         qg.addEdge(expand(tes), NodeSet::single(numNodes, newNode), emptyNode, op, mlir::relalg::QueryGraph::EdgeType::IMPLICIT);
      }
   } else if (mlir::isa<mlir::relalg::SelectionOp>(op.getOperation()) || mlir::isa<mlir::relalg::InnerJoinOp>(op.getOperation())) {
      NodeSet ses = calcSES(op, resolver);
      if (ses.count() == 1) {
         //if selection is only based on one node -> add selection to node
         auto nodeId = ses.findFirst();
         qg.nodes[nodeId].additionalPredicates.push_back(op);
      } else {
         auto first = ses.findFirst();
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
                  auto representant = (subopTes & ses).findFirst();
                  for (auto pos : (subopTes & ses)) {
                     cannotBeSeperated.unionSets(representant, pos);
                  }
               }
            }
         }
         if (cannotBeSeperated.getNumClasses() == 1) {
            qg.addEdge(getNodeSetFromClass(cannotBeSeperated, first, numNodes), emptyNode, emptyNode, op, mlir::relalg::QueryGraph::EdgeType::REAL);
         } else {
            NodeSet decisions = emptyNode;
            for (const auto& a : cannotBeSeperated) {
               if (a.isLeader()) {
                  decisions.set(a.getData());
               }
            }
            decisions.iterateSubsets([&](NodeSet left) {
               NodeSet right = decisions & ~left;
               if (left < right) {
                  left = getNodeSetFromClasses(cannotBeSeperated, left);
                  right = getNodeSetFromClasses(cannotBeSeperated, right);

                  qg.addEdge(left, right, emptyNode, op, mlir::relalg::QueryGraph::EdgeType::REAL);
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
            alreadyConnected.unionSets(edge.left.findFirst(), edge.right.findFirst());
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
                  NodeSet left = getNodeSetFromClass(alreadyConnected, a.getData(), numNodes);
                  if (expand(left) != left) {
                     continue;
                  }
                  NodeSet right = getNodeSetFromClass(alreadyConnected, b.getData(), numNodes);
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
                     qg.addEdge(left, right, NodeSet(numNodes), Operator(), QueryGraph::EdgeType::REAL);
                  }
               }
            }
         }
      }
   }
}
NodeSet QueryGraphBuilder::calcTES(Operator op, NodeResolver& resolver) {
   if (teSs.count(op.getOperation())) {
      return teSs[op.getOperation()];
   } else {
      NodeSet tes = calcSES(op, resolver);
      auto children = op.getChildren();
      if (auto b = mlir::dyn_cast_or_null<BinaryOperator>(op.getOperation())) {
         auto [b_left, b_right] = normalizeChildren(op);
         for (auto subOp : b_left.getAllSubOperators()) {
            if (auto a = mlir::dyn_cast_or_null<BinaryOperator>(subOp.getOperation())) {
               auto [a_left, a_right] = normalizeChildren(subOp);
               if (!a.isAssoc(b)) {
                  tes |= calcT(a_left, resolver);
               }
               if (!a.isLAsscom(b)) {
                  tes |= calcT(a_right, resolver);
               }
            } else {
               if (mlir::isa<mlir::relalg::AggregationOp>(subOp.getOperation())) {
                  tes |= calcT(subOp, resolver);
               }
            }
         }
         for (auto subOp : b_right.getAllSubOperators()) {
            if (auto a = mlir::dyn_cast_or_null<BinaryOperator>(subOp.getOperation())) {
               auto [a_left, a_right] = normalizeChildren(subOp);
               if (!b.isAssoc(a)) {
                  tes |= calcT(a_right, resolver);
               }
               if (!b.isRAsscom(a)) {
                  tes |= calcT(a_left, resolver);
               }
            } else {
               if (mlir::isa<mlir::relalg::AggregationOp>(subOp.getOperation())) {
                  tes |= calcT(subOp, resolver);
               }
            }
         }

      } else if (children.size() == 1) {
         auto onlyChild = children[0];
         if (mlir::isa<mlir::relalg::AggregationOp>(op.getOperation())) {
            tes |= calcT(onlyChild, resolver);
         }
         if (auto renameop = mlir::dyn_cast_or_null<mlir::relalg::RenamingOp>(op.getOperation())) {
            for (auto a : onlyChild.getAllSubOperators()) {
               if (a.getUsedAttributes().intersects(renameop.getUsedAttributes()) || a.getCreatedAttributes().intersects(renameop.getCreatedAttributes())) {
                  tes |= calcT(onlyChild, resolver);
               }
            }
         }
      }
      teSs[op.getOperation()] = tes;
      return tes;
   }
}
NodeSet QueryGraphBuilder::calcSES(Operator op, NodeResolver& resolver) const {
   NodeSet res = NodeSet(numNodes);
   for (auto* attr : op.getUsedAttributes()) {
      res.set(resolver.resolve(attr));
   }
   return res;
}

QueryGraphBuilder::QueryGraphBuilder(Operator root, llvm::SmallPtrSet<mlir::Operation*,12>& alreadyOptimized) : root(root),
                                                                                                              alreadyOptimized(alreadyOptimized),
                                                                                                              numNodes(countCreatingOperators(root, alreadyOptimized)),
                                                                                                              qg(numNodes),
                                                                                                              emptyNode(numNodes) {}

} // namespace mlir::relalg