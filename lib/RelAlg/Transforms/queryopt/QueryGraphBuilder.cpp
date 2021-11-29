#include "mlir/Dialect/RelAlg/Transforms/queryopt/QueryGraphBuilder.h"
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

size_t countCreatingOperators(Operator op, llvm::SmallPtrSet<mlir::Operation*, 12>& alreadyOptimized) {
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
      for (const auto* attr : op.getAvailableAttributes()) {
         resolver.add(attr, newNode);
      }
      return resolver;
   }
   for (auto child : children) {
      resolver.merge(populateQueryGraph(child));
   }
   if (mlir::isa<mlir::relalg::CrossProductOp>(op.getOperation())) {
      //do not construct crossproducts in the querygraph
   } else if (mlir::isa<mlir::relalg::SelectionOp>(op.getOperation()) || mlir::isa<mlir::relalg::InnerJoinOp>(op.getOperation())) {
      NodeSet ses = calcSES(op, resolver);
      if (ses.count() == 1) {
         //if selection is only based on one node -> add selection to node
         auto nodeId = ses.findFirst();
         qg.nodes[nodeId].additionalPredicates.push_back(op);
      } else {
         qg.addSelectionEdge(ses, op);
      }
   } else if (mlir::relalg::detail::isJoin(op.getOperation())) {
      //add join edges into the query graph
      NodeSet tes = calcTES(op, resolver);
      NodeSet leftTes = calcT(children[0], resolver) & tes;
      NodeSet rightTes = calcT(children[1], resolver) & tes;
      llvm::Optional<size_t> createdNode;
      if (!created.empty()) {
         size_t newNode = addPseudoNode();
         for (const auto* attr : op.getCreatedAttributes()) {
            resolver.add(attr, newNode);
         }
         createdNode=newNode;
      }
      qg.addJoinEdge(leftTes, rightTes, op,createdNode);

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
   size_t last=0;
   while (alreadyConnected.getNumClasses() > 1) {
      size_t best1 = 0, best2 = 0;
      for (const auto& join : qg.getJoinEdges()) {
         auto leftRep = join.left.findFirst();
         auto rightRep = join.right.findFirst();
         auto class1 = alreadyConnected.getLeaderValue(leftRep);
         auto class2 = alreadyConnected.getLeaderValue(rightRep);
         NodeSet left = getNodeSetFromClass(alreadyConnected, class1, numNodes);
         NodeSet right = getNodeSetFromClass(alreadyConnected, class2, numNodes);
         if (class1 == class2) {
            continue;
         }
         if (join.connects(left, right)) {
            best1 = class1;
            best2 = class2;
            break;
         }
      }
      if (best1 == best2) {
         best1=last;
         for(best2=0;best2<qg.getNodes().size();best2++){
            if(best2!=best1&&alreadyConnected.getLeaderValue(best2)==best2){
               break;
            }
         }
         NodeSet left = getNodeSetFromClass(alreadyConnected, best1, numNodes);
         NodeSet right = getNodeSetFromClass(alreadyConnected, best2, numNodes);
         //construct cross-product
         qg.addJoinEdge(left, right, Operator(),llvm::Optional<size_t>());
      }

      size_t newSet = *alreadyConnected.unionSets(best1, best2);
      last = newSet;
   }
}
NodeSet QueryGraphBuilder::calcTES(Operator op, NodeResolver& resolver) {
   if (teSs.count(op.getOperation())) {
      return teSs[op.getOperation()];
   } else {
      NodeSet tes = calcSES(op, resolver);
      auto children = op.getChildren();
      if (auto b = mlir::dyn_cast_or_null<BinaryOperator>(op.getOperation())) {
         auto bLeft = mlir::cast<Operator>(b.getOperation()).getChildren()[0];
         auto bRight = mlir::cast<Operator>(b.getOperation()).getChildren()[1];

         for (auto subOp : bLeft.getAllSubOperators()) {
            if (auto a = mlir::dyn_cast_or_null<BinaryOperator>(subOp.getOperation())) {
               auto aLeft = subOp.getChildren()[0];
               auto aRight = subOp.getChildren()[1];
               if (!a.isAssoc(b)) {
                  tes |= calcT(aLeft, resolver);
               }
               if (!a.isLAsscom(b)) {
                  tes |= calcT(aRight, resolver);
               }
            }
         }
         for (auto subOp : bRight.getAllSubOperators()) {
            if (auto a = mlir::dyn_cast_or_null<BinaryOperator>(subOp.getOperation())) {
               auto aLeft = subOp.getChildren()[0];
               auto aRight = subOp.getChildren()[1];
               if (!b.isAssoc(a)) {
                  tes |= calcT(aRight, resolver);
               }
               if (!b.isRAsscom(a)) {
                  tes |= calcT(aLeft, resolver);
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
   for (const auto* attr : op.getUsedAttributes()) {
      res.set(resolver.resolve(attr));
   }
   return res;
}

QueryGraphBuilder::QueryGraphBuilder(Operator root, llvm::SmallPtrSet<mlir::Operation*, 12>& alreadyOptimized) : root(root),
                                                                                                                 alreadyOptimized(alreadyOptimized),
                                                                                                                 numNodes(countCreatingOperators(root, alreadyOptimized)),
                                                                                                                 qg(numNodes) {}

} // namespace mlir::relalg