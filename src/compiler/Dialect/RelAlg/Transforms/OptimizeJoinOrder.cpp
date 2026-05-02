#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "lingodb/compiler/Dialect/RelAlg/Transforms/queryopt/DPhyp.h"
#include "lingodb/compiler/Dialect/RelAlg/Transforms/queryopt/GOO.h"
#include "lingodb/compiler/Dialect/RelAlg/Transforms/queryopt/QueryGraphBuilder.h"

namespace {
using namespace lingodb::compiler::dialect;

class OptimizeJoinOrder : public mlir::PassWrapper<OptimizeJoinOrder, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-optimize-join-order"; }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizeJoinOrder)
   private:
   llvm::SmallPtrSet<mlir::Operation*, 12> alreadyOptimized;

   bool isUnsupportedOp(mlir::Operation* op) {
      return ::llvm::TypeSwitch<mlir::Operation*, bool>(op)
         .Case<relalg::CrossProductOp, relalg::SelectionOp>(
            [&](mlir::Operation* op) {
               return false;
            })
         .Case<BinaryOperator>([&](mlir::Operation* op) {
            if (relalg::detail::isJoin(op)) {
               Operator asOperator = mlir::cast<Operator>(op);
               auto subOps = asOperator.getAllSubOperators();
               auto used = asOperator.getUsedColumns();
               relalg::AvailabilityCache cache;
               return !(used.intersects(subOps[0].getAvailableColumns(cache)) && used.intersects(subOps[1].getAvailableColumns(cache)));
            } else {
               return true;
            }
         })
         .Default([&](auto x) {
            return true;
         });
   }

   bool isOptimizationRoot(mlir::Operation* op) {
      //reason one: used by multiple parent operators (DAG)
      auto users = op->getUsers();
      if (!users.empty() && ++users.begin() != users.end()) {
         return true;
      }
      //reason two: result of operation is accessed by non-operator
      if (llvm::any_of(op->getUsers(), [](mlir::OpOperand user) { return !mlir::isa<Operator>(user.getOwner()); })) {
         return true;
      }
      return isUnsupportedOp(op);
   }

   void estimateUnsupported(Operator op) {
      auto children = op.getChildren();
      llvm::TypeSwitch<mlir::Operation*>(op.getOperation())
         .Case<relalg::MapOp>([&](relalg::MapOp mapOp) {
            if (children.size() == 1 && children[0]->hasAttr("rows")) {
               mapOp->setAttr("rows", children[0]->getAttr("rows"));
            }
         })
         .Case<relalg::LimitOp>([&](relalg::LimitOp limitOp) {
            if (children.size() == 1 && children[0]->hasAttr("rows")) {
               limitOp->setAttr("rows", mlir::FloatAttr::get(mlir::Float64Type::get(&getContext()), std::min(mlir::cast<mlir::FloatAttr>(children[0]->getAttr("rows")).getValueAsDouble(), (double) limitOp.getMaxRows())));
            }
         })
         .Case<relalg::BaseTableOp>([&](relalg::BaseTableOp baseTableOp) {
            relalg::annotateBaseTable(baseTableOp);
         })
         .Case<relalg::UnionOp>([&](relalg::UnionOp unionOp) {
            double sum = 0;
            for (auto child : children) {
               if (child->hasAttr("rows")) {
                  sum += mlir::cast<mlir::FloatAttr>(child->getAttr("rows")).getValueAsDouble();
               }
            }
            if (sum != 0) {
               unionOp->setAttr("rows", mlir::FloatAttr::get(mlir::Float64Type::get(&getContext()), sum));
            }
         })
         .Case<relalg::AggregationOp>([&](relalg::AggregationOp aggregationOp) {
            if (aggregationOp.getGroupByCols().empty()) {
               aggregationOp->setAttr("rows", mlir::FloatAttr::get(mlir::Float64Type::get(&getContext()), 1.0));
            } else if (children.size() == 1 && children[0]->hasAttr("rows")) {
               aggregationOp->setAttr("rows", children[0]->getAttr("rows"));
            }
         })
         .Case<UnaryOperator>([&](UnaryOperator unaryOperator) {
            if (children.size() == 1 && children[0]->hasAttr("rows")) {
               unaryOperator->setAttr("rows", children[0]->getAttr("rows"));
            }
         });
   }
   Operator optimize(Operator op) {
      if (alreadyOptimized.count(op.getOperation())) {
         return op;
      }
      if (isUnsupportedOp(op)) {
         auto children = op.getChildren();
         for (size_t i = 0; i < children.size(); i++) {
            children[i] = optimize(children[i]);
         }
         op.setChildren(children);
         estimateUnsupported(op);
         alreadyOptimized.insert(op.getOperation());
         return op;
      } else {
         relalg::QueryGraphBuilder queryGraphBuilder(op, alreadyOptimized);
         queryGraphBuilder.generate();
         relalg::QueryGraph& queryGraph = queryGraphBuilder.getQueryGraph();
         queryGraph.estimate();

         std::shared_ptr<relalg::Plan> solution;
         relalg::DPHyp solver(queryGraph);
         if (solver.countSubGraphs(1000) < 1000) {
            solution = solver.solve();
         } else {
            relalg::GOO fallBackSolver(queryGraph);
            solution = fallBackSolver.solve();
         }
         if (!solution) {
            llvm::dbgs() << "no valid join order found\n";
            return op;
         }

         llvm::SmallVector<Operator, 4> before = op.getAllSubOperators();
         llvm::SmallPtrSet<mlir::Operation*, 8> prevUsers{op->user_begin(), op->user_end()};

         llvm::DenseMap<mlir::Block*, llvm::DenseMap<mlir::Operation*, int>> blockPositions;
         llvm::DenseMap<mlir::Block*, llvm::SmallVector<mlir::Operation*, 4>> leavesByBlock;
         llvm::DenseMap<mlir::Operation*, mlir::Operation*> prevNodeMap;
         llvm::SmallPtrSet<mlir::Operation*, 16> seenLeaves;

         for (auto subOp : before) {
            mlir::Operation* subOpNode = subOp.getOperation();
            if (alreadyOptimized.count(subOpNode) && seenLeaves.insert(subOpNode).second) {
               mlir::Block* b = subOpNode->getBlock();
               if (!blockPositions.count(b)) {
                  int pos = 0;
                  for (auto& blockOp : *b) {
                     blockPositions[b][&blockOp] = pos++;
                  }
               }
               leavesByBlock[b].push_back(subOpNode);
               prevNodeMap[subOpNode] = subOpNode->getPrevNode();
            }
         }

         Operator realized = solution->realizePlan();

         // 1. Restore leaves to their original blocks and exact positions
         for (auto& pair : leavesByBlock) {
            mlir::Block* b = pair.first;
            auto& leaves = pair.second;

            std::sort(leaves.begin(), leaves.end(), [&](mlir::Operation* a, mlir::Operation* bOp) {
               return blockPositions[b][a] < blockPositions[b][bOp];
            });
            for (auto leaf : leaves) {
               mlir::Operation* pNode = prevNodeMap[leaf];
               if (pNode && pNode->getBlock() == b) {
                  leaf->moveAfter(pNode);
               } else {
                  leaf->moveBefore(&b->front());
               }
            }
         }

         llvm::SmallVector<Operator, 4> after = realized.getAllSubOperators();
         mlir::Block* opBlock = op->getBlock();

         // 2. Shepherd ALL new/reused internal operators to the root op's block to prevent cross-region dominance faults.
         for (auto aop : after) {
            mlir::Operation* aopNode = aop.getOperation();
            if (!seenLeaves.contains(aopNode) && aopNode != op.getOperation()) {
               aopNode->moveBefore(op.getOperation());
            }
         }

         // 3. Block-local robust topological sort using region walking
         bool dominanceChanged = true;
         while (dominanceChanged) {
            dominanceChanged = false;
            for (mlir::Operation& blockOp : *opBlock) {
               mlir::Operation* opPtr = &blockOp;
               mlir::Operation* moveAfterTarget = nullptr;

               opPtr->walk([&](mlir::Operation* nestedOp) {
                  for (mlir::Value operand : nestedOp->getOperands()) {
                     mlir::Operation* defOp = operand.getDefiningOp();
                     if (defOp && defOp->getBlock() == opBlock && opPtr->isBeforeInBlock(defOp)) {
                        if (!moveAfterTarget || moveAfterTarget->isBeforeInBlock(defOp)) {
                           moveAfterTarget = defOp;
                        }
                     }
                  }
               });

               if (moveAfterTarget) {
                  opPtr->moveAfter(moveAfterTarget);
                  dominanceChanged = true;
                  break;
               }
            }
         }

         if (realized != op) {
            op->getResult(0).replaceUsesWithIf(realized->getResult(0), [prevUsers](mlir::OpOperand& operand) {
               return prevUsers.contains(operand.getOwner());
            });
         }

         llvm::SmallPtrSet<mlir::Operation*, 8> afterHt;
         for (auto aop : after) {
            afterHt.insert(aop.getOperation());
         }

         llvm::SmallVector<mlir::Operation*, 8> toDelete;
         for (auto bop : before) {
            if (!afterHt.contains(bop.getOperation())) {
               toDelete.push_back(bop.getOperation());
            }
         }

         bool changed = true;
         while (changed) {
            changed = false;
            for (auto it = toDelete.begin(); it != toDelete.end();) {
               if ((*it)->use_empty()) {
                  (*it)->erase();
                  it = toDelete.erase(it);
                  changed = true;
               } else {
                  ++it;
               }
            }
         }

         alreadyOptimized.insert(realized);
         return realized;
      }
   }

   void runOnOperation() override {
      //walk over all operators:
      getOperation()->walk([&](Operator op) {
         //check if current operator is root for join order optimization
         if (isOptimizationRoot(op.getOperation())) {
            //if so: optimize subtree
            optimize(op);
         }
      });
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> relalg::createOptimizeJoinOrderPass() { return std::make_unique<OptimizeJoinOrder>(); }
