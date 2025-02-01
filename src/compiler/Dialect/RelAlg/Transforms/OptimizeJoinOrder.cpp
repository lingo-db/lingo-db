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
               return !(used.intersects(subOps[0].getAvailableColumns()) && used.intersects(subOps[1].getAvailableColumns()));
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
      if (llvm::any_of(op->getUsers(),
                       [](mlir::OpOperand user) { return !mlir::isa<Operator>(user.getOwner()); })) {
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
            auto numRows = baseTableOp.getMeta().getMeta()->getNumRows();
            baseTableOp->setAttr("rows", mlir::FloatAttr::get(mlir::Float64Type::get(&getContext()), numRows));
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
               aggregationOp->setAttr("rows", mlir::FloatAttr::get(mlir::Float64Type::get(&getContext()),1.0));
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
         //don't do anything, subtree is already optimized
         return op;
      }
      if (isUnsupportedOp(op)) {
         //unsupported optimization -> try to optimize subtree(s) below
         auto children = op.getChildren();
         for (size_t i = 0; i < children.size(); i++) {
            children[i] = optimize(children[i]);
         }
         op.setChildren(children);
         estimateUnsupported(op);
         alreadyOptimized.insert(op.getOperation());
         return op;
      } else {
         //generate querygraph
         relalg::QueryGraphBuilder queryGraphBuilder(op, alreadyOptimized);
         queryGraphBuilder.generate();
         relalg::QueryGraph& queryGraph = queryGraphBuilder.getQueryGraph();
         queryGraph.estimate();
         //queryGraph.dump();
         //enumerates possible plans and find best one
         std::shared_ptr<relalg::Plan> solution;
         relalg::DPHyp solver(queryGraph);
         if (solver.countSubGraphs(1000) < 1000) {
            solution = solver.solve();
         } else {
            relalg::GOO fallBackSolver(queryGraph);
            solution = fallBackSolver.solve();
         }
         if (!solution) {
            //no solution was found
            llvm::dbgs() << "no valid join order found\n";
            return op;
         }
         //now: realize found plan
         //first: collect all operators that are reachable now
         llvm::SmallVector<Operator, 4> before = op.getAllSubOperators();
         llvm::SmallPtrSet<mlir::Operation*, 8> prevUsers{op->user_begin(), op->user_end()};
         //realize plan
         Operator realized = solution->realizePlan();
         //second: collect all operators that are reachable now
         llvm::SmallVector<Operator, 4> after = realized.getAllSubOperators();

         //maybe op is now somewhere deep in the subtree -> replace all "relevant" uses with the new subtree root
         if (realized != op) {
            op->getResult(0).replaceUsesWithIf(realized->getResult(0), [prevUsers](mlir::OpOperand& operand) {
               return prevUsers.contains(operand.getOwner());
            });
         }
         //cleanup: make sure that all operators that are not reachable any more are cleaned up
         llvm::SmallPtrSet<mlir::Operation*, 8> afterHt;
         for (auto op : after) {
            afterHt.insert(op.getOperation());
         }
         for (auto op : before) {
            if (!afterHt.contains(op.getOperation())) {
               op->dropAllUses();
               op->dropAllReferences();
               op->remove();
               //op->destroy();
            }
         }
         //mark subtree as already optimized
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
