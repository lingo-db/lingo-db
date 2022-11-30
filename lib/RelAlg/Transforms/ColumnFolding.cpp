#include "mlir/Dialect/RelAlg/Transforms/ColumnFolding.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h"
#include "mlir/Pass/Pass.h"
#include <unordered_set>
namespace {
class ColumnFoldingPass : public mlir::PassWrapper<ColumnFoldingPass, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-fold-columns"; }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ColumnFoldingPass)

   void runOnOperation() override {
      std::unordered_set<mlir::Operation*> alreadyHandled;
      getOperation()->walk([&](ColumnFoldable columnFoldable) {
         if (!alreadyHandled.contains(columnFoldable.getOperation())) {

            mlir::relalg::ColumnFoldInfo columnFoldInfo;
            ColumnFoldable current = columnFoldable;
            while (current) {
               alreadyHandled.insert(current.getOperation());
               if (current.foldColumns(columnFoldInfo).failed()) {
                  break;
               }
               if (current->getNumResults() != 1)
                  break;
               auto next = current->getResult(0);
               if (!next.getType().isa<mlir::tuples::TupleStreamType>()) {
                  break;
               }
               if (!next.hasOneUse()) {
                  break;
               }
               current = mlir::dyn_cast_or_null<ColumnFoldable>(*next.getUsers().begin());
            }
         }
      });
      mlir::relalg::ColumnSet usedColumns;
      getOperation()->walk([&](Operator op) {
         usedColumns.insert(op.getUsedColumns());
      });
      getOperation()->walk([&](mlir::relalg::MaterializeOp op) {
         usedColumns.insert(mlir::relalg::ColumnSet::fromArrayAttr(op.getCols()));
      });
      getOperation()->walk([&](ColumnFoldable columnFoldable) {
         if (columnFoldable->getNumResults() != 1) {
            return;
         }
         mlir::Value v = columnFoldable->getResult(0);
         if (!v.getType().isa<mlir::tuples::TupleStreamType>()) {
            return;
         }
         if (columnFoldable.eliminateDeadColumns(usedColumns, v).succeeded()) {
            if (v != columnFoldable->getResult(0)) {
               columnFoldable->getResult(0).replaceAllUsesWith(v);
            }
         }
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<mlir::Pass> createColumnFoldingPass() { return std::make_unique<ColumnFoldingPass>(); }
} // end namespace relalg
} // end namespace mlir