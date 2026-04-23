#include "lingodb/compiler/Dialect/RelAlg/Transforms/ColumnFolding.h"

#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"

#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include <unordered_set>
namespace {
using namespace lingodb::compiler::dialect;
class ColumnFoldingPass : public mlir::PassWrapper<ColumnFoldingPass, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-fold-columns"; }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ColumnFoldingPass)

   void runOnOperation() override {
      std::unordered_set<mlir::Operation*> alreadyHandled;
      getOperation()->walk([&](ColumnFoldable columnFoldable) {
         if (!alreadyHandled.contains(columnFoldable.getOperation())) {
            relalg::ColumnFoldInfo columnFoldInfo;
            ColumnFoldable current = columnFoldable;
            while (current) {
               alreadyHandled.insert(current.getOperation());
               if (current.foldColumns(columnFoldInfo).failed()) {
                  break;
               }
               if (current->getNumResults() != 1)
                  break;
               auto next = current->getResult(0);
               if (!mlir::isa<tuples::TupleStreamType>(next.getType())) {
                  break;
               }
               if (!next.hasOneUse()) {
                  break;
               }
               current = mlir::dyn_cast_or_null<ColumnFoldable>(*next.getUsers().begin());
            }
         }
      });
      while (true) {
         relalg::ColumnSet usedColumns;
         getOperation()->walk([&](Operator op) {
            usedColumns.insert(op.getUsedColumns());
         });
         getOperation()->walk([&](relalg::MaterializeOp op) {
            usedColumns.insert(relalg::ColumnSet::fromArrayAttr(op.getCols()));
         });

         getOperation()->walk([&](mlir::Operation* op) {
            if (mlir::isa<Operator>(op) || mlir::isa<relalg::MaterializeOp>(op)) return;

            for (auto namedAttr : op->getAttrs()) {
               auto attr = namedAttr.getValue();
               if (auto colRef = mlir::dyn_cast<tuples::ColumnRefAttr>(attr)) {
                  usedColumns.insert(&colRef.getColumn());
               } else if (auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
                  for (auto a : arrayAttr) {
                     if (auto colRef = mlir::dyn_cast<tuples::ColumnRefAttr>(a)) {
                        usedColumns.insert(&colRef.getColumn());
                     }
                  }
               } else if (auto mappingAttr = mlir::dyn_cast<subop::ColumnRefMemberMappingAttr>(attr)) {
                  for (auto& pair : mappingAttr.getMapping()) {
                     usedColumns.insert(&pair.second.getColumn());
                  }
               }
            }
         });

         bool existsSucceed = false;
         getOperation()->walk([&](ColumnFoldable columnFoldable) {
            if (columnFoldable->getNumResults() != 1) {
               return;
            }
            mlir::Value v = columnFoldable->getResult(0);
            if (!mlir::isa<tuples::TupleStreamType>(v.getType())) {
               return;
            }
            if (columnFoldable.eliminateDeadColumns(usedColumns, v).succeeded()) {
               existsSucceed = true;
               if (v != columnFoldable->getResult(0)) {
                  columnFoldable->getResult(0).replaceAllUsesWith(v);
               }
            }
         });
         if (!existsSucceed) {
            break;
         }
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> relalg::createColumnFoldingPass() { return std::make_unique<ColumnFoldingPass>(); } // NOLINT(misc-use-internal-linkage)