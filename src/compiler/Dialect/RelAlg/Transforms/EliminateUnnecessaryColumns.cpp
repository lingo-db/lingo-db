#include "json.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <unordered_set>
namespace {
using namespace lingodb::compiler::dialect;

class EliminateUnnecessaryColumns : public mlir::PassWrapper<EliminateUnnecessaryColumns, mlir::OperationPass<mlir::func::FuncOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EliminateUnnecessaryColumns)
   virtual llvm::StringRef getArgument() const override { return "relalg-eliminate-unnecessary-columns"; }

   void runOnOperation() override {
      llvm::DenseSet<const lingodb::compiler::dialect::tuples::Column*> usedColumns;

      mlir::OpBuilder builder(&getContext());

      getOperation().walk([&](mlir::Operation* o) {
         if (auto op = mlir::dyn_cast<Operator>(o)) {
            for (const auto* c : op.getUsedColumns()) {
               usedColumns.insert(c);
            }
         } else if (auto op = mlir::dyn_cast<relalg::MaterializeOp>(o)) {
            for (auto colAttr : op.getCols()) {
               usedColumns.insert(&mlir::cast<tuples::ColumnRefAttr>(colAttr).getColumn());
            }
         } else if (auto op = mlir::dyn_cast<relalg::GetScalarOp>(o)) {
            usedColumns.insert(&op.getAttr().getColumn());
         }
      });
      getOperation().walk([&](relalg::BaseTableOp baseTableOp) {
         std::vector<mlir::NamedAttribute> mapping;
         for (auto x : baseTableOp.getColumnsAttr()) {
            auto colDef = mlir::cast<tuples::ColumnDefAttr>(x.getValue());
            if (usedColumns.contains(&colDef.getColumn())) {
               mapping.push_back(x);
            } else {
               //colDef.dump();
            }
         }
         baseTableOp.setColumnsAttr(builder.getDictionaryAttr(mapping));
      });
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> relalg::createEliminateUnnecessaryColumnsPass() { return std::make_unique<EliminateUnnecessaryColumns>(); }
