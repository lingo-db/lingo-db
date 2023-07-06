#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
class IntroduceTmp : public mlir::PassWrapper<IntroduceTmp, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-introduce-tmp"; }
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IntroduceTmp)
   mlir::relalg::ColumnSet getUsed(mlir::Operation* op) {
      if (auto asOperator = mlir::dyn_cast_or_null<Operator>(op)) {
         auto cols = asOperator.getUsedColumns();
         for (auto* user : asOperator.asRelation().getUsers()) {
            cols.insert(getUsed(user));
         }
         return cols;
      } else if (auto matOp = mlir::dyn_cast_or_null<mlir::relalg::MaterializeOp>(op)) {
         return mlir::relalg::ColumnSet::fromArrayAttr(matOp.getCols());
      }
      return {};
   }
   void runOnOperation() override {
      getOperation().walk([&](Operator op) {
         if (!op->use_empty() && !op->hasOneUse()) {
            mlir::OpBuilder builder(&getContext());
            builder.setInsertionPointAfter(op.getOperation());
            mlir::relalg::ColumnSet usedAttributes;
            for (auto& use : op->getUses()) {
               usedAttributes.insert(getUsed(use.getOwner()));
            }
            usedAttributes = usedAttributes.intersect(op.getAvailableColumns());
            mlir::Type tupleStreamType = op.asRelation().getType();
            std::vector<mlir::Type> resultingTypes;
            for (auto it = op->getUses().begin(); it != op->getUses().end(); it++)
               resultingTypes.push_back(tupleStreamType);
            auto tmp = builder.create<mlir::relalg::TmpOp>(op->getLoc(), resultingTypes, op.asRelation(), usedAttributes.asRefArrayAttr(&getContext()));
            size_t i = 0;
            for (auto& use : llvm::make_early_inc_range(op->getUses()))
               if (use.getOwner() != tmp)
                  use.set(tmp.getResult(i++));
         }
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createIntroduceTmpPass() { return std::make_unique<IntroduceTmp>(); }
} // end namespace relalg
} // end namespace mlir