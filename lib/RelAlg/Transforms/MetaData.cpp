#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "runtime/Catalog.h"
namespace {
class AttachMetaData : public mlir::PassWrapper<AttachMetaData, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-attach-meta-data"; }
   runtime::Catalog& catalog;

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AttachMetaData)
   AttachMetaData(runtime::Catalog& catalog) : catalog(catalog) {}
   void runOnOperation() override {
      getOperation().walk([&](mlir::relalg::BaseTableOp op) {
         auto relation = catalog.findRelation(op.getTableIdentifier().str());
         if (relation) {
            op.setMetaAttr(mlir::relalg::TableMetaDataAttr::get(&getContext(), relation->getMetaData()));
         }
      });
   }
};
class DetachMetaData : public mlir::PassWrapper<DetachMetaData, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-detach-meta-data"; }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DetachMetaData)
   void runOnOperation() override {
      getOperation().walk([&](mlir::relalg::BaseTableOp op) {
         getOperation().walk([&](mlir::relalg::BaseTableOp op) {
            op.setMetaAttr(mlir::relalg::TableMetaDataAttr::get(&getContext(), std::make_shared<runtime::TableMetaData>()));
         });
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createAttachMetaDataPass(runtime::Catalog& catalog) { return std::make_unique<AttachMetaData>(catalog); }
std::unique_ptr<Pass> createDetachMetaDataPass() { return std::make_unique<DetachMetaData>(); }
} // end namespace relalg
} // end namespace mlir