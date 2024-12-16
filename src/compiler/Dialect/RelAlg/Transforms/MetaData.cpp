#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "lingodb/runtime/Catalog.h"

namespace {
using namespace lingodb::compiler::dialect;

class AttachMetaData : public mlir::PassWrapper<AttachMetaData, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-attach-meta-data"; }
   lingodb::runtime::Catalog& catalog;

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AttachMetaData)
   AttachMetaData(lingodb::runtime::Catalog& catalog) : catalog(catalog) {}
   void runOnOperation() override {
      getOperation().walk([&](relalg::BaseTableOp op) {
         auto relation = catalog.findRelation(op.getTableIdentifier().str());
         if (relation) {
            op.setMetaAttr(relalg::TableMetaDataAttr::get(&getContext(), relation->getMetaData()));
         }
      });
   }
};
class DetachMetaData : public mlir::PassWrapper<DetachMetaData, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-detach-meta-data"; }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DetachMetaData)
   void runOnOperation() override {
      getOperation().walk([&](relalg::BaseTableOp op) {
         op.setMetaAttr(relalg::TableMetaDataAttr::get(&getContext(), std::make_shared<lingodb::runtime::TableMetaData>()));
      });
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> relalg::createAttachMetaDataPass(runtime::Catalog& catalog) { return std::make_unique<AttachMetaData>(catalog); }
std::unique_ptr<mlir::Pass> relalg::createDetachMetaDataPass() { return std::make_unique<DetachMetaData>(); }
