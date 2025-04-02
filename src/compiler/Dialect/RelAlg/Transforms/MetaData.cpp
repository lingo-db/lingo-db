#include "lingodb/catalog/Catalog.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"

#include <lingodb/catalog/TableCatalogEntry.h>

namespace {
using namespace lingodb::compiler::dialect;

class AttachMetaData : public mlir::PassWrapper<AttachMetaData, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-attach-meta-data"; }
   lingodb::catalog::Catalog& catalog;

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AttachMetaData)
   AttachMetaData(lingodb::catalog::Catalog& catalog) : catalog(catalog) {}
   void runOnOperation() override {
      getOperation().walk([&](relalg::BaseTableOp op) {
         if (auto relation = catalog.getTypedEntry<lingodb::catalog::TableCatalogEntry>(op.getTableIdentifier().str())) {
            op->setAttr("meta", relalg::TableMetaDataAttr::get(&getContext(), std::dynamic_pointer_cast<lingodb::catalog::TableMetaDataProvider>(relation.value())));
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
         op->removeAttr("meta");
      });
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> relalg::createAttachMetaDataPass(catalog::Catalog& catalog) { return std::make_unique<AttachMetaData>(catalog); }
std::unique_ptr<mlir::Pass> relalg::createDetachMetaDataPass() { return std::make_unique<DetachMetaData>(); }
