#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "runtime/Database.h"
namespace {
class AttachMetaData : public mlir::PassWrapper<AttachMetaData, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-attach-meta-data"; }
   runtime::Database& db;
   public:
   AttachMetaData(runtime::Database& db):db(db){}
   void runOnOperation() override {
      getOperation().walk([&](mlir::relalg::BaseTableOp op) {
         op.metaAttr(mlir::relalg::TableMetaDataAttr::get(&getContext(),db.getTableMetaData(op.table_identifier().str())));
      });
   }
};
class DetachMetaData : public mlir::PassWrapper<DetachMetaData, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-detach-meta-data"; }

   public:
   void runOnOperation() override {
      getOperation().walk([&](mlir::relalg::BaseTableOp op) {
         getOperation().walk([&](mlir::relalg::BaseTableOp op) {
            op.metaAttr(mlir::relalg::TableMetaDataAttr::get(&getContext(),std::make_shared<runtime::TableMetaData>()));
         });
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createAttachMetaDataPass(runtime::Database& db) { return std::make_unique<AttachMetaData>(db); }
std::unique_ptr<Pass> createDetachMetaDataPass() { return std::make_unique<DetachMetaData>(); }
} // end namespace relalg
} // end namespace mlir