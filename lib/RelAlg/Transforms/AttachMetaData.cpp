#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "runtime/database.h"
namespace {
class AttachMetaData : public mlir::PassWrapper<AttachMetaData, mlir::FunctionPass> {
   virtual llvm::StringRef getArgument() const override { return "relalg-attach-meta-data"; }
   runtime::Database& db;
   public:
   AttachMetaData(runtime::Database& db):db(db){}


   void runOnFunction() override {
      getFunction().walk([&](mlir::relalg::BaseTableOp op) {
         op.metaAttr(mlir::relalg::TableMetaDataAttr::get(&getContext(),db.getTableMetaData(op.table_identifier().str())));
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createAttachMetaDataPass(runtime::Database& db) { return std::make_unique<AttachMetaData>(db); }
} // end namespace relalg
} // end namespace mlir