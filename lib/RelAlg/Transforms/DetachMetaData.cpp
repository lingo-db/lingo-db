#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
class DetachMetaData : public mlir::PassWrapper<DetachMetaData, mlir::FunctionPass> {
   virtual llvm::StringRef getArgument() const override { return "relalg-detach-meta-data"; }

   public:


   void runOnFunction() override {
      getFunction().walk([&](mlir::relalg::BaseTableOp op) {
         getFunction().walk([&](mlir::relalg::BaseTableOp op) {
            op.metaAttr(mlir::relalg::TableMetaDataAttr::get(&getContext(),std::make_shared<runtime::TableMetaData>()));
         });
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createDetachMetaDataPass() { return std::make_unique<DetachMetaData>(); }
} // end namespace relalg
} // end namespace mlir