#include "mlir/Conversion/RelAlgToDB/RelAlgToDBPass.h"


namespace {
class LowerToDBPass : public mlir::PassWrapper<LowerToDBPass, mlir::FunctionPass> {

   void runOnFunction() override {
      getFunction().walk([&](mlir::Operation* op) {
         //TODO
      });

   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createLowerToDBPass() { return std::make_unique<LowerToDBPass>(); }
} // end namespace relalg
} // end namespace mlir