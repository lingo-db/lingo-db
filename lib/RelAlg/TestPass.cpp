
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <iostream>
#include <list>
#include <queue>
#include <unordered_set>

namespace {
// query optimization pass
class TestPass : public mlir::PassWrapper<TestPass, mlir::FunctionPass> {

public:
  void runOnFunction() override {
  }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createTestPass() { return std::make_unique<TestPass>(); }
} // end namespace relalg
} // end namespace mlir