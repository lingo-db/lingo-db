#include "lingodb/compiler/helper.h"
#include "lingodb/utility/Setting.h"

#include <mlir/Transforms/Passes.h>
namespace {
lingodb::utility::GlobalSetting<bool> extraOpt("system.opt.patterns.extra_opt", false);

} // namespace
mlir::LogicalResult lingodb::compiler::applyPatternsGreedily(mlir::Region& region, const mlir::FrozenRewritePatternSet& patterns) {
   if (extraOpt.getValue()) {
      return mlir::applyPatternsGreedily(region, std::move(patterns));
   } else {
      return mlir::applyPatternsGreedily(region, std::move(patterns), mlir::GreedyRewriteConfig{.enableRegionSimplification = mlir::GreedySimplifyRegionLevel::Disabled, .fold = false, .cseConstants = false});
   }
}

std::unique_ptr<mlir::Pass> lingodb::compiler::createCanonicalizerPass() {
   if (extraOpt.getValue()) {
      return mlir::createCanonicalizerPass();
   } else {
      return mlir::createCanonicalizerPass({.enableRegionSimplification = mlir::GreedySimplifyRegionLevel::Disabled});
   }
}