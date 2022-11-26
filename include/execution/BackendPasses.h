#ifndef EXECUTION_BACKENDPASSES_H
#define EXECUTION_BACKENDPASSES_H
#include <memory>
#include <mlir/Pass/Pass.h>
namespace runner {
std::unique_ptr<mlir::Pass> createEnforceCABI();
std::unique_ptr<mlir::Pass> createAnnotateProfilingDataPass();
} // namespace runner
#endif //EXECUTION_BACKENDPASSES_H
