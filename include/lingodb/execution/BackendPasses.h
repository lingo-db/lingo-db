#ifndef LINGODB_EXECUTION_BACKENDPASSES_H
#define LINGODB_EXECUTION_BACKENDPASSES_H
#include <memory>
#include <mlir/Pass/Pass.h>
namespace lingodb::execution {
std::unique_ptr<mlir::Pass> createEnforceCABI();
std::unique_ptr<mlir::Pass> createDecomposeTuplePass();
void registerBackendPasses();
} // namespace lingodb::execution
#endif //LINGODB_EXECUTION_BACKENDPASSES_H
