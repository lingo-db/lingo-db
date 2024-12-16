#ifndef LINGODB_EXECUTION_INSTRUMENTATION_H
#define LINGODB_EXECUTION_INSTRUMENTATION_H
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include <memory>
#include <string>
#include <vector>
namespace mlir {
class PassInstrumentation;
} // end namespace mlir
namespace lingodb::execution {
struct PassData {
   std::string fileName;
   std::string passName;
   std::string description;
   std::string argument;
};
struct SnapshotState {
   bool serialize = false;
   size_t counter = 0;
   std::vector<PassData> passes;
};
std::string getSnapshotDir();
void addLingoDBInstrumentation(mlir::PassManager& pm, std::shared_ptr<SnapshotState> serializationState);
void snapshotImportantStep(std::string shortName, mlir::ModuleOp& moduleOp, std::shared_ptr<SnapshotState> serializationState);
} // end namespace lingodb::execution
#endif //LINGODB_EXECUTION_INSTRUMENTATION_H
