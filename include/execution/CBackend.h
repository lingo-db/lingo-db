#ifndef EXECUTION_CBACKEND_H
#define EXECUTION_CBACKEND_H
#include "Backend.h"
#include "mlir/IR/Operation.h"
namespace execution {
std::unique_ptr<ExecutionBackend> createCBackend();
mlir::LogicalResult emitC(mlir::Operation* op, llvm::raw_ostream& os, bool declareVariablesAtTop);
} // namespace execution
#endif //EXECUTION_CBACKEND_H
