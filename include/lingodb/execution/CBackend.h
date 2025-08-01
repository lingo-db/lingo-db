#ifndef LINGODB_EXECUTION_CBACKEND_H
#define LINGODB_EXECUTION_CBACKEND_H
#include "Backend.h"
#include "mlir/IR/Operation.h"
namespace lingodb::execution {
std::unique_ptr<ExecutionBackend> createCBackend();
mlir::LogicalResult emitC(mlir::Operation* op, llvm::raw_ostream& os, bool declareVariablesAtTop, bool leaveFunctionsUnresolved);
} // namespace lingodb::execution
#endif //LINGODB_EXECUTION_CBACKEND_H
