#ifndef MLIR_DIALECT_RELALG_IR_RELALGDIALECT_H
#define MLIR_DIALECT_RELALG_IR_RELALGDIALECT_H
#include<memory>
namespace llvm {
class hash_code;
template <typename T>
llvm::hash_code hash_value(std::shared_ptr<T> arg);
}
#include "mlir/Dialect/RelAlg/IR/RelationalAttributeManager.h"
#include "mlir/IR/Dialect.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOpsDialect.h.inc"


namespace llvm {
template <typename T>
llvm::hash_code hash_value(std::shared_ptr<T> arg) {
   return hash_value(arg.get());
}
}
#endif // MLIR_DIALECT_RELALG_IR_RELALGDIALECT_H
