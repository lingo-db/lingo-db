#ifndef MLIR_DIALECT_RELALG_IR_RELALGDIALECT_H
#define MLIR_DIALECT_RELALG_IR_RELALGDIALECT_H
#include <memory>
namespace llvm {
class hash_code;// NOLINT (readability-identifier-naming)
template <typename T>
llvm::hash_code hash_value(std::shared_ptr<T> arg);// NOLINT (readability-identifier-naming)
} // end namespace llvm
#include "mlir/Dialect/RelAlg/IR/ColumnManager.h"
#include "mlir/IR/Dialect.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOpsDialect.h.inc"

namespace llvm {
template <typename T>
llvm::hash_code hash_value(std::shared_ptr<T> arg) {// NOLINT (readability-identifier-naming)
   return hash_value(arg.get());
}
} // end namespace llvm
#endif // MLIR_DIALECT_RELALG_IR_RELALGDIALECT_H
