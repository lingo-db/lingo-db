#ifndef LINGODB_COMPILER_DIALECT_TUPLESTREAM_TUPLESTREAMDIALECT_H
#define LINGODB_COMPILER_DIALECT_TUPLESTREAM_TUPLESTREAMDIALECT_H
#include <memory>

namespace llvm {
class hash_code; // NOLINT (readability-identifier-naming)
template <typename T>
llvm::hash_code hash_value(std::shared_ptr<T> arg); // NOLINT (readability-identifier-naming)
} // end namespace llvm
#include "lingodb/compiler/Dialect/TupleStream/ColumnManager.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOpsDialect.h.inc"
#include "mlir/IR/Dialect.h"

#ifndef MLIR_HASHCODE_SHARED_PTR
#define MLIR_HASHCODE_SHARED_PTR
namespace llvm {
template <typename T>
llvm::hash_code hash_value(std::shared_ptr<T> arg) { // NOLINT (readability-identifier-naming)
   return hash_value(arg.get());
}
} // end namespace llvm
#endif // MLIR_HASHCODE_SHARED_PTR
#endif //LINGODB_COMPILER_DIALECT_TUPLESTREAM_TUPLESTREAMDIALECT_H
