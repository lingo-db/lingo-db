#ifndef LINGODB_COMPILER_DIALECT_RELALG_TRANSFORMS_NULLCOLUMNTYPECHANGING_H
#define LINGODB_COMPILER_DIALECT_RELALG_TRANSFORMS_NULLCOLUMNTYPECHANGING_H
#include "lingodb/compiler/Dialect/TupleStream/Column.h"
namespace lingodb::compiler::dialect::relalg {

struct ColumnNullableChangeInfo {
   llvm::DenseMap<lingodb::compiler::dialect::tuples::Column*, lingodb::compiler::dialect::tuples::Column*> directMappings;
};
} // namespace lingodb::compiler::dialect::relalg

#endif //LINGODB_COMPILER_DIALECT_RELALG_TRANSFORMS_NULLCOLUMNTYPECHANGING_H
