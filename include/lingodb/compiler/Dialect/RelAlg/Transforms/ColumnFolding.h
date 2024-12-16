#ifndef LINGODB_COMPILER_DIALECT_RELALG_TRANSFORMS_COLUMNFOLDING_H
#define LINGODB_COMPILER_DIALECT_RELALG_TRANSFORMS_COLUMNFOLDING_H
#include "lingodb/compiler/Dialect/TupleStream/Column.h"
namespace lingodb::compiler::dialect::relalg {

struct ColumnFoldInfo {
   std::unordered_map<lingodb::compiler::dialect::tuples::Column*, lingodb::compiler::dialect::tuples::Column*> directMappings;
};
} // namespace lingodb::compiler::dialect::relalg

#endif //LINGODB_COMPILER_DIALECT_RELALG_TRANSFORMS_COLUMNFOLDING_H
