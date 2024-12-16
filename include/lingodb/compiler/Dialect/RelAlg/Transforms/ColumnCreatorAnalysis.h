#ifndef LINGODB_COMPILER_DIALECT_RELALG_TRANSFORMS_COLUMNCREATORANALYSIS_H
#define LINGODB_COMPILER_DIALECT_RELALG_TRANSFORMS_COLUMNCREATORANALYSIS_H
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
namespace lingodb::compiler::dialect::relalg {
class ColumnCreatorAnalysis {
   std::unordered_map<const lingodb::compiler::dialect::tuples::Column*, Operator> creatorForColumn;

   public:
   Operator getCreator(const lingodb::compiler::dialect::tuples::Column* col) {
      return creatorForColumn.at(col);
   }
   ColumnCreatorAnalysis(mlir::Operation* op) {
      op->walk([&](Operator o) {
         for (const auto* c : o.getCreatedColumns()) {
            creatorForColumn.insert({c, o});
         }
      });
   }
};
} // namespace lingodb::compiler::dialect::relalg

#endif //LINGODB_COMPILER_DIALECT_RELALG_TRANSFORMS_COLUMNCREATORANALYSIS_H
