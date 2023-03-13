#ifndef MLIR_DIALECT_RELALG_TRANSFORMS_COLUMNCREATORANALYSIS_H
#define MLIR_DIALECT_RELALG_TRANSFORMS_COLUMNCREATORANALYSIS_H
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
namespace mlir::relalg {
class ColumnCreatorAnalysis {
   std::unordered_map<const mlir::tuples::Column*, Operator> creatorForColumn;

   public:
   Operator getCreator(const mlir::tuples::Column* col) {
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
} // namespace mlir::relalg

#endif //MLIR_DIALECT_RELALG_TRANSFORMS_COLUMNCREATORANALYSIS_H
