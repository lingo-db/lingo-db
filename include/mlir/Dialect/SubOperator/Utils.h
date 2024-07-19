#ifndef MLIR_DIALECT_SUBOPERATOR_UTILS_H
#define MLIR_DIALECT_SUBOPERATOR_UTILS_H
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include <unordered_map>
namespace mlir::subop {
class MapCreationHelper {
   std::unordered_map<mlir::tuples::Column*, size_t> columnToIndex;
   std::vector<mlir::Attribute> colRefs;
   mlir::Block* mapBlock;
   mlir::MLIRContext* context;

   public:
   MapCreationHelper(mlir::MLIRContext* context) : context(context) {
      mapBlock = new mlir::Block;
   }
   mlir::Value access(mlir::tuples::ColumnRefAttr columnRefAttr, mlir::Location loc) {
      auto *keyColumn = &columnRefAttr.getColumn();
      if (columnToIndex.contains(keyColumn)) {
         return mapBlock->getArgument(columnToIndex[keyColumn]);
      } else {
         auto arg = mapBlock->addArgument(keyColumn->type, loc);
         columnToIndex[keyColumn] = columnToIndex.size();
         colRefs.push_back(columnRefAttr);
         return arg;
      }
   }
   mlir::ArrayAttr getColRefs() {
      return mlir::ArrayAttr::get(context, colRefs);
   }
   template <class B, class F>
   void buildBlock(B& builder, const F& f) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(mapBlock);
      f(builder);
   }
   mlir::Block* getMapBlock() {
      return mapBlock;
   }
};
} //end namespace mlir::subop

#endif //MLIR_DIALECT_SUBOPERATOR_UTILS_H
