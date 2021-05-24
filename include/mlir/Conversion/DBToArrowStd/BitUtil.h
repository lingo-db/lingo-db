#ifndef MLIR_CONVERSION_DBTOARROWSTD_BITUTIL_H
#define MLIR_CONVERSION_DBTOARROWSTD_BITUTIL_H
#include "FunctionRegistry.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir::db::codegen {
class BitUtil {
   public:
   static Value getBit(OpBuilder builder, Value bits, Value pos, bool negated = false) {
      auto i1Type = IntegerType::get(builder.getContext(), 1);
      auto i8Type = IntegerType::get(builder.getContext(), 8);

      auto indexType = IndexType::get(builder.getContext());
      Value const3 = builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), indexType, builder.getIntegerAttr(indexType, 3));
      Value const7 = builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), indexType, builder.getIntegerAttr(indexType, 7));
      Value const1Byte = builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), i8Type, builder.getIntegerAttr(i8Type, 1));

      Value div8 = builder.create<UnsignedShiftRightOp>(builder.getUnknownLoc(), indexType, pos, const3);
      Value rem8 = builder.create<mlir::AndOp>(builder.getUnknownLoc(), indexType, pos, const7);
      Value loadedByte = builder.create<memref::LoadOp>(builder.getUnknownLoc(), bits, ValueRange({div8}));
      Value rem8AsByte = builder.create<IndexCastOp>(builder.getUnknownLoc(), rem8, i8Type);
      Value shifted = builder.create<UnsignedShiftRightOp>(builder.getUnknownLoc(), i8Type, loadedByte, rem8AsByte);
      Value res1 = shifted;
      if (negated) {
         Value constTrue = builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), i8Type, builder.getIntegerAttr(i8Type, 1));
         res1 = builder.create<XOrOp>(builder.getUnknownLoc(), res1, constTrue); //negate
      }
      Value anded = builder.create<mlir::AndOp>(builder.getUnknownLoc(), i8Type, res1, const1Byte);
      Value res = builder.create<TruncateIOp>(builder.getUnknownLoc(), i1Type, anded);
      return res;
   }
};
} // namespace mlir::db::codegen
#endif // MLIR_CONVERSION_DBTOARROWSTD_BITUTIL_H
