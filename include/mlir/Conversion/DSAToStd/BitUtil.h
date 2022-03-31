#ifndef MLIR_CONVERSION_DSATOSTD_BITUTIL_H
#define MLIR_CONVERSION_DSATOSTD_BITUTIL_H
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/util/UtilOps.h"

namespace mlir::dsa::codegen {
class BitUtil {
   public:
   static Value getBit(OpBuilder builder, Location loc, Value bits, Value pos) {
      auto i1Type = IntegerType::get(builder.getContext(), 1);
      auto i8Type = IntegerType::get(builder.getContext(), 8);

      auto indexType = IndexType::get(builder.getContext());
      Value const3 = builder.create<arith::ConstantOp>(loc, indexType, builder.getIntegerAttr(indexType, 3));
      Value const7 = builder.create<arith::ConstantOp>(loc, indexType, builder.getIntegerAttr(indexType, 7));
      Value const1Byte = builder.create<arith::ConstantOp>(loc, i8Type, builder.getIntegerAttr(i8Type, 1));

      Value div8 = builder.create<arith::ShRUIOp>(loc, indexType, pos, const3);
      Value rem8 = builder.create<arith::AndIOp>(loc, indexType, pos, const7);
      Value loadedByte = builder.create<mlir::util::LoadOp>(loc, i8Type, bits, div8);
      Value rem8AsByte = builder.create<arith::IndexCastOp>(loc, i8Type, rem8);
      Value shifted = builder.create<arith::ShRUIOp>(loc, i8Type, loadedByte, rem8AsByte);
      Value res1 = shifted;

      Value anded = builder.create<arith::AndIOp>(loc, i8Type, res1, const1Byte);
      Value res = builder.create<arith::CmpIOp>(loc, i1Type, mlir::arith::CmpIPredicate::eq, anded, const1Byte);
      return res;
   }
};
} // namespace mlir::dsa::codegen
#endif // MLIR_CONVERSION_DSATOSTD_BITUTIL_H
