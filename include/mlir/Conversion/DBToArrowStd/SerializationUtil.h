#ifndef MLIR_CONVERSION_DBTOARROWSTD_SERIALIZATIONUTIL_H
#define MLIR_CONVERSION_DBTOARROWSTD_SERIALIZATIONUTIL_H

#include "mlir/Dialect/DB/IR/DBTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"


namespace mlir::db::codegen {

class SerializationUtil {
   public:
   static Type serializedType(OpBuilder& builder, TypeConverter& typeConverter, Type type) {
      if (auto originalTupleType = type.dyn_cast_or_null<TupleType>()) {
         std::vector<Type> types;
         for (size_t i = 0; i < originalTupleType.size(); i++) {
            Type t = serializedType(builder, typeConverter,originalTupleType.getType(i));
            types.push_back(t);
         }
         return TupleType::get(builder.getContext(), types);
      } else if (auto stringType = type.dyn_cast_or_null<db::StringType>()) {
         if (stringType.isNullable()) {
            return TupleType::get(builder.getContext(), TypeRange({builder.getI1Type(), builder.getI64Type(), builder.getI64Type()}));
         } else {
            return TupleType::get(builder.getContext(), TypeRange({builder.getI64Type(), builder.getI64Type()}));
         }
      } else {
         return typeConverter.convertType(type);
      }
   }
   static Value deserialize(OpBuilder& builder, Value rawValues, Value element, Type type) {
      if (auto originalTupleType = type.dyn_cast_or_null<TupleType>()) {
         auto tupleType = element.getType().dyn_cast_or_null<TupleType>();
         std::vector<Value> serializedValues;
         std::vector<Type> types;
         auto unPackOp = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), tupleType.getTypes(), element);
         for (size_t i = 0; i < tupleType.size(); i++) {
            Value currVal = unPackOp.getResult(i);
            Value serialized = deserialize(builder, rawValues, currVal, originalTupleType.getType(i));
            serializedValues.push_back(serialized);
            types.push_back(serialized.getType());
         }
         return builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), TupleType::get(builder.getContext(), types), serializedValues);
      } else if (auto stringType = type.dyn_cast_or_null<db::StringType>()) {
         Value pos, len, isNull;
         if (stringType.isNullable()) {
            auto unPackOp = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), TypeRange({builder.getI1Type(), builder.getIndexType(), builder.getIndexType()}), element);
            isNull = unPackOp.getResult(0);
            pos = unPackOp.getResult(1);
            len = unPackOp.getResult(2);
         } else {
            auto unPackOp = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), TypeRange({builder.getIndexType(), builder.getIndexType()}), element);
            pos = unPackOp.getResult(0);
            len = unPackOp.getResult(1);
         }
         Value val = builder.create<mlir::memref::ViewOp>(builder.getUnknownLoc(), MemRefType::get({-1}, builder.getIntegerType(8)), rawValues, pos, mlir::ValueRange({len}));
         if (isNull) {
            val = builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), TupleType::get(builder.getContext(), {builder.getI1Type(), val.getType()}), ValueRange({isNull, val}));
         }
         return val;
      } else {
         return element;
      }
   }
};
}  // namespace mlir::db::codegen
#endif // MLIR_CONVERSION_DBTOARROWSTD_SERIALIZATIONUTIL_H
