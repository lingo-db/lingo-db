#ifndef MLIR_CONVERSION_DBTOARROWSTD_NULLHANDLER_H
#define MLIR_CONVERSION_DBTOARROWSTD_NULLHANDLER_H
#include "mlir/Transforms/DialectConversion.h"
namespace mlir {
namespace db {
class NullHandler {
   std::vector<Value> nullValues;
   TypeConverter& typeConverter;
   OpBuilder& builder;

   public:
   NullHandler(TypeConverter& typeConverter, OpBuilder& builder) : typeConverter(typeConverter), builder(builder) {}
   Value isNull();
   Value combineResult(Value res);
   Value getValue(Value v, Value operand = Value());
};
} // end namespace db
} // end namespace mlir
#endif // MLIR_CONVERSION_DBTOARROWSTD_NULLHANDLER_H
