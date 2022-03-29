#ifndef MLIR_CONVERSION_DSATOARROWSTD_COLLECTIONITERATION_H
#define MLIR_CONVERSION_DSATOARROWSTD_COLLECTIONITERATION_H
#include "mlir/Conversion/DSAToStd/FunctionRegistry.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::dsa {
class CollectionIterationImpl {
   public:
   virtual std::vector<Value> implementLoop(mlir::Location loc,mlir::ValueRange iterArgs, Value flag, mlir::TypeConverter& typeConverter, ConversionPatternRewriter& builder, mlir::ModuleOp parentModule, std::function<std::vector<Value>(std::function<Value(OpBuilder&)>,ValueRange, OpBuilder)> bodyBuilder) = 0;
   virtual ~CollectionIterationImpl() {
   }
   static std::unique_ptr<mlir::dsa::CollectionIterationImpl> getImpl(mlir::Type collectionType, mlir::Value collection,mlir::Value loweredCollection, mlir::dsa::codegen::FunctionRegistry& functionRegistry);
};

} // namespace mlir::dsa
#endif // MLIR_CONVERSION_DSATOARROWSTD_COLLECTIONITERATION_H
