#ifndef MLIR_CONVERSION_DBTOARROWSTD_COLLECTIONIMPLEMENTATION_H
#define MLIR_CONVERSION_DBTOARROWSTD_COLLECTIONIMPLEMENTATION_H
namespace mlir::db {
class CollectionImplementation {
   virtual std::vector<mlir::Value> implementLoop(mlir::ValueRange iterArgs, mlir::TypeConverter& typeConverter, ConversionPatternRewriter& builder, mlir::ModuleOp parentModule, std::function<std::vector<Value>(ValueRange, OpBuilder)> bodyBuilder) = 0;
};
}
#endif // MLIR_CONVERSION_DBTOARROWSTD_COLLECTIONIMPLEMENTATION_H
