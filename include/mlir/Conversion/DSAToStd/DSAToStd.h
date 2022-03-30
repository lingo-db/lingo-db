#ifndef MLIR_CONVERSION_DSATOSTD_DSATOSTD_H
#define MLIR_CONVERSION_DSATOSTD_DSATOSTD_H

#include "mlir/Conversion/DSAToStd/FunctionRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {
namespace dsa {
void populateScalarToStdPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns);
void populateRuntimeSpecificScalarToStdPatterns(mlir::dsa::codegen::FunctionRegistry& functionRegistry, mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns);
void populateBuilderToStdPatterns(mlir::dsa::codegen::FunctionRegistry& joinHtBuilderType, mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns);
void populateDSAToStdPatterns(mlir::dsa::codegen::FunctionRegistry& joinHtBuilderType, mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns);
void populateCollectionsToStdPatterns(mlir::dsa::codegen::FunctionRegistry& joinHashtableType, mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns);

std::unique_ptr<Pass> createLowerToStdPass();

} // end namespace dsa
} // end namespace mlir

#endif // MLIR_CONVERSION_DSATOSTD_DSATOSTD_H