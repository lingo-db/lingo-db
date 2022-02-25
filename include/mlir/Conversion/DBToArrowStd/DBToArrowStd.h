#ifndef MLIR_CONVERSION_DBTOARROWSTD_DBTOARROWSTD_H
#define MLIR_CONVERSION_DBTOARROWSTD_DBTOARROWSTD_H

#include "mlir/Conversion/DBToArrowStd/FunctionRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {
namespace db {
void populateScalarToStdPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns);
void populateControlFlowToStdPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns);
void populateRuntimeSpecificScalarToStdPatterns(mlir::db::codegen::FunctionRegistry& functionRegistry, mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns);
void populateBuilderToStdPatterns(mlir::db::codegen::FunctionRegistry& joinHtBuilderType, mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns);
void populateCollectionsToStdPatterns(mlir::db::codegen::FunctionRegistry& joinHashtableType, mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns);

std::unique_ptr<Pass> createLowerToStdPass();

} // end namespace db
} // end namespace mlir

#endif // MLIR_CONVERSION_DBTOARROWSTD_DBTOARROWSTD_H