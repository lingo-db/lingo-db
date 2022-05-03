#ifndef MLIR_CONVERSION_DBTOARROWSTD_DBTOARROWSTD_H
#define MLIR_CONVERSION_DBTOARROWSTD_DBTOARROWSTD_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {
namespace db {
void populateScalarToStdPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns);
void populateRuntimeSpecificScalarToStdPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns);

std::unique_ptr<Pass> createLowerToStdPass();
void registerDBConversionPasses();
void createLowerDBPipeline(mlir::OpPassManager& pm);

} // end namespace db
} // end namespace mlir

#endif // MLIR_CONVERSION_DBTOARROWSTD_DBTOARROWSTD_H