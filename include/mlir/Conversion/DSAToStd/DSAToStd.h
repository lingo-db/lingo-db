#ifndef MLIR_CONVERSION_DSATOSTD_DSATOSTD_H
#define MLIR_CONVERSION_DSATOSTD_DSATOSTD_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {
namespace dsa {
void populateScalarToStdPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns);
void populateBuilderToStdPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns);
void populateDSAToStdPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns);
void populateCollectionsToStdPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns);

std::unique_ptr<Pass> createLowerToStdPass();

} // end namespace dsa
} // end namespace mlir

#endif // MLIR_CONVERSION_DSATOSTD_DSATOSTD_H