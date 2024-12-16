#ifndef LINGODB_COMPILER_CONVERSION_DSATOSTD_DSATOSTD_H
#define LINGODB_COMPILER_CONVERSION_DSATOSTD_DSATOSTD_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace lingodb::compiler::dialect {
namespace dsa {
void populateScalarToStdPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns);
void populateBuilderToStdPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns);
void populateDSAToStdPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns);
void populateCollectionsToStdPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns);

std::unique_ptr<mlir::Pass> createLowerToStdPass();

} // end namespace dsa
} // end namespace lingodb::compiler::dialect

#endif //LINGODB_COMPILER_CONVERSION_DSATOSTD_DSATOSTD_H