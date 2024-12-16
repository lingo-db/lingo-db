#ifndef LINGODB_COMPILER_CONVERSION_DBTOSTD_DBTOSTD_H
#define LINGODB_COMPILER_CONVERSION_DBTOSTD_DBTOSTD_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace lingodb::compiler::dialect {
namespace db {

std::unique_ptr<mlir::Pass> createLowerToStdPass();
void registerDBConversionPasses();
void createLowerDBPipeline(mlir::OpPassManager& pm);

} // end namespace db
} // end namespace lingodb::compiler::dialect

#endif //LINGODB_COMPILER_CONVERSION_DBTOSTD_DBTOSTD_H