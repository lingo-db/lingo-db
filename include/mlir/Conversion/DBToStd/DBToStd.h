#ifndef MLIR_CONVERSION_DBTOSTD_DBTOSTD_H
#define MLIR_CONVERSION_DBTOSTD_DBTOSTD_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {
namespace db {

std::unique_ptr<Pass> createLowerToStdPass();
void registerDBConversionPasses();
void createLowerDBPipeline(mlir::OpPassManager& pm);

} // end namespace db
} // end namespace mlir

#endif // MLIR_CONVERSION_DBTOSTD_DBTOSTD_H