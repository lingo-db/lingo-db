#ifndef MLIR_DB_PASSES_H
#define MLIR_DB_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createTestPass();
std::unique_ptr<Pass> createExtractNestedOperatorsPass();

} // end namespace db
} // end namespace mlir

#endif // MLIR_DB_PASSES_H