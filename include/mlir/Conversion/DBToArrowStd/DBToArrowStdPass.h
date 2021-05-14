#ifndef MLIR_CONVERSION_DBTOARROWSTD_DBTOARROWSTDPASS_H
#define MLIR_CONVERSION_DBTOARROWSTD_DBTOARROWSTDPASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
    namespace db {
        std::unique_ptr<Pass> createLowerToStdPass();
    }// end namespace db
}// end namespace mlir

#endif // MLIR_CONVERSION_DBTOARROWSTD_DBTOARROWSTDPASS_H