#ifndef MLIR_CONVERSION_RELALGTODB_RELALGTODBPASS_H
#define MLIR_CONVERSION_RELALGTODB_RELALGTODBPASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
    namespace relalg {
        std::unique_ptr<Pass> createLowerToDBPass();
    }// end namespace relalg
}// end namespace mlir

#endif // MLIR_CONVERSION_RELALGTODB_RELALGTODBPASS_H