#ifndef MLIR_CONVERSION_RELALGTODB_RELALGTODBPASS_H
#define MLIR_CONVERSION_RELALGTODB_RELALGTODBPASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
    namespace relalg {
        std::unique_ptr<Pass> createLowerToDBPass();
        void registerRelAlgConversionPasses();
        void createLowerRelAlgPipeline(mlir::OpPassManager& pm);
    }// end namespace relalg
}// end namespace mlir

#endif // MLIR_CONVERSION_RELALGTODB_RELALGTODBPASS_H