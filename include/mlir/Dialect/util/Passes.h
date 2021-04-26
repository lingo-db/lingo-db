#ifndef MLIR_UTIL_PASSES_H
#define MLIR_UTIL_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {
    namespace util {
    void populateUtilToLLVMConversionPatterns(LLVMTypeConverter& typeConverter,RewritePatternSet &patterns);
    }// end namespace util
}// end namespace mlir

#endif// MLIR_UTIL_PASSES_H