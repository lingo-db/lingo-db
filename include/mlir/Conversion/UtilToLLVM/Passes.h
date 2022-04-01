#ifndef MLIR_CONVERSION_UTILTOLLVM_PASSES_H
#define MLIR_CONVERSION_UTILTOLLVM_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {
namespace util {
void populateUtilToLLVMConversionPatterns(LLVMTypeConverter& typeConverter, RewritePatternSet& patterns);
void populateUtilTypeConversionPatterns(TypeConverter& typeConverter, RewritePatternSet& patterns);
} // end namespace util
} // end namespace mlir

#endif // MLIR_CONVERSION_UTILTOLLVM_PASSES_H