#ifndef LINGODB_COMPILER_CONVERSION_UTILTOLLVM_PASSES_H
#define LINGODB_COMPILER_CONVERSION_UTILTOLLVM_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Transforms/DialectConversion.h>

namespace lingodb::compiler::dialect {
namespace util {
void populateUtilToLLVMConversionPatterns(mlir::LLVMTypeConverter& typeConverter, mlir::RewritePatternSet& patterns);
void populateUtilTypeConversionPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns);
std::unique_ptr<mlir::Pass> createUtilToLLVMPass();
void registerUtilConversionPasses();
} // end namespace util
} // end namespace lingodb::compiler::dialect

#endif //LINGODB_COMPILER_CONVERSION_UTILTOLLVM_PASSES_H