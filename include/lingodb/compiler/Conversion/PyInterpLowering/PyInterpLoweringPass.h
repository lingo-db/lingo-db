#ifndef LINGODB_COMPILER_CONVERSION_PYINTERPLOWERING_PYINTERPLOWERINGPASS_H
#define LINGODB_COMPILER_CONVERSION_PYINTERPLOWERING_PYINTERPLOWERINGPASS_H
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
namespace lingodb::compiler::dialect::py_interp {
std::unique_ptr<mlir::Pass> createLowerToStdPass();

} // end namespace lingodb::compiler::dialect::py_interp

#endif //LINGODB_COMPILER_CONVERSION_PYINTERPLOWERING_PYINTERPLOWERINGPASS_H
