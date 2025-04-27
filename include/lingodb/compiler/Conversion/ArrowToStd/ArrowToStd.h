#ifndef LINGODB_COMPILER_CONVERSION_ARROWTOSTD_ARROWTOSTD_H
#define LINGODB_COMPILER_CONVERSION_ARROWTOSTD_ARROWTOSTD_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace lingodb::compiler::dialect {
namespace arrow {

std::unique_ptr<mlir::Pass> createLowerToStdPass();

} // end namespace arrow
} // end namespace lingodb::compiler::dialect

#endif //LINGODB_COMPILER_CONVERSION_ARROWTOSTD_ARROWTOSTD_H