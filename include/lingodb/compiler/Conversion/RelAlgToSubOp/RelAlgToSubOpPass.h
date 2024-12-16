#ifndef LINGODB_COMPILER_CONVERSION_RELALGTOSUBOP_RELALGTOSUBOPPASS_H
#define LINGODB_COMPILER_CONVERSION_RELALGTOSUBOP_RELALGTOSUBOPPASS_H
#include "mlir/Pass/Pass.h"
#include <memory>

namespace lingodb::compiler::dialect {
namespace relalg {
std::unique_ptr<mlir::Pass> createLowerToSubOpPass();
void registerRelAlgToSubOpConversionPasses();
void createLowerRelAlgToSubOpPipeline(mlir::OpPassManager& pm);
} // end namespace relalg
} // end namespace lingodb::compiler::dialect
#endif //LINGODB_COMPILER_CONVERSION_RELALGTOSUBOP_RELALGTOSUBOPPASS_H
