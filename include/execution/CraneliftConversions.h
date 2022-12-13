#ifndef EXECUTION_CRANELIFTCONVERSIONS_H
#define EXECUTION_CRANELIFTCONVERSIONS_H
#include "mlir/Pass/Pass.h"
namespace mlir::cranelift{
void registerCraneliftConversionPasses();
std::unique_ptr<Pass> createLowerToCraneliftPass();

} // namespace mlir::cranelift

#endif //EXECUTION_CRANELIFTCONVERSIONS_H
