#include "execution/BackendPasses.h"
#include "execution/CraneliftConversions.h"

void execution::registerBackendPasses() {
#if CRANELIFT_ENABLED == 1
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return execution::createDecomposeTuplePass();
   });
   mlir::cranelift::registerCraneliftConversionPasses();
#endif
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return execution::createEnforceCABI();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return execution::createAnnotateProfilingDataPass();
   });
}