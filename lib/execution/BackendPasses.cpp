#include "execution/BackendPasses.h"

void execution::registerBackendPasses() {
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return execution::createEnforceCABI();
   });

}