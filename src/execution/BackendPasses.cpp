#include "lingodb/execution/BackendPasses.h"

void lingodb::execution::registerBackendPasses() {
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return execution::createEnforceCABI();
   });

}