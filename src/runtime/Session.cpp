#include "lingodb/runtime/Session.h"
#include "lingodb/runtime/ExecutionContext.h"
#include "lingodb/runtime/PythonRuntime.h"
#include "mlir/IR/MLIRContext.h"

std::unique_ptr<lingodb::runtime::ExecutionContext> lingodb::runtime::Session::createExecutionContext() {
   return std::make_unique<ExecutionContext>(*this);
}

std::shared_ptr<lingodb::catalog::Catalog> lingodb::runtime::Session::getCatalog() {
   return catalog;
}

std::shared_ptr<lingodb::runtime::Session> lingodb::runtime::Session::createSession() {
   return std::make_shared<Session>(catalog::Catalog::createEmpty());
}
std::shared_ptr<lingodb::runtime::Session> lingodb::runtime::Session::createSession(std::string dbDir, bool eagerLoading) {
   return std::make_shared<Session>(catalog::Catalog::create(dbDir, eagerLoading));
}

lingodb::runtime::Session::~Session() {
#ifdef USE_CPYTHON_RUNTIME
   // Free-threaded model: workers share the single main interpreter (see
   // ExecutionContext::setupPython) — there are no per-worker sub-interpreters
   // to end here. This removes the whole fragile cross-thread
   // Py_EndInterpreter dance and its version-gated workarounds (the 3.12
   // threading._shutdown hang and the 3.13+ Py_FinalizeEx auto-reap path):
   // there is simply nothing to tear down. Any PyGILState a worker acquired is
   // released in teardownPython; the interpreter itself is owned by the host
   // (the standalone binary's Py_Initialize, or the embedding Python process
   // for the wheel). We only drop the per-worker extension caches.
   //
   // The cached PyObject*s inside each PythonExtState are intentionally not
   // decref'd: at session/process teardown the interpreter may already be
   // finalizing, and leaking a handful of module references is harmless.
   for (auto* extState : pythonExtStates) {
      delete extState;
   }
#endif
}
