#include "lingodb/runtime/Session.h"
#include "lingodb/runtime/ExecutionContext.h"
#include "lingodb/runtime/PythonRuntime.h"
#include "mlir/IR/MLIRContext.h"
#ifdef USE_CPYTHON_RUNTIME
#include "Python.h"
#endif

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
   // If the host runtime is already finalizing (3.13+ wheel scenario:
   // ~Session runs from inside Py_FinalizeEx via nanobind tp_dealloc), do
   // NOT call Py_EndInterpreter ourselves. Py_EndInterpreter's first step
   // is wait_for_thread_shutdown / threading._shutdown, which blocks on a
   // sub-interpreter Thread._tstate_lock that the worker that created the
   // sub-interpreter never released. Py_FinalizeEx in 3.13+ reaps leftover
   // sub-interpreters itself.
   //
   // Outside finalize, end each sub-interpreter ourselves. We must NOT use
   // PyGILState_Ensure here: PyThreadState_Clear on the sub-interpreter's
   // tstate (called inside Py_EndInterpreter) calls unbind_gilstate_tstate
   // because the sub-interpreter's tstate has bound_gilstate=1 (set by the
   // worker thread that created it via Py_NewInterpreterFromConfig). That
   // unbind clobbers gilstate TSS for *this* OS thread, breaking the
   // PyGILState pairing.
   //
   // Raw PyThreadState_Swap also manages the GIL (3.12.8 pystate.c:1905)
   // — releasing the old interp's GIL and acquiring the new one — so it's
   // a complete substitute as long as we save/restore the entry tstate.
#if PY_VERSION_HEX >= 0x030D0000
   if (Py_IsFinalizing()) {
#else
   if (_Py_IsFinalizing()) {
#endif
      // Skip Py_EndInterpreter; only delete the C++-side ext state.
   } else {
      PyThreadState* host_saved = PyThreadState_Swap(nullptr);
      for (auto* state_void : pythonThreadStates) {
         if (state_void == nullptr) continue;
         PyThreadState* sub_main = (PyThreadState*) state_void;
         PyThreadState_Swap(sub_main);
         Py_EndInterpreter(sub_main);
      }
      pythonThreadStates.clear();
      if (host_saved != nullptr) {
         PyThreadState_Swap(host_saved);
      }
   }
   for (auto* extState : pythonExtStates) {
      delete extState;
   }
#endif
}
