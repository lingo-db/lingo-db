#include "lingodb/runtime/ExecutionContext.h"
#include "lingodb/runtime/PythonRuntime.h"
#include "lingodb/utility/Setting.h"
#ifdef USE_CPYTHON_RUNTIME
#include "Python.h"
#endif
#ifdef USE_CPYTHON_WASM_RUNTIME
#include "lingodb/runtime/WASM.h"
#endif
#include <cassert>

#ifdef USE_CPYTHON_RUNTIME
namespace {
// Free-threaded model (CPython 3.14t, Py_GIL_DISABLED): every worker shares the
// single main interpreter and runs Python concurrently with no GIL. This
// replaces the former two-mode scheme selected by
// system.python.subinterpreter_mode:
//   - "isolated"   — one sub-interpreter per worker (each with its own GIL).
//                    Ran scalar UDFs in parallel but could not import legacy
//                    single-phase-init C extensions (pyarrow/numpy).
//   - "compatible" — all workers shared the host GIL, so legacy extensions
//                    loaded but Python execution serialised.
// On a free-threaded build that trade-off disappears: the shared interpreter
// is both parallel AND able to import legacy extensions. Sub-interpreters (and
// their fragile cross-thread Py_EndInterpreter teardown, see Session.cpp) are
// gone. The setting is kept — accepted but ignored — so existing configs and
// the LINGODB_PYTHON_SUBINTERPRETER_MODE env var don't error.
[[maybe_unused]] lingodb::utility::GlobalSetting<std::string> pythonSubinterpMode("system.python.subinterpreter_mode", "shared");
} // namespace
#endif

void lingodb::runtime::ExecutionContext::setResult(uint32_t id, uint8_t* ptr) {
   auto* context = getCurrentExecutionContext();
   assert(context);
   context->results[id] = ptr;
}
void lingodb::runtime::ExecutionContext::clearResult(uint32_t id) {
   auto* context = getCurrentExecutionContext();
   context->results.erase(id);
}

void lingodb::runtime::ExecutionContext::setTupleCount(uint32_t id, int64_t tupleCount) {
   auto* context = getCurrentExecutionContext();
   context->tupleCounts[id] = tupleCount;
}

lingodb::runtime::ExecutionContext::~ExecutionContext() {
   for (auto threadLocal : perWorkerStates) {
      for (auto s : threadLocal) {
         s.freeFn(s.ptr);
      }
   }
   for (auto local : allocators) {
      for (auto a : local) {
         a.second.freeFn(a.second.ptr);
      }
   }
   allocators.clear();
   perWorkerStates.clear();
}

uint8_t* lingodb::runtime::ExecutionContext::allocStateRaw(size_t size) {
   auto* context = getCurrentExecutionContext();
   assert(context);
   uint8_t* ptr = static_cast<uint8_t*>(malloc(size));
   context->registerState({ptr, [](void* p) { free(p); }});
   return ptr;
}

namespace {
thread_local lingodb::runtime::ExecutionContext* currentExecutionContext = nullptr;
} // end namespace
void lingodb::runtime::setCurrentExecutionContext(lingodb::runtime::ExecutionContext* context) {
   currentExecutionContext = context;
}

lingodb::runtime::ExecutionContext* lingodb::runtime::getCurrentExecutionContext() {
   assert(currentExecutionContext);
   return currentExecutionContext;
}

#ifdef USE_CPYTHON_RUNTIME
void lingodb::runtime::ExecutionContext::resetPythonSessionCache() {
   for (size_t i = 0; i < session.pythonExtStates.size(); ++i) {
      if (session.pythonExtStates[i]) {
         session.pythonExtStates[i]->clearCache();
      }
   }
}
void lingodb::runtime::ExecutionContext::setupPython() {
   auto workerId = scheduler::currentWorkerId();
   // Attach this worker's OS thread to the shared main interpreter.
   // PyGILState_Ensure creates a PyThreadState for the thread on first use and
   // reuses it afterwards. On a free-threaded build it does NOT serialise
   // workers: there is no GIL to contend for, so many workers run Python in
   // parallel against the one interpreter. Legacy single-phase-init extensions
   // (pyarrow/numpy) import once into that interpreter and are shared — which
   // is why tabular UDFs no longer need a separate "compatible" mode.
   PyGILState_STATE gstate = PyGILState_Ensure();
   // Pack the gilstate into the per-worker slot so teardownPython can release
   // it. Offset by 1 so a set slot is always non-null (LOCKED is 0, UNLOCKED
   // is 1); nullptr means "this worker is not currently in a Python region".
   session.pythonThreadStates[workerId] = reinterpret_cast<void*>(static_cast<uintptr_t>(gstate) + 1);
   if (session.pythonExtStates[workerId] == nullptr) {
      session.pythonExtStates[workerId] = PythonRuntime::createPythonExtState();
   }
}
void lingodb::runtime::ExecutionContext::teardownPython() {
   auto workerId = scheduler::currentWorkerId();
   // Match the PyGILState_Ensure from setupPython (same OS thread).
   auto packed = reinterpret_cast<uintptr_t>(session.pythonThreadStates[workerId]);
   if (packed >= 1) {
      PyGILState_STATE gstate = static_cast<PyGILState_STATE>(packed - 1);
      PyGILState_Release(gstate);
      session.pythonThreadStates[workerId] = nullptr;
   }
}
#endif

#ifdef USE_CPYTHON_WASM_RUNTIME
void lingodb::runtime::ExecutionContext::setupWasm() {
   auto workerId = scheduler::currentWorkerId();
   auto* wasmSession = session.wasmEnvironments[workerId];
   if (wasmSession == nullptr) {
      session.wasmEnvironments[workerId] = wasm::WASM::initializeWASM();
      wasmSession = session.wasmEnvironments[workerId];
   }
   // Note: WAMR queries the native stack via pthread_attr; since the new
   // scheduler places every fiber's stack inside the worker's pthread stack
   // range, no explicit wasm_runtime_set_native_stack_boundary is needed.
   PythonRuntime::setWasmSession(wasmSession);
}
void lingodb::runtime::ExecutionContext::teardownWasm() {
   //TODO teardown
}
lingodb::wasm::WASMSession* lingodb::runtime::ExecutionContext::getWasmSession() {
   auto workerId = scheduler::currentWorkerId();
   return session.wasmEnvironments[workerId];
}
#endif
