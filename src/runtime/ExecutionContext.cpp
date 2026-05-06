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
// "isolated"   — each worker gets its own GIL + strict multi-interpreter
//                extension check. Best for parallel scalar UDFs that only
//                touch stdlib. Refuses to load legacy C extensions like
//                pyarrow inside sub-interpreters.
// "compatible" — workers share the host interpreter's GIL (serialised Python
//                execution) but legacy extensions can be imported. Required
//                for tabular Python UDFs (which load pyarrow / numpy).
lingodb::utility::GlobalSetting<std::string> pythonSubinterpMode("system.python.subinterpreter_mode", "isolated");
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
   const auto mode = pythonSubinterpMode.getValue();
   const bool compatible = (mode == "compatible");
   if (compatible) {
      // Run all tabular UDFs against the host's main interpreter. Sub-
      // interpreters can't share legacy single-phase-init extensions like
      // numpy/pandas: each one would be the "first interpreter" from
      // numpy's POV, and the second sub-interpreter to import it dies with
      // "Interpreter change detected". With a single shared interpreter
      // numpy is imported once and reused across queries / workers.
      //
      // PyGILState_Ensure attaches the calling OS thread to the main
      // interpreter (creating a tstate for it on first call) and acquires
      // the GIL. Multiple workers therefore serialise on the main GIL —
      // acceptable because that's the trade-off of compatible mode.
      PyGILState_STATE gstate = PyGILState_Ensure();
      // Pack the gilstate into the per-worker slot so teardownPython can
      // release it. Add 2 to avoid colliding with kMainInterpreterMarker
      // (LOCKED is 0, UNLOCKED is 1 in CPython 3.12).
      session.pythonThreadStates[workerId] = reinterpret_cast<void*>(static_cast<uintptr_t>(gstate) + 2);
      if (session.pythonExtStates[workerId] == nullptr) {
         session.pythonExtStates[workerId] = PythonRuntime::createPythonExtState();
      }
      return;
   }
   if (session.pythonThreadStates[workerId] == nullptr) {
      // First time this worker enters a Python region — give it its own
      // sub-interpreter (with its own GIL) so workers don't fight over the GIL.
      PyThreadState* tstate = nullptr;
      PyInterpreterConfig config = {
         .use_main_obmalloc = 0,
         .allow_fork = 0,
         .allow_threads = 0,
         .allow_daemon_threads = 0,
         .check_multi_interp_extensions = 1,
         .gil = PyInterpreterConfig_OWN_GIL,
      };
      PyStatus status = Py_NewInterpreterFromConfig(&tstate, &config);
      if (PyStatus_Exception(status)) {
         Py_ExitStatusException(status);
      }
      session.pythonExtStates[workerId] = PythonRuntime::createPythonExtState();
   } else {
      PyThreadState_Swap((PyThreadState*) session.pythonThreadStates[workerId]);
   }
}
void lingodb::runtime::ExecutionContext::teardownPython() {
   auto workerId = scheduler::currentWorkerId();
   const auto mode = pythonSubinterpMode.getValue();
   if (mode == "compatible") {
      // Match the PyGILState_Ensure from setupPython.
      auto packed = reinterpret_cast<uintptr_t>(session.pythonThreadStates[workerId]);
      if (packed >= 2) {
         PyGILState_STATE gstate = static_cast<PyGILState_STATE>(packed - 2);
         PyGILState_Release(gstate);
         session.pythonThreadStates[workerId] = nullptr;
      }
      return;
   }
   auto* state = PyThreadState_Swap(nullptr);
   if (state) {
      session.pythonThreadStates[workerId] = state;
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
