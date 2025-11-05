#include "lingodb/runtime/ExecutionContext.h"
#include "lingodb/runtime/PythonRuntime.h"
#ifdef USE_CPYTHON_RUNTIME
#include "Python.h"
#endif
#ifdef USE_CPYTHON_WASM_RUNTIME
#include "lingodb/runtime/WASM.h"
#include "wasm_export.h"
#endif
#include <cassert>

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

void lingodb::runtime::ExecutionContext::resetPythonSessionCache(){
#ifdef USE_CPYTHON_WASM_RUNTIME
   for (auto *x: session.wasmEnvironments) {
      if (x) {
         x->clearCache();
      }
   }
#endif
#ifdef USE_CPYTHON_RUNTIME
   for (size_t i = 0; i < session.pythonExtStates.size(); ++i) {
      if (session.pythonExtStates[i]) {
         session.pythonExtStates[i]->clearCache();
      }
   }
#endif
}

uint8_t* lingodb::runtime::ExecutionContext::allocStateRaw(size_t size) {
   auto* context = getCurrentExecutionContext();
   assert(context);
   uint8_t* ptr = static_cast<uint8_t*>(malloc(size));
   context->registerState({ptr, [](void* p) { free(p); }});
   return ptr;
}
#ifdef USE_CPYTHON_WASM_RUNTIME

void lingodb::runtime::ExecutionContext::setupWasm() {
   auto workerId = scheduler::currentWorkerId();
   auto wasmSession = session.wasmEnvironments[workerId];
   if (wasmSession == nullptr) {
      session.wasmEnvironments[workerId]=
         wasm::WASM::initializeWASM();
      wasmSession = session.wasmEnvironments[workerId];
   } else {
#if !ASAN_ACTIVE
      wasm_runtime_set_native_stack_boundary(static_cast<wasm_exec_env_t>(session.wasmEnvironments[workerId]->execEnv), scheduler::getStackBoundary());
#endif
   }
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
void lingodb::runtime::ExecutionContext::setupPython() {
   auto workerId = scheduler::currentWorkerId();
   if (session.pythonThreadStates[workerId] == nullptr) {
      //PyGILState_STATE gs = PyGILState_Ensure();
      //auto mainState= PyThreadState_Get();
      PyThreadState *tstate = nullptr;
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
      //auto x = PyThreadState_Swap(mainState);
      //PyGILState_Release(gs);
      //PyThreadState_Swap(x);
   } else {
      PyThreadState_Swap((PyThreadState*)session.pythonThreadStates[workerId]);
   }
}
void lingodb::runtime::ExecutionContext::teardownPython() {
   //auto mainState= PyThreadState_Get();
   auto workerId = scheduler::currentWorkerId();
   auto state = PyThreadState_Swap(nullptr);//(mainState);
   if (state) {
      session.pythonThreadStates[workerId] = state;
   }
}
#endif