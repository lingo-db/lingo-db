#include "lingodb/runtime/ExecutionContext.h"
#include "Python.h"
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

void lingodb::runtime::ExecutionContext::setupPython() {
   auto workerId = scheduler::currentWorkerId();
   if (pythonThreadStates[workerId] == nullptr) {
      PyGILState_STATE gs = PyGILState_Ensure();
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
   } else {
      PyThreadState_Swap((PyThreadState*)pythonThreadStates[workerId]);
   }
}
void lingodb::runtime::ExecutionContext::teardownPython() {
   auto workerId = scheduler::currentWorkerId();
   auto state = PyThreadState_Swap(nullptr);
   pythonThreadStates[workerId] = state;
}