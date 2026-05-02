#ifndef LINGODB_RUNTIME_SESSION_H
#define LINGODB_RUNTIME_SESSION_H
#include "lingodb/catalog/Catalog.h"
#include "lingodb/scheduler/Scheduler.h"

#include <memory>
#include <mutex>
#include <stack>
#include <vector>
#ifdef USE_CPYTHON_RUNTIME
namespace lingodb::runtime {
struct PythonExtState;
}
#endif
#ifdef USE_CPYTHON_WASM_RUNTIME
namespace lingodb::wasm {
struct WASMSession;
} // namespace lingodb::wasm
#endif

namespace lingodb::runtime {
class ExecutionContext;
class Session {
   std::shared_ptr<catalog::Catalog> catalog;

   public:
#ifdef USE_CPYTHON_RUNTIME
   // One sub-interpreter thread state and one extension state per worker thread.
   // Lazily populated on first Python use; freed in ~Session.
   std::vector<void*> pythonThreadStates;
   std::vector<PythonExtState*> pythonExtStates;
#endif
#ifdef USE_CPYTHON_WASM_RUNTIME
   // One WASMSession per worker thread (runs CPython compiled to WASM in WAMR).
   std::vector<lingodb::wasm::WASMSession*> wasmEnvironments;
#endif

   Session(std::shared_ptr<catalog::Catalog> catalog) : catalog(catalog) {
#ifdef USE_CPYTHON_RUNTIME
      // Sized once here — the scheduler must already be running. Doing it once
      // up front avoids racing resizes when workers concurrently take their
      // first Python op. All callers (tools/, bridge) start the scheduler
      // before constructing the Session.
      pythonThreadStates.resize(lingodb::scheduler::getNumWorkers(), nullptr);
      pythonExtStates.resize(lingodb::scheduler::getNumWorkers(), nullptr);
#endif
#ifdef USE_CPYTHON_WASM_RUNTIME
      wasmEnvironments.resize(lingodb::scheduler::getNumWorkers(), nullptr);
#endif
   }
   static std::shared_ptr<Session> createSession();
   static std::shared_ptr<Session> createSession(std::string dbDir, bool eagerLoading = true);
   std::shared_ptr<catalog::Catalog> getCatalog();
   std::unique_ptr<ExecutionContext> createExecutionContext();
   ~Session();
};
} //end namespace lingodb::runtime
#endif //LINGODB_RUNTIME_SESSION_H
