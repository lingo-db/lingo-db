#ifndef LINGODB_RUNTIME_SESSION_H
#define LINGODB_RUNTIME_SESSION_H
#include "lingodb/catalog/Catalog.h"
#include "lingodb/scheduler/Scheduler.h"

#include<vector>
#include <memory>
#include <mutex>
#include <stack>

#ifdef USE_CPYTHON_WASM_RUNTIME
namespace lingodb::wasm {
struct WASMSession;
} // namespace lingodb::wasm
#endif
#ifdef USE_CPYTHON_RUNTIME
namespace lingodb::runtime {
struct PythonExtState;
}
#endif

namespace lingodb::runtime {
class ExecutionContext;
class Session {
   std::shared_ptr<catalog::Catalog> catalog;
   public:
#ifdef USE_CPYTHON_RUNTIME
   std::vector<void*> pythonThreadStates;
   std::vector<PythonExtState*> pythonExtStates;
#endif
#ifdef USE_CPYTHON_WASM_RUNTIME
   std::vector<lingodb::wasm::WASMSession*> wasmEnvironments;
#endif
   public:
   Session(std::shared_ptr<catalog::Catalog> catalog) : catalog(catalog) {
#ifdef USE_CPYTHON_RUNTIME
      pythonThreadStates.resize(lingodb::scheduler::getNumWorkers(), nullptr);
      pythonExtStates.resize(lingodb::scheduler::getNumWorkers());
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
