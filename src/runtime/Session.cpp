#include "lingodb/runtime/Session.h"
#include "lingodb/runtime/ExecutionContext.h"

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
   for (auto state : pythonThreadStates) {
      if (state != nullptr) {
         PyThreadState_Swap((PyThreadState*)state);
         Py_EndInterpreter(PyThreadState_Get());
      }
   }
#endif
}