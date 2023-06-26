#include "runtime/Session.h"
#include "runtime/ExecutionContext.h"
std::unique_ptr<runtime::ExecutionContext> runtime::Session::createExecutionContext() {
   return std::make_unique<ExecutionContext>(*this);
}

std::shared_ptr<runtime::Catalog> runtime::Session::getCatalog() {
   return catalog;
}

std::shared_ptr<runtime::Session> runtime::Session::createSession() {
   return std::shared_ptr<Session>(new Session(LocalCatalog::create(Catalog::createEmpty())));
}
std::shared_ptr<runtime::Session> runtime::Session::createSession(std::string dbDir, bool eagerLoading) {
   return std::shared_ptr<Session>(new Session(DBCatalog::create(Catalog::createEmpty(), dbDir, eagerLoading)));
}