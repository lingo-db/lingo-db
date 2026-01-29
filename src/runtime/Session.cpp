#include "lingodb/runtime/Session.h"
#include "lingodb/runtime/ExecutionContext.h"
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
#ifndef MLIR_DISABLED

   std::lock_guard<std::mutex> lock(contextStackMutex);
   while (!llvmContextStack.empty()) {
      auto* ctx = llvmContextStack.top();
      llvmContextStack.pop();
      delete ctx;
   }
   while (!noLlvmContextStack.empty()) {
      auto* ctx = noLlvmContextStack.top();
      noLlvmContextStack.pop();
      delete ctx;
   }
#endif
}