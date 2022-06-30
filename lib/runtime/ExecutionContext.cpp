#include "runtime/ExecutionContext.h"
runtime::Database* runtime::ExecutionContext::getDatabase() {
   return db.get();
}