#ifndef LINGODB_RUNTIME_RELATIONHELPER_H
#define LINGODB_RUNTIME_RELATIONHELPER_H

#include "ExecutionContext.h"
#include "HashIndex.h"
#include "helpers.h"

#include <cstddef>
namespace lingodb::runtime {
class RelationHelper {
   public:
   static void createTable(runtime::ExecutionContext* context, runtime::VarLen32 name, runtime::VarLen32 meta);
   static void appendTableFromResult(runtime::VarLen32 tableName, runtime::ExecutionContext* context, size_t resultId);
   static void copyFromIntoTable(runtime::ExecutionContext* context, runtime::VarLen32 tableName, runtime::VarLen32 fileName, runtime::VarLen32 delimiter, runtime::VarLen32 escape);
   static void setPersist(runtime::ExecutionContext* context, bool value);
   static HashIndexAccess* getIndex(runtime::ExecutionContext* context, runtime::VarLen32 description);
};
} // end namespace lingodb::runtime

#endif //LINGODB_RUNTIME_RELATIONHELPER_H
