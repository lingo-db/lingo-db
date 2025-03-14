#ifndef LINGODB_RUNTIME_RELATIONHELPER_H
#define LINGODB_RUNTIME_RELATIONHELPER_H

#include "ExecutionContext.h"
#include "HashIndex.h"
#include "helpers.h"

#include <cstddef>
namespace lingodb::runtime {
class RelationHelper {
   public:
   static void createTable(runtime::VarLen32 name, runtime::VarLen32 meta);
   static void appendTableFromResult(runtime::VarLen32 tableName, size_t resultId);
   static void copyFromIntoTable(runtime::VarLen32 tableName, runtime::VarLen32 fileName, runtime::VarLen32 delimiter, runtime::VarLen32 escape);
   static void setPersist(bool value);
   static HashIndexAccess* getIndex(runtime::VarLen32 description);
};
} // end namespace lingodb::runtime

#endif //LINGODB_RUNTIME_RELATIONHELPER_H
