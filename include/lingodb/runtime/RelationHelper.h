#ifndef LINGODB_RUNTIME_RELATIONHELPER_H
#define LINGODB_RUNTIME_RELATIONHELPER_H

#include "ExecutionContext.h"
#include "LingoDBHashIndex.h"
#include "helpers.h"

#include <cstddef>
namespace lingodb::runtime {
class RelationHelper {
   public:
   static void appendToTable(runtime::Session& session, std::string tableName, std::shared_ptr<arrow::Table> table);
   static void createTable(runtime::VarLen32 meta);
   static void createFunction(runtime::VarLen32 meta);
   static void appendTableFromResult(runtime::VarLen32 tableName, size_t resultId);
   static void copyFromIntoTable(runtime::VarLen32 tableName, runtime::VarLen32 fileName, runtime::VarLen32 delimiter, runtime::VarLen32 escape);
   static void setPersist(bool value);
   static HashIndexAccess* accessHashIndex(runtime::VarLen32 description);
};
} // end namespace lingodb::runtime

#endif //LINGODB_RUNTIME_RELATIONHELPER_H
