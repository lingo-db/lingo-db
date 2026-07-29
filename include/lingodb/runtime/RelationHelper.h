#ifndef LINGODB_RUNTIME_RELATIONHELPER_H
#define LINGODB_RUNTIME_RELATIONHELPER_H

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
   static void copyFromIntoTableCSV(runtime::VarLen32 tableName, runtime::VarLen32 fileName, runtime::VarLen32 delimiter, runtime::VarLen32 escape, bool header);
   static void copyToFromTableCSV(runtime::VarLen32 tableName, runtime::VarLen32 fileName, runtime::VarLen32 delimiter, bool header);
   static void copyToFromTableParquet(runtime::VarLen32 tableName, runtime::VarLen32 fileName, runtime::VarLen32 compression);
   static void setPersist(bool value);
   static HashIndexAccess* accessHashIndex(runtime::VarLen32 description);

   private:
   static std::shared_ptr<arrow::Table> getArrowTableFromName(std::string tableName);
};
} // end namespace lingodb::runtime

#endif //LINGODB_RUNTIME_RELATIONHELPER_H
