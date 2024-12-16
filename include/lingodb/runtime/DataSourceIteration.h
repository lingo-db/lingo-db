#ifndef LINGODB_RUNTIME_DATASOURCEITERATION_H
#define LINGODB_RUNTIME_DATASOURCEITERATION_H
#include "ArrowTable.h"
#include "RecordBatchInfo.h"
#include "lingodb/runtime/ExecutionContext.h"
#include "lingodb/runtime/helpers.h"
namespace lingodb::runtime {
class DataSource {
   public:
   virtual size_t getColumnId(std::string member) = 0;
   virtual void iterate(bool parallel, std::vector<size_t> colIds, const std::function<void(runtime::RecordBatchInfo*)>& cb) = 0;
   virtual ~DataSource() {}
   static DataSource* get(ExecutionContext* executionContext, runtime::VarLen32 description);
   static DataSource* getFromTable(ArrowTable* arrowTable, runtime::VarLen32 mappingVal,runtime::VarLen32 columnArray);
};
class DataSourceIteration {
   std::shared_ptr<arrow::RecordBatch> currChunk;
   DataSource* dataSource;
   std::vector<size_t> colIds;

   public:
   DataSourceIteration(DataSource* dataSource, const std::vector<size_t>& colIds);

   static DataSourceIteration* init(DataSource* dataSource, runtime::VarLen32 members);
   static void end(DataSourceIteration*);
   void iterate(bool parallel, void (*forEachChunk)(RecordBatchInfo*, void*), void*);
};
} // end namespace lingodb::runtime
#endif // LINGODB_RUNTIME_DATASOURCEITERATION_H
