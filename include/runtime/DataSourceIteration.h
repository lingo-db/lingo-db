#ifndef RUNTIME_DATASOURCEITERATION_H
#define RUNTIME_DATASOURCEITERATION_H
#include "runtime/ExecutionContext.h"
#include "runtime/helpers.h"
namespace runtime {
class DataSource {
   public:
   virtual std::shared_ptr<arrow::RecordBatch> getNext() = 0;
   virtual ~DataSource() {}
};
class DataSourceIteration {
   std::shared_ptr<arrow::RecordBatch> currChunk;
   std::shared_ptr<DataSource> dataSource;
   std::vector<size_t> colIds;

   public:
   DataSourceIteration(const std::shared_ptr<DataSource>& dataSource, const std::vector<size_t>& colIds);

   struct ColumnInfo {
      size_t offset;
      size_t validMultiplier;
      uint8_t* validBuffer;
      uint8_t* dataBuffer;
      uint8_t* varLenBuffer;
   };
   struct RecordBatchInfo {
      size_t numRows;
      ColumnInfo columnInfo[];
   };
   static DataSourceIteration* start(ExecutionContext* executionContext, runtime::VarLen32 description);
   bool isValid();
   void next();
   void access(RecordBatchInfo* info);
   static void end(DataSourceIteration*);
};
} // end namespace runtime
#endif // RUNTIME_DATASOURCEITERATION_H
