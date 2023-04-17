#ifndef RUNTIME_EXTERNALHASHINDEX_H
#define RUNTIME_EXTERNALHASHINDEX_H

#include "DataSourceIteration.h"

#include "arrow/record_batch.h"
#include "arrow/table.h"
#include <arrow/api.h>
#include <arrow/result.h>

#include "runtime/Database.h"
#include "runtime/ExecutionContext.h"
#include "runtime/helpers.h"
namespace runtime {

struct ExternalHashIndex;

// Used to iterate over the results of a lookup
class ExternalHashIndexIteration {
   int64_t currOffset;
   ExternalHashIndex* externalHashIndex;
   ExternalHashIndexMapping* externalHashIndexMapping;

   public:
   ExternalHashIndexIteration(int64_t currOffset, ExternalHashIndex* externalIndex, ExternalHashIndexMapping* externalIndexMapping)
      : currOffset{currOffset}, externalHashIndex{externalIndex}, externalHashIndexMapping{externalIndexMapping} {}
   void consumeRecordBatch(RecordBatchInfo*);
   bool containsValue();
   static void close(ExternalHashIndexIteration* iteration);
};

// Custom hashmap that exploits the precomputed hash values and known size. Functions as index for a table.
struct ExternalHashIndex {
   std::shared_ptr<arrow::Table> table;
   std::vector<std::shared_ptr<arrow::RecordBatch>> recordBatches;
   std::vector<int64_t> firstGlobalIndexInBatch;
   std::vector<int64_t> hashValues, next, ht; // to materialize the hashValues, chaining, and pointer to first entry respectively
   int64_t mask;

   ExternalHashIndexIteration* lookup(size_t hashValue, ExternalHashIndexMapping* externalHashIndexMapping);
};

// Contains the relevant columns (and RecordBatchInfos) for a specific GetExternalHashIndex operations
struct ExternalHashIndexMapping {
   ExternalHashIndex* externalHashIndex;
   std::vector<size_t> colIds;
   std::vector<RecordBatchInfo*> recordBatchInfos;
   size_t recordBatchInfoSize;

   ExternalHashIndexIteration* lookup(size_t hashValue);

   ExternalHashIndexMapping(ExternalHashIndex* externalHashIndex, const std::vector<std::string>& mapping) : externalHashIndex{externalHashIndex} {
      // Find column ids for relevant columns
      for (auto columnToMap : mapping) {
         auto columnNames = externalHashIndex->table->ColumnNames();
         size_t columnId = 0;
         bool found = false;
         for (auto column : columnNames) {
            if (column == columnToMap) {
               colIds.push_back(columnId);
               found = true;
               break;
            }
            columnId++;
         }
         if (!found) throw std::runtime_error("column not found: " + columnToMap);
      }

      // Calculate size of RecordBatchInfo for relevant columns
      recordBatchInfoSize = sizeof(size_t) + colIds.size() * sizeof(ColumnInfo);

      // Prepare RecordBatchInfo for each record batch to facilitate computation for individual tuples at runtime
      for (auto& recordBatchPtr : externalHashIndex->recordBatches) {
         RecordBatchInfo* recordBatchInfo = static_cast<RecordBatchInfo*>(malloc(recordBatchInfoSize));
         recordBatchInfo->numRows = 1;
         for (size_t i = 0; i != colIds.size(); ++i) {
            auto colId = colIds[i];
            ColumnInfo& colInfo = recordBatchInfo->columnInfo[i];
            // Base offset for record batch, will need to add individual tuple offset in record batch
            colInfo.offset = recordBatchPtr->column_data(colId)->offset;
            // Facilitates handling of null values
            colInfo.validMultiplier = recordBatchPtr->column_data(colId)->buffers[0] ? 1 : 0;
            // Compact representation of null values (inversed)
            colInfo.validBuffer = RecordBatchInfo::getBuffer(recordBatchPtr.get(), colId, 0);
            // Pointer to fixed size data for column
            colInfo.dataBuffer = RecordBatchInfo::getBuffer(recordBatchPtr.get(), colId, 1);
            // Pointer to variable length data for column
            colInfo.varLenBuffer = RecordBatchInfo::getBuffer(recordBatchPtr.get(), colId, 2);
         }
         recordBatchInfos.push_back(recordBatchInfo);
      }
   }

   ~ExternalHashIndexMapping() {
      for (auto* ptr : recordBatchInfos) free(ptr);
   }
};

// Stores all created indices and allows to retrieve them by name
class ExternalHashIndexManager {
   std::unordered_map<std::string, ExternalHashIndex> existingIndices;

   public:
   void addIndex(std::string name, std::shared_ptr<arrow::Table> table, std::shared_ptr<TableMetaData> metaData);
   static ExternalHashIndexMapping* get(runtime::ExecutionContext* executionContext, runtime::VarLen32 description);
   ExternalHashIndexMapping* getIndex(const std::string& name, const std::vector<std::string>& mapping);
};
} // namespace runtime

#endif // RUNTIME_EXTERNALHASHINDEX_H