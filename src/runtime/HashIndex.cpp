#include "lingodb/runtime/HashIndex.h"

#include "lingodb/execution/Execution.h"
#include "lingodb/runtime/helpers.h"

#include <filesystem>

#include <arrow/api.h>
#include <arrow/array/array_primitive.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
namespace {
uint64_t nextPow2(uint64_t v) {
   v--;
   v |= v >> 1;
   v |= v >> 2;
   v |= v >> 4;
   v |= v >> 8;
   v |= v >> 16;
   v |= v >> 32;
   v++;
   return v;
}
} //end namespace
namespace lingodb::runtime {

void HashIndex::build() {
   size_t numRows = table->num_rows();
   arrow::TableBatchReader reader(table);
   std::shared_ptr<arrow::RecordBatch> recordBatch;
   size_t htSize = nextPow2(numRows);
   ht = FixedSizedBuffer<Entry*>::createZeroed(htSize);
   mask = htSize - 1;
   auto hashValues = std::static_pointer_cast<arrow::Int64Array>(hashData);
   size_t totalOffset = 0;
   while (reader.ReadNext(&recordBatch).ok() && recordBatch) {
      // save necessary data about record batches in table
      size_t currRecordBatch = recordBatches.size();
      recordBatches.push_back(recordBatch);

      // iterate over tuples in record batch and insert into index
      for (int additionalOffset = 0; additionalOffset != recordBatch->num_rows(); ++additionalOffset) {
         int64_t hashValue = hashValues->Value(totalOffset);
         Entry*& pos = ht[hashValue & mask];
         Entry* newEntry = (Entry*) buffer.insert();
         newEntry->next = pos;
         pos = newEntry;
         newEntry->hash = hashValue;
         newEntry->recordBatch = currRecordBatch;
         newEntry->offset = additionalOffset;
         totalOffset++;
      }
   }
}
void HashIndex::flush() {
   if (persist) {
      auto dataFile = dbDir + "/" + relation.getName() + "." + name + ".arrow";
      auto schema = std::make_shared<arrow::Schema>(arrow::FieldVector{std::make_shared<arrow::Field>("hash", arrow::int64(), false)});
      auto batch = arrow::RecordBatch::Make(schema, hashData->length(), {hashData});
      auto inputFile = arrow::io::FileOutputStream::Open(dataFile).ValueOrDie();
      auto batchWriter = arrow::ipc::MakeFileWriter(inputFile, schema).ValueOrDie();
      if (!batchWriter->WriteRecordBatch(*batch).ok() || !batchWriter->Close().ok() || !inputFile->Close().ok()) {
         throw std::runtime_error("HashIndex: could not write record batch");
      }
   }
}
void HashIndex::setPersist(bool value) {
   Index::setPersist(value);
   flush();
}
void HashIndex::computeHashes() {
   if (table->num_rows() == 0) {
      hashData = arrow::MakeEmptyArray(arrow::int64()).ValueOrDie();
   } else {
      std::string query = "select hash(";
      for (auto c : indexedColumns) {
         if (!query.ends_with("(")) {
            query += ",";
         }
         query += c;
      }
      query += ") from tmp";
      auto tmpSession = Session::createSession();
      tmpSession->getCatalog()->addTable("tmp", relation.getMetaData());
      tmpSession->getCatalog()->findRelation("tmp")->append(table);
      auto queryExecutionConfig = execution::createQueryExecutionConfig(execution::ExecutionMode::SPEED, true);
      queryExecutionConfig->parallel = false;
      std::shared_ptr<arrow::Table> result;
      queryExecutionConfig->resultProcessor = execution::createTableRetriever(result);

      auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig), *tmpSession);
      executer->fromData(query);
      executer->execute();
      result = result->CombineChunks().ValueOrDie();
      hashData = result->column(0)->chunk(0);
   }
}
void HashIndex::ensureLoaded() {
   auto dataFile = dbDir + "/" + relation.getName() + "." + name + ".arrow";
   table = relation.getTable();
   if (std::filesystem::exists(dataFile)) {
      auto inputFile = arrow::io::ReadableFile::Open(dataFile).ValueOrDie();
      auto batchReader = arrow::ipc::RecordBatchFileReader::Open(inputFile).ValueOrDie();
      assert(batchReader->num_record_batches() == 1);
      auto batch = batchReader->ReadRecordBatch(0).ValueOrDie();
      hashData = batch->column(0);
   } else {
      computeHashes();
   }
   build();
}
void HashIndex::appendRows(std::shared_ptr<arrow::Table> toAppend) {
   std::vector<std::shared_ptr<arrow::RecordBatch>> newTableBatches;

   if (table->num_rows() != 0) {
      newTableBatches.push_back(table->CombineChunksToBatch().ValueOrDie());
   }
   newTableBatches.push_back(toAppend->CombineChunksToBatch().ValueOrDie());
   table = arrow::Table::FromRecordBatches(newTableBatches).ValueOrDie();
   computeHashes();
   build();
   flush();
}
HashIndexIteration* HashIndexAccess::lookup(size_t hash) {
   return new HashIndexIteration(*this, hash, hashIndex.ht[hash & hashIndex.mask]);
}
void HashIndexIteration::close(lingodb::runtime::HashIndexIteration* iteration) {
   delete iteration;
}
bool HashIndexIteration::hasNext() {
   while (current) {
      if (current->hash == hash) {
         return true;
      }
      current = current->next;
   }
   return false;
}
void HashIndexIteration::consumeRecordBatch(lingodb::runtime::RecordBatchInfo* info) {
   auto* targetInfo = access.recordBatchInfos.at(current->recordBatch);
   memcpy(info, targetInfo, access.infoSize);
   for (size_t i = 0; i != access.colIds.size(); ++i) {
      info->columnInfo[i].offset += current->offset;
   }
   current = current->next;
}
HashIndexAccess::HashIndexAccess(lingodb::runtime::HashIndex& hashIndex, std::vector<std::string> cols) : hashIndex(hashIndex) {
   // Find column ids for relevant columns
   for (auto columnToMap : cols) {
      auto columnNames = hashIndex.table->ColumnNames();
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
   infoSize = sizeof(size_t) + colIds.size() * sizeof(ColumnInfo);

   // Prepare RecordBatchInfo for each record batch to facilitate computation for individual tuples at runtime
   for (auto& recordBatchPtr : hashIndex.recordBatches) {
      RecordBatchInfo* recordBatchInfo = static_cast<RecordBatchInfo*>(malloc(infoSize));
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
std::shared_ptr<Index> Index::createHashIndex(lingodb::runtime::IndexMetaData& metaData, lingodb::runtime::Relation& relation, std::string dbDir) {
   auto res = std::make_shared<HashIndex>(relation, metaData.columns, dbDir);
   res->name = metaData.name;
   return res;
}
} // end namespace lingodb::runtime