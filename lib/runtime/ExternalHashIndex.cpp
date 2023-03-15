#include "runtime/ExternalHashIndex.h"
#include <json.h>

void runtime::ExternalHashIndexManager::addIndex(std::string name, std::shared_ptr<arrow::Table> table, std::shared_ptr<TableMetaData> metaData) {
   // only create index on primary key
   if (metaData->getPrimaryKey().empty() || table->num_rows() == 0) return;

   // Create fixed size hash table for known number of elements in table
   existingIndices[name] = ExternalHashIndex();
   existingIndices[name].table = table;
   int64_t size = 1ull << ((8*sizeof(uint64_t)) - __builtin_clzl(table->num_rows())); // next power of 2
   existingIndices[name].hashValues = std::vector<int64_t>(table->num_rows());
   existingIndices[name].next = std::vector<int64_t>(table->num_rows(), -1);
   existingIndices[name].ht = std::vector<int64_t>(size, -1);
   existingIndices[name].mask = size - 1;

   ExternalHashIndex& externalHashIndex = existingIndices[name];

   arrow::TableBatchReader reader(table);
   std::shared_ptr<arrow::RecordBatch> recordBatch;

   // for each tuple calculate and save hash value
   int64_t firstGlobalIndexInBatch = 0, mask = existingIndices[name].mask, currOffset = 0;
   while (reader.ReadNext(&recordBatch).Equals(arrow::Status::OK()) && recordBatch) {
      // save necessary data about record batches in table
      externalHashIndex.recordBatches.push_back(recordBatch);
      externalHashIndex.firstGlobalIndexInBatch.push_back(firstGlobalIndexInBatch);

      auto column = recordBatch->GetColumnByName("primaryKeyHashValue");
      auto hashValues = std::static_pointer_cast<arrow::Int64Array>(column);

      // iterate over tuples in record batch and insert into index
      for (int additionalOffset = 0; additionalOffset != recordBatch->num_rows(); ++additionalOffset) {
         int64_t hashValue = hashValues->Value(additionalOffset);
         int64_t nextPtr = externalHashIndex.ht[hashValue & mask];

         externalHashIndex.hashValues[currOffset] = hashValue;    // materialize hashValue in buffer
         externalHashIndex.next[currOffset] = nextPtr;            // set next pointer to enable chaining
         externalHashIndex.ht[hashValue & mask] = currOffset;     // set entry pointer to first entry in chain
         currOffset++;
      }
      firstGlobalIndexInBatch += recordBatch->num_rows();
   }
}

runtime::ExternalHashIndexMapping* runtime::ExternalHashIndexManager::get(runtime::ExecutionContext* executionContext, runtime::VarLen32 description) {
   nlohmann::json descr = nlohmann::json ::parse(description.str());
   std::string tableName = descr["externalHashIndex"];
   if (!executionContext->db){
      throw std::runtime_error("no database attached");
   }
   if (!executionContext->db->hasTable(tableName)){
      throw std::runtime_error{"could not find table"};
   }
   std::vector<std::string> mapping;
   for (auto m : descr["mapping"].get<nlohmann::json::object_t>()){
      mapping.push_back(m.second.get<std::string>());
   }
   return executionContext->db->getIndex(tableName, mapping);
}

runtime::ExternalHashIndexMapping* runtime::ExternalHashIndexManager::getIndex(const std::string& name, const std::vector<std::string>& mapping) {
   return new ExternalHashIndexMapping{&existingIndices[name], mapping};
}

runtime::ExternalHashIndexIteration* runtime::ExternalHashIndex::lookup(size_t hashValue, ExternalHashIndexMapping* externalHashIndexMapping) {
   return new ExternalHashIndexIteration(ht[hashValue & mask], this, externalHashIndexMapping);
}

runtime::ExternalHashIndexIteration* runtime::ExternalHashIndexMapping::lookup(size_t hashValue) {
   return externalHashIndex->lookup(hashValue, this);
}

bool runtime::ExternalHashIndexIteration::containsValue() {
   return currOffset != -1;
}

void runtime::ExternalHashIndexIteration::consumeRecordBatch(runtime::RecordBatchInfo* info) {
   // find and copy record batch for accessed tuple at currOffset
   auto recordBatchIterator = std::upper_bound(externalHashIndex->firstGlobalIndexInBatch.begin(), externalHashIndex->firstGlobalIndexInBatch.end(), currOffset);
   int recordBatchIndex = std::distance(externalHashIndex->firstGlobalIndexInBatch.begin(), recordBatchIterator);
   if (recordBatchIterator == externalHashIndex->firstGlobalIndexInBatch.end() || externalHashIndex->firstGlobalIndexInBatch[recordBatchIndex] != currOffset) recordBatchIndex--;
   memcpy(info, externalHashIndexMapping->recordBatchInfos[recordBatchIndex], externalHashIndexMapping->recordBatchInfoSize);

   // update copied record with offset in record batch
   int recordBatchOffset = currOffset - externalHashIndex->firstGlobalIndexInBatch[recordBatchIndex];
   for (size_t i=0; i!=externalHashIndexMapping->colIds.size(); ++i){
      info->columnInfo[i].offset += recordBatchOffset;
   }

   // set to next value in chain
   currOffset = externalHashIndex->next[currOffset];
}

void runtime::ExternalHashIndexIteration::close(runtime::ExternalHashIndexIteration* iteration) {
   delete iteration;
}

