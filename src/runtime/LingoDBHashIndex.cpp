#include "lingodb/runtime/LingoDBHashIndex.h"

#include "lingodb/catalog/Defs.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/execution/Execution.h"
#include "lingodb/runtime/helpers.h"
#include "lingodb/runtime/storage/LingoDBTable.h"

#include <filesystem>
#include <fstream>

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

void LingoDBHashIndex::rawBuild() {
   size_t numRows = buffer.getLen();
   size_t htSize = nextPow2(std::max(1ul, numRows));
   if (ht) {
      FixedSizedBuffer<Entry*>::deallocate(ht, mask + 1);
   }
   ht = FixedSizedBuffer<Entry*>::createZeroed(htSize);
   mask = htSize - 1;
   buffer.iterate([&](uint8_t* e) {
      Entry* entry = reinterpret_cast<Entry*>(e);
      size_t hash = entry->hash;
      size_t idx = hash & mask;
      entry->next = ht[idx];
      ht[idx] = entry;
   });
}

void LingoDBHashIndex::rawInsert(size_t startRowId, std::shared_ptr<arrow::Table> t) {
   if (t->num_rows() == 0) {
      throw std::runtime_error("empty table");
   } else {
      std::string query = "select row_number() over() -1 as rowid, hash(";
      for (auto c : indexedColumns) {
         if (!query.ends_with("(")) {
            query += ",";
         }
         query += c;
      }
      query += ") as hash from tmp";
      auto tmpSession = Session::createSession();
      auto createTableDef = catalog::CreateTableDef{"tmp", table->getColumns(), {}};
      tmpSession->getCatalog()->insertEntry(catalog::LingoDBTableCatalogEntry::createFromCreateTable(createTableDef));
      tmpSession->getCatalog()->getTypedEntry<catalog::LingoDBTableCatalogEntry>("tmp").value()->getTableStorage().append(t);
      auto queryExecutionConfig = execution::createQueryExecutionConfig(execution::ExecutionMode::SPEED, true);
      queryExecutionConfig->parallel = false;
      std::shared_ptr<arrow::Table> result;
      queryExecutionConfig->resultProcessor = execution::createTableRetriever(result);

      auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig), *tmpSession);
      executer->fromData(query);
      scheduler::awaitChildTask(std::make_unique<execution::QueryExecutionTask>(std::move(executer)));
      auto asBatch = result->CombineChunksToBatch().ValueOrDie();
      auto hashColumn = std::static_pointer_cast<arrow::Int64Array>(asBatch->GetColumnByName("hash"));
      auto rowIdColumn = std::static_pointer_cast<arrow::Int64Array>(asBatch->GetColumnByName("rowid"));
      for (auto i = 0ll; i < asBatch->num_rows(); i++) {
         Entry* entry = (Entry*) buffer.insert();
         entry->rowId = rowIdColumn->Value(i) + startRowId;
         entry->hash = hashColumn->Value(i);
         entry->next = nullptr;
      }
   }
}

void LingoDBHashIndex::flush() {
   if (persist) {
      ensureLoaded();
      auto dataFile = dbDir + "/" + filename;
      std::ofstream file(dataFile, std::ios::binary);
      if (!file) {
         throw std::runtime_error("could not open file");
      }
      size_t numEntries = buffer.getLen();
      file.write(reinterpret_cast<const char*>(&numEntries), sizeof(numEntries));
      buffer.iterate([&](uint8_t* e) {
         Entry* entry = reinterpret_cast<Entry*>(e);
         file.write(reinterpret_cast<const char*>(&entry->hash), sizeof(entry->hash));
         file.write(reinterpret_cast<const char*>(&entry->rowId), sizeof(entry->rowId));
      });
      file.close();
   }
}

void LingoDBHashIndex::ensureLoaded() {
   if (loaded) {
      return;
   }
   auto dataFile = dbDir + "/" + filename;
   if (std::filesystem::exists(dataFile)) {
      std::ifstream file(dataFile, std::ios::binary);
      size_t numEntries;
      file.read(reinterpret_cast<char*>(&numEntries), sizeof(numEntries));
      for (size_t i = 0; i < numEntries; i++) {
         auto* entry = (Entry*) buffer.insert();
         file.read(reinterpret_cast<char*>(&entry->hash), sizeof(entry->hash));
         file.read(reinterpret_cast<char*>(&entry->rowId), sizeof(entry->rowId));
         entry->next = nullptr;
      }
      rawBuild();
   }
   loaded = true;
}
void LingoDBHashIndex::appendRows(size_t startRowId, std::shared_ptr<arrow::RecordBatch> table) {
   auto astable = arrow::Table::FromRecordBatches({table}).ValueOrDie();
   rawInsert(startRowId, astable);
   rawBuild();
   flush();
}
void LingoDBHashIndex::bulkInsert(size_t startRowId, std::shared_ptr<arrow::Table> newRows) {
   rawInsert(startRowId, newRows);
   rawBuild();
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
void LingoDBHashIndex::setTable(catalog::LingoDBTableCatalogEntry* table) {
   this->table = table;
   tableStorage = dynamic_cast<LingoDBTable*>(&table->getTableStorage());
}

void HashIndexIteration::consumeRecordBatch(lingodb::runtime::RecordBatchInfo* info) {
   auto currRowId = current->rowId;
   auto [tableChunk, offset] = access.hashIndex.tableStorage->getByRowId(currRowId);
   info->numRows = 1;
   for (size_t i = 0; i != access.colIds.size(); ++i) {
      auto colId = access.colIds[i];
      auto* colInfo = &info->columnInfo[i];
      std::memcpy(colInfo, &tableChunk->getColumnInfo(colId), sizeof(ColumnInfo));
      colInfo->offset += offset;
   }
   current = current->next;
}
HashIndexAccess::HashIndexAccess(lingodb::runtime::LingoDBHashIndex& hashIndex, std::vector<std::string> cols) : hashIndex(hashIndex) {
   for (const auto& c : cols) {
      colIds.push_back(dynamic_cast<LingoDBTable*>(&hashIndex.table->getTableStorage())->getColIndex(c));
   }
}
LingoDBHashIndex::~LingoDBHashIndex() {
   if (ht) {
      runtime::FixedSizedBuffer<Entry*>::deallocate(ht, mask + 1);
   }
}
void LingoDBHashIndex::serialize(lingodb::utility::Serializer& serializer) const {
   serializer.writeProperty(0, filename);
   serializer.writeProperty(1, indexedColumns);
}
std::unique_ptr<LingoDBHashIndex> LingoDBHashIndex::deserialize(lingodb::utility::Deserializer& deserializer) {
   auto filename = deserializer.readProperty<std::string>(0);
   auto indexedColumns = deserializer.readProperty<std::vector<std::string>>(1);
   return std::make_unique<LingoDBHashIndex>(filename, indexedColumns);
}
} // end namespace lingodb::runtime
