#include "lingodb/runtime/LingoDBHashIndex.h"

#include "lingodb/catalog/Defs.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/runtime/Hash.h"
#include "lingodb/runtime/storage/LingoDBTable.h"

#include <filesystem>
#include <fstream>
#include <vector>

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
   }
   auto batch = t->CombineChunksToBatch().ValueOrDie();
   const int64_t numRows = batch->num_rows();

   std::vector<uint64_t> totalHash(static_cast<size_t>(numRows), 0);
   bool isFirstColumn = true;
   for (const auto& colName : indexedColumns) {
      auto arr = batch->GetColumnByName(std::string(colName));
      assert(arr->length() == numRows);
      dbHashApplyColumn(totalHash, *arr, isFirstColumn);
      isFirstColumn = false;
   }
   for (int64_t row = 0; row < numRows; ++row) {
      Entry* entry = (Entry*) buffer.insert();
      entry->rowId = static_cast<size_t>(row) + startRowId;
      entry->hash = totalHash[static_cast<size_t>(row)];
      entry->next = nullptr;
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
   auto& iter = iteration[lingodb::scheduler::currentWorkerId()];
   iter.reset(hash, hashIndex.ht[hash & hashIndex.mask]);
   return &iter;
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

void HashIndexIteration::consumeRecordBatch(lingodb::runtime::BatchView* batchView) {
   auto currRowId = current->rowId;
   auto [tableChunk, offset] = access.hashIndex.tableStorage->getByRowId(currRowId);
   batchView->length = 1;
   batchView->offset = offset;
   for (size_t i = 0; i != access.colIds.size(); ++i) {
      auto colId = access.colIds[i];
      arrayViewPtrs[i] = tableChunk->getArrayView(colId);
   }
   batchView->arrays = arrayViewPtrs.data();
   current = current->next;
}
HashIndexAccess::HashIndexAccess(lingodb::runtime::LingoDBHashIndex& hashIndex, std::vector<std::string> cols) : hashIndex(hashIndex) {
   for (const auto& c : cols) {
      colIds.push_back(dynamic_cast<LingoDBTable*>(&hashIndex.table->getTableStorage())->getColIndex(c));
   }
   for (auto i = 0ull; i < lingodb::scheduler::getNumWorkers(); i++) {
      iteration.push_back(HashIndexIteration(*this, 0, nullptr));
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

HashIndexIteration::HashIndexIteration(HashIndexAccess& access, size_t hash, LingoDBHashIndex::Entry* current) : access(access), hash(hash), current(current) {
   arrayViewPtrs.resize(access.colIds.size());
}

} // end namespace lingodb::runtime
