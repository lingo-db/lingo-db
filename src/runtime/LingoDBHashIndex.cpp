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
#include <arrow/compute/exec.h>
#include <arrow/ipc/api.h>
#include <arrow/visit_array_inline.h>
#include <type_traits>

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
// TODO use this in the visitor!!
uint64_t hashCombine(uint64_t hash1, uint64_t hash2) {
   return hash1 ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash2 >> 2));
}

struct HashColumnVisitor {
    // 1. Specific Overload for Strings and Binary (Standard & Large)
    // This covers StringArray, BinaryArray, LargeStringArray, LargeBinaryArray
    arrow::Status Visit(const arrow::StringArray& array, std::vector<uint64_t>& hashAccumulator) { return HashStringLike(array, hashAccumulator); }
    arrow::Status Visit(const arrow::BinaryArray& array, std::vector<uint64_t>& hashAccumulator) { return HashStringLike(array, hashAccumulator); }
    arrow::Status Visit(const arrow::LargeStringArray& array, std::vector<uint64_t>& hashAccumulator) { return HashStringLike(array, hashAccumulator); }
    arrow::Status Visit(const arrow::LargeBinaryArray& array, std::vector<uint64_t>& hashAccumulator) { return HashStringLike(array, hashAccumulator); }

    // 2. Specific Overload for View Types
    arrow::Status Visit(const arrow::StringViewArray& array, std::vector<uint64_t>& hashAccumulator) { return HashStringLike(array, hashAccumulator); }
    arrow::Status Visit(const arrow::BinaryViewArray& array, std::vector<uint64_t>& hashAccumulator) { return HashStringLike(array, hashAccumulator); }

    // 3. Specific Overload for Booleans (special bit-packed handling)
    arrow::Status Visit(const arrow::BooleanArray& array, std::vector<uint64_t>& hashAccumulator) {
        for (int64_t row = 0; row < array.length(); row++) {
            if (!array.IsNull(row)) {
                hashAccumulator[row] = hashCombine(hashAccumulator[row], std::hash<bool>{}(array.Value(row)));
            }
        }
        return arrow::Status::OK();
    }

    // 4. Generic Template for everything else
    template <typename ArrayType>
    arrow::Status Visit(const ArrayType& array, std::vector<uint64_t>& hashAccumulator) {
        using T = typename ArrayType::TypeClass;

        // Use if constexpr to handle types with/without c_type safely
        if constexpr (arrow::has_c_type<T>::value) {
            using C_TYPE = typename T::c_type;

            // Further check if std::hash supports this C_TYPE
            if constexpr (std::is_arithmetic_v<C_TYPE> || std::is_enum_v<C_TYPE>) {
                for (int64_t row = 0; row < array.length(); row++) {
                    if (!array.IsNull(row)) {
                        hashAccumulator[row] = hashCombine(hashAccumulator[row], std::hash<C_TYPE>{}(array.Value(row)));
                    }
                }
            } else {
                  return arrow::Status::NotImplemented();
            }
        }
        return arrow::Status::OK();
    }

private:
    template <typename T>
    arrow::Status HashStringLike(const T& array, std::vector<uint64_t>& hashAccumulator) {
        for (int64_t row = 0; row < array.length(); row++) {
            if (!array.IsNull(row)) {
                // array.GetView(row) is consistent across String, LargeString, and Views
                auto view = array.GetView(row);
                auto hashValue = std::hash<std::string_view>{}(std::string_view(view.data(), view.size()));
                hashAccumulator[row] = hashCombine(hashAccumulator[row], hashValue);
            }
        }
        return arrow::Status::OK();
    }
};
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
#ifndef MLIR_DISABLED
   if (t->num_rows() == 0) {
      throw std::runtime_error("empty table");
   }

   arrow::TableBatchReader reader(t);
   std::shared_ptr<arrow::RecordBatch> batch;

   size_t batchStartRowId = 0;

   while (reader.ReadNext(&batch).ok() && batch != nullptr) {
      int64_t n_rows = batch->num_rows();
      int64_t n_cols = batch->num_columns();

      // TODO check 0 initialized
      std::vector<uint64_t> hashArray(n_rows);

      for (auto col=0; col<n_cols; col++) {
         const auto& array = *batch->column(col);
         HashColumnVisitor visitor;
         // TODO check hashArray is passed as reference
         auto hash_ok = arrow::VisitArrayInline(array, &visitor, hashArray).ok();
         if (!hash_ok) {
           // TODO error handling
         }
     }

      for (auto row = 0ll; row < n_rows; row++) {
         Entry* entry = (Entry*) buffer.insert();
         entry->rowId = startRowId + batchStartRowId + row;
         entry->hash = hashArray[row];
         entry->next = nullptr;
      }

      batchStartRowId += n_rows;
   }

#else
   assert(false && "LingoDBHashIndex::rawInsert not supported without MLIR");
#endif
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
// FixMe removed lookup definition from here
HashIndexIteration* HashIndexAccess::lookupContinuation(size_t hash) {
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
