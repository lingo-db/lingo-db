#ifndef LINGODB_RUNTIME_LINGODBHASHINDEX_H
#define LINGODB_RUNTIME_LINGODBHASHINDEX_H
#include "lingodb/runtime/ArrowView.h"
#include "lingodb/runtime/Buffer.h"
#include "lingodb/runtime/storage/Index.h"
#include "lingodb/utility/Serialization.h"
#include <arrow/type_fwd.h>
namespace lingodb::catalog {
class LingoDBTableCatalogEntry;
} // namespace lingodb::catalog
namespace lingodb::runtime {
//todo: HashIndex maps hash to logical row id
//todo: we persist (hash, logical row id), we can even cluster by hash and store the required hashtable size
//todo: we can also create the hash index in parallel
class HashIndexIteration;
class HashIndexAccess;
class LingoDBTable;
struct GenericHasher {
   template <typename T>
   uint64_t hashValue(const T& val) {
      if constexpr (std::is_convertible_v<T, std::string_view>) {
         // Special handling for strings/char*
         return std::hash<std::string_view>{}(val);
      } else {
         // Default for numeric types/booleans
         return std::hash<T>{}(val);
      }
   }
};
class LingoDBHashIndex : public Index, GenericHasher {
   struct Entry {
      size_t hash;
      Entry* next;
      size_t rowId;
   };

   Entry** ht = nullptr;
   int64_t mask;
   runtime::FlexibleBuffer buffer;
   std::string filename;
   std::string dbDir;
   bool persist;
   catalog::LingoDBTableCatalogEntry* table = nullptr;
   LingoDBTable* tableStorage;
   std::vector<std::string> indexedColumns;
   bool loaded = false;
   //void build();
   //void computeHashes();
   void rawInsert(size_t startRowId, std::shared_ptr<arrow::Table> t);
   void rawBuild();

   public:
   virtual void setDBDir(std::string dbDir) {
      this->dbDir = dbDir;
   };
   LingoDBHashIndex(std::string filename, std::vector<std::string> indexedColumns) : buffer(16, sizeof(Entry)), filename(filename), indexedColumns(indexedColumns) {}
   void setTable(catalog::LingoDBTableCatalogEntry* table);
   void flush();
   void ensureLoaded() override;
   void appendRows(size_t startRowId, std::shared_ptr<arrow::RecordBatch> table) override;
   void bulkInsert(size_t startRowId, std::shared_ptr<arrow::Table> newRows) override;
   void setPersist(bool value) {
      persist = value;
      if (persist) {
         flush();
      }
   }
   void serialize(lingodb::utility::Serializer& serializer) const;
   static std::unique_ptr<LingoDBHashIndex> deserialize(lingodb::utility::Deserializer& deserializer);
   friend class HashIndexAccess;
   friend class HashIndexIteration;
   ~LingoDBHashIndex();
};
class HashIndexAccess : GenericHasher {
   LingoDBHashIndex& hashIndex;
   std::vector<size_t> colIds;
   std::vector<HashIndexIteration> iteration;
   HashIndexIteration* lookupContinuation(size_t hash);

   public:
   HashIndexAccess(LingoDBHashIndex& hashIndex, std::vector<std::string> cols);

   template<typename... Args>
   HashIndexIteration* lookup(Args... args) {
      uint64_t combinedHash = 0;
      ((combinedHash = hashCombine(combinedHash, hashValue(args))), ...);

      return lookupContinuation(combinedHash);
   }

   friend class HashIndexIteration;
};
class HashIndexIteration {
   HashIndexAccess& access;
   size_t hash;
   LingoDBHashIndex::Entry* current;
   std::vector<const ArrayView*> arrayViewPtrs;

   public:
   HashIndexIteration(HashIndexAccess& access, size_t hash, LingoDBHashIndex::Entry* current);
   void reset(size_t hash, LingoDBHashIndex::Entry* current) {
      this->hash = hash;
      this->current = current;
   }
   bool hasNext();
   void consumeRecordBatch(lingodb::runtime::BatchView* batchView);
};

} //end namespace lingodb::runtime
#endif //LINGODB_RUNTIME_LINGODBHASHINDEX_H
