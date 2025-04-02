#ifndef LINGODB_RUNTIME_LINGODBHASHINDEX_H
#define LINGODB_RUNTIME_LINGODBHASHINDEX_H
#include "lingodb/runtime/Buffer.h"
#include "lingodb/runtime/RecordBatchInfo.h"
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
class LingoDBHashIndex : public Index {
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
   void setTable(catalog::LingoDBTableCatalogEntry* table) {
      this->table = table;
   }
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
class HashIndexAccess {
   LingoDBHashIndex& hashIndex;
   std::vector<size_t> colIds;

   public:
   HashIndexAccess(LingoDBHashIndex& hashIndex, std::vector<std::string> cols);
   HashIndexIteration* lookup(size_t hash);
   friend class HashIndexIteration;
};
class HashIndexIteration {
   HashIndexAccess& access;
   size_t hash;
   LingoDBHashIndex::Entry* current;

   public:
   HashIndexIteration(HashIndexAccess& access, size_t hash, LingoDBHashIndex::Entry* current) : access(access), hash(hash), current(current) {}
   bool hasNext();
   void consumeRecordBatch(RecordBatchInfo*);
   static void close(HashIndexIteration* iteration);
};

} //end namespace lingodb::runtime
#endif //LINGODB_RUNTIME_LINGODBHASHINDEX_H
