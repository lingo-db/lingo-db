#ifndef LINGODB_RUNTIME_HASHINDEX_H
#define LINGODB_RUNTIME_HASHINDEX_H
#include "Index.h"
#include "lingodb/runtime/Buffer.h"
#include "lingodb/runtime/RecordBatchInfo.h"
#include <arrow/type_fwd.h>
namespace lingodb::runtime {
class HashIndexIteration;
class HashIndexAccess;
class HashIndex : public Index {
   struct Entry {
      size_t hash;
      Entry* next;
      size_t recordBatch;
      size_t offset;
   };

   Entry** ht;
   int64_t mask;
   runtime::FlexibleBuffer buffer;
   std::shared_ptr<arrow::Array> hashData;
   std::shared_ptr<arrow::Table> table;
   std::vector<std::shared_ptr<arrow::RecordBatch>> recordBatches;
   std::string dbDir;
   void build();
   void computeHashes();

   public:
   HashIndex(Relation& r, std::vector<std::string> keyColumns, std::string dbDir) : Index(r, keyColumns), buffer(16, sizeof(Entry)), dbDir(dbDir) {}
   void flush();
   void ensureLoaded() override;
   void appendRows(std::shared_ptr<arrow::Table> table) override;
   void setPersist(bool value) override;
   friend class HashIndexAccess;
   friend class HashIndexIteration;
};
class HashIndexAccess {
   HashIndex& hashIndex;
   std::vector<size_t> colIds;
   std::vector<RecordBatchInfo*> recordBatchInfos;
   size_t infoSize;

   public:
   HashIndexAccess(HashIndex& hashIndex, std::vector<std::string> cols);
   HashIndexIteration* lookup(size_t hash);
   friend class HashIndexIteration;
};
class HashIndexIteration {
   HashIndexAccess& access;
   size_t hash;
   HashIndex::Entry* current;

   public:
   HashIndexIteration(HashIndexAccess& access, size_t hash, HashIndex::Entry* current) : access(access), hash(hash), current(current) {}
   bool hasNext();
   void consumeRecordBatch(RecordBatchInfo*);
   static void close(HashIndexIteration* iteration);
};

} //end namespace lingodb::runtime
#endif //LINGODB_RUNTIME_HASHINDEX_H
