#ifndef LINGODB_RUNTIME_STORAGE_INDEX_H
#define LINGODB_RUNTIME_STORAGE_INDEX_H
#include <arrow/type_fwd.h>
namespace lingodb::runtime {
class Index {
   public:
   virtual void ensureLoaded() = 0;
   virtual void bulkInsert(size_t startRowId, std::shared_ptr<arrow::Table> newRows) = 0;
   virtual void appendRows(size_t startRowId, std::shared_ptr<arrow::RecordBatch> newRows) = 0;
   virtual ~Index() {}
};
} // namespace lingodb::runtime
#endif //LINGODB_RUNTIME_STORAGE_INDEX_H
