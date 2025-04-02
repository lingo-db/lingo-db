#ifndef LINGODB_RUNTIME_STORAGE_TABLESTORAGE_H
#define LINGODB_RUNTIME_STORAGE_TABLESTORAGE_H
#include "lingodb/runtime/RecordBatchInfo.h"
#include "lingodb/scheduler/Task.h"

#include <functional>
#include <memory>

#include <arrow/type_fwd.h>

namespace lingodb::runtime {
struct ScanConfig {
   bool parallel;
   std::vector<std::string> columns;
   std::function<void(lingodb::runtime::RecordBatchInfo*)> cb;
};
class TableStorage {
   public:
   virtual std::shared_ptr<arrow::DataType> getColumnStorageType(const std::string& columnName) const = 0;
   virtual std::unique_ptr<scheduler::Task> createScanTask(const ScanConfig& scanConfig) = 0;
   virtual void append(const std::vector<std::shared_ptr<arrow::RecordBatch>>& toAppend) = 0;
   virtual size_t nextRowId() = 0;
   virtual void append(const std::shared_ptr<arrow::Table>& toAppend) = 0;
   virtual ~TableStorage() = default;
};
} // namespace lingodb::runtime
#endif //LINGODB_RUNTIME_STORAGE_TABLESTORAGE_H
