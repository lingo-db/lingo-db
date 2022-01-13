#ifndef RUNTIME_METADATA_H
#define RUNTIME_METADATA_H
#include <optional>
#include <unordered_map>

#include <arrow/record_batch.h>
namespace runtime {
class ColumnMetaData {
   std::optional<size_t> distinctValues;

   public:
   const std::optional<size_t>& getDistinctValues() const;
   void setDistinctValues(const std::optional<size_t>& distinctValues);
};
class TableMetaData {
   bool present;
   size_t numRows;
   std::vector<std::string> primaryKey;
   std::unordered_map<std::string, std::shared_ptr<ColumnMetaData>> columns;
   std::shared_ptr<arrow::RecordBatch> sample;

   public:
   TableMetaData() : present(false) {}
   size_t getNumRows() const {
      return numRows;
   }

   const std::vector<std::string>& getPrimaryKey() const {
      return primaryKey;
   }
   const std::shared_ptr<ColumnMetaData> getColumnMetaData(const std::string& name) const {
      return columns.at(name);
   }
   const std::shared_ptr<arrow::RecordBatch>& getSample() const {
      return sample;
   }
   static std::shared_ptr<TableMetaData> deserialize(std::string);
   std::string serialize() const;
   static std::shared_ptr<TableMetaData> create(const std::string& json, const std::string& name, std::shared_ptr<arrow::RecordBatch> sample);
   bool isPresent() const;
};
} // end namespace runtime

#endif // RUNTIME_METADATA_H
