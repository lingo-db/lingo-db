#ifndef LINGODB_RUNTIME_STORAGE_LINGODBTABLE_H
#define LINGODB_RUNTIME_STORAGE_LINGODBTABLE_H

#include "TableStorage.h"
#include "lingodb/catalog/TableCatalogEntry.h"

#include <functional>
#include <string>

namespace lingodb::runtime {
class LingoDBTable : public TableStorage {
   public:
   class TableChunk {
      std::shared_ptr<arrow::RecordBatch> internalData;
      size_t startRowId;
      size_t numRows;
      std::vector<ColumnInfo> columnInfo;

      public:
      TableChunk(std::shared_ptr<arrow::RecordBatch> data, size_t startRowId);

      const std::shared_ptr<arrow::RecordBatch>& data() const {
         return internalData;
      }
      const ColumnInfo& getColumnInfo(size_t colId) const {
         return columnInfo[colId];
      }
      size_t getNumRows() {
         return numRows;
      }
      friend class LingoDBTable;
   };

   private:
   bool persist;
   std::string fileName;
   std::string dbDir;
   catalog::Sample sample;
   std::shared_ptr<arrow::Schema> schema;
   std::vector<TableChunk> tableData;

   struct TransparentStringHasher : std::hash<std::string>, std::hash<std::string_view> {
      using is_transparent = void;
      using std::hash<std::string>::operator();
      using std::hash<std::string_view>::operator();
   };
   using ColumnStatisticsMap = std::unordered_map<std::string, catalog::ColumnStatistics, TransparentStringHasher, std::equal_to<>>;
   ColumnStatisticsMap columnStatistics;
   size_t numRows;
   bool loaded = false;
   //todo: somehow we must be aware of the indices that are built on this table, and update them...
   public:
   LingoDBTable(std::string fileName, std::shared_ptr<arrow::Schema> schema);
   LingoDBTable(std::string fileName, std::shared_ptr<arrow::Schema> schema, size_t numRows, catalog::Sample sample, ColumnStatisticsMap columnStatistics) : persist(false), fileName(std::move(fileName)), sample(std::move(sample)), schema(std::move(schema)), columnStatistics(std::move(columnStatistics)), numRows(numRows) {}
   void setPersist(bool persist) {
      this->persist = persist;
      if (persist) {
         flush();
      }
   }
   size_t nextRowId() override {
      return numRows;
   }
   size_t getColIndex(std::string colName);
   std::unique_ptr<scheduler::Task> createScanTask(const ScanConfig& scanConfig) override;
   const catalog::Sample& getSample() const {
      return sample;
   }
   const catalog::ColumnStatistics& getColumnStatistics(std::string_view column) const;
   size_t getNumRows() const {
      return numRows;
   }
   ~LingoDBTable() = default;

   //flushes the data to disk
   void flush();
   //ensures that the data is loaded
   void ensureLoaded();
   virtual void setDBDir(std::string dbDir) {
      this->dbDir = dbDir;
   };

   void serialize(lingodb::utility::Serializer& serializer) const;
   static std::unique_ptr<LingoDBTable> deserialize(lingodb::utility::Deserializer& deserializer);
   void append(const std::vector<std::shared_ptr<arrow::RecordBatch>>& toAppend) override;
   void append(const std::shared_ptr<arrow::Table>& toAppend) override;
   static std::unique_ptr<LingoDBTable> create(const catalog::CreateTableDef& def);
   std::pair<const TableChunk*, size_t> getByRowId(size_t rowId) const;

   std::shared_ptr<arrow::DataType> getColumnStorageType(std::string_view columnName) const override;
};
} // namespace lingodb::runtime

#endif //LINGODB_RUNTIME_STORAGE_LINGODBTABLE_H
