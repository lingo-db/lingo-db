#include "lingodb/catalog/Catalog.h"
#include "lingodb/catalog/Defs.h"
#include "lingodb/catalog/IndexCatalogEntry.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/catalog/Types.h"
#include "lingodb/runtime/LingoDBHashIndex.h"
#include "lingodb/runtime/storage/LingoDBTable.h"
#include "lingodb/utility/Serialization.h"

#include <arrow/buffer.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/type.h>

#include <filesystem>

namespace lingodb::catalog {
Catalog Catalog::deserialize(lingodb::utility::Deserializer& deSerializer) {
   Catalog res;
   res.entries = deSerializer.readProperty<std::unordered_map<std::string, std::shared_ptr<CatalogEntry>>>(1);
   return res;
}
void Catalog::serialize(lingodb::utility::Serializer& serializer) const {
   serializer.writeProperty(1, entries);
}

void CatalogEntry::serialize(lingodb::utility::Serializer& serializer) const {
   serializer.writeProperty(1, entryType);
   serializeEntry(serializer);
}
std::shared_ptr<CatalogEntry> CatalogEntry::deserialize(lingodb::utility::Deserializer& deserializer) {
   auto entryType = deserializer.readProperty<CatalogEntryType>(1);
   switch (entryType) {
      case CatalogEntryType::INVALID_ENTRY:
         return nullptr;
      case CatalogEntryType::LINGODB_TABLE_ENTRY:
         return LingoDBTableCatalogEntry::deserialize(deserializer);
      case CatalogEntryType::LINGODB_HASH_INDEX_ENTRY:
         return LingoDBHashIndexEntry::deserialize(deserializer);
   }
}
LingoDBTableCatalogEntry::LingoDBTableCatalogEntry(std::string name, std::vector<Column> columns, std::vector<std::string> primaryKey, std::vector<std::string> indices, std::unique_ptr<runtime::LingoDBTable> impl) : TableCatalogEntry(CatalogEntryType::LINGODB_TABLE_ENTRY, name, columns, primaryKey, indices), impl(std::move(impl)) {}

void LingoDBTableCatalogEntry::serializeEntry(lingodb::utility::Serializer& serializer) const {
   serializer.writeProperty(2, name);
   serializer.writeProperty(3, columns.size());
   arrow::FieldVector fields;
   for (const auto& column : columns) {
      serializer.writeProperty(4, column);
   }
   serializer.writeProperty(8, primaryKey);
   serializer.writeProperty(9, indices);

   serializer.writeProperty(10, impl);
}
std::shared_ptr<LingoDBTableCatalogEntry> LingoDBTableCatalogEntry::deserialize(lingodb::utility::Deserializer& deserializer) {
   auto name = deserializer.readProperty<std::string>(2);
   auto numColumns = deserializer.readProperty<size_t>(3);
   std::vector<Column> columns;
   for (size_t i = 0; i < numColumns; i++) {
      auto column = deserializer.readProperty<Column>(4);
      columns.push_back(column);
   }
   auto primaryKey = deserializer.readProperty<std::vector<std::string>>(8);
   auto indices = deserializer.readProperty<std::vector<std::string>>(9);
   auto rawTable = deserializer.readProperty<std::unique_ptr<lingodb::runtime::LingoDBTable>>(10);
   return std::make_shared<LingoDBTableCatalogEntry>(name, columns, primaryKey, indices, std::move(rawTable));
}

runtime::TableStorage& LingoDBTableCatalogEntry::getTableStorage() {
   return *impl;
}

std::shared_ptr<LingoDBTableCatalogEntry> LingoDBTableCatalogEntry::createFromCreateTable(const CreateTableDef& def) {
   auto impl = runtime::LingoDBTable::create(def);
   auto res = std::make_shared<LingoDBTableCatalogEntry>(def.name, def.columns, def.primaryKey, std::vector<std::string>{}, std::move(impl));
   return res;
}

const ColumnStatistics& LingoDBTableCatalogEntry::getColumnStatistics(std::string column) const {
   return impl->getColumnStatistics(column);
}
size_t LingoDBTableCatalogEntry::getNumRows() const {
   return impl->getNumRows();
}
const Sample& LingoDBTableCatalogEntry::getSample() const {
   return impl->getSample();
}

void Sample::serialize(utility::Serializer& serializer) const {
   std::shared_ptr<arrow::ResizableBuffer> buffer = arrow::AllocateResizableBuffer(0).ValueOrDie();
   std::unique_ptr<arrow::io::BufferOutputStream> bufferOutputStream = std::make_unique<arrow::io::BufferOutputStream>(buffer);
   std::shared_ptr<arrow::ipc::RecordBatchWriter> recordBatchWriter = arrow::ipc::MakeStreamWriter(bufferOutputStream.get(), sampleData->schema()).ValueOrDie();

   if (!recordBatchWriter->WriteRecordBatch(*sampleData).ok() || !recordBatchWriter->Close().ok()) {
      throw std::runtime_error("MetaData: Failed to write record batch");
   }
   auto resBuffer = bufferOutputStream->Finish().ValueOrDie();
   serializer.writeProperty(1, std::string_view(*resBuffer));
}
Sample Sample::deserialize(utility::Deserializer& deserializer) {
   auto data = deserializer.readProperty<std::string>(1);
   auto bufferReader = arrow::io::BufferReader::FromString(data);
   auto recordBatchReader = arrow::ipc::RecordBatchStreamReader::Open(bufferReader.get()).ValueOrDie();
   std::shared_ptr<arrow::RecordBatch> batch;
   if (!recordBatchReader->ReadNext(&batch).ok()) {
      throw std::runtime_error("Failed to deserialize sample");
   }
   return {batch};
}
Sample::Sample(std::shared_ptr<arrow::Schema> schema) {
   sampleData = arrow::RecordBatch::MakeEmpty(schema).ValueOrDie();
}

ColumnStatistics ColumnStatistics::deserialize(utility::Deserializer& deserializer) {
   auto numDistinctValues = deserializer.readProperty<std::optional<size_t>>(1);
   return {numDistinctValues};
}
void ColumnStatistics::serialize(utility::Serializer& serializer) const {
   serializer.writeProperty(1, numDistinctValues);
}

void TableMetaDataProvider::serialize(utility::Serializer& serializer) const {
   serializer.writeProperty(1, getSample());
   serializer.writeProperty(2, getNumRows());
   serializer.writeProperty(3, getPrimaryKey());
   serializer.writeProperty(4, getColumnNames().size());
   for (const auto& column : getColumnNames()) {
      serializer.writeProperty(5, column);
      serializer.writeProperty(6, getColumnStatistics(column));
   }
   serializer.writeProperty(7, getIndices());
}

class StoredTableMetaData : public TableMetaDataProvider {
   std::unique_ptr<Sample> sample;
   std::vector<std::string> primaryKey;
   std::vector<std::string> columnNames;
   std::vector<std::unique_ptr<ColumnStatistics>> columnStatistics;
   size_t numRows;
   std::vector<std::pair<std::string, std::vector<std::string>>> indices;

   public:
   StoredTableMetaData(std::unique_ptr<Sample> sample, size_t numRows, std::vector<std::string> primaryKey, std::vector<std::string> columnNames, std::vector<std::unique_ptr<ColumnStatistics>> columnStatistics, std::vector<std::pair<std::string, std::vector<std::string>>> indices) : sample(std::move(sample)), primaryKey(std::move(primaryKey)), columnNames(std::move(columnNames)), columnStatistics(std::move(columnStatistics)), numRows(numRows), indices(indices) {}
   Sample& getSample() const override { return *sample; }
   size_t getNumRows() const override { return numRows; }
   std::vector<std::string> getPrimaryKey() const override { return primaryKey; }
   std::vector<std::string> getColumnNames() const override { return columnNames; }
   ColumnStatistics& getColumnStatistics(std::string column) const override {
      for (size_t i = 0; i < columnNames.size(); i++) {
         if (columnNames[i] == column) {
            return *columnStatistics[i];
         }
      }
      throw std::runtime_error("MetaData: Column not found");
   }
   std::vector<std::pair<std::string, std::vector<std::string>>> getIndices() const override {
      return indices;
   }
   ~StoredTableMetaData() override = default;
};

std::shared_ptr<TableMetaDataProvider> TableMetaDataProvider::deserialize(utility::Deserializer& deserializer) {
   auto sample = deserializer.readProperty<std::unique_ptr<Sample>>(1);
   auto numRows = deserializer.readProperty<size_t>(2);
   auto primaryKey = deserializer.readProperty<std::vector<std::string>>(3);
   auto numColumns = deserializer.readProperty<size_t>(4);
   std::vector<std::string> columnNames;
   std::vector<std::unique_ptr<ColumnStatistics>> columnStatistics;
   for (size_t i = 0; i < numColumns; i++) {
      columnNames.push_back(deserializer.readProperty<std::string>(5));
      columnStatistics.push_back(deserializer.readProperty<std::unique_ptr<ColumnStatistics>>(6));
   }
   auto indices = deserializer.readProperty<std::vector<std::pair<std::string, std::vector<std::string>>>>(7);
   return std::make_shared<StoredTableMetaData>(std::move(sample), numRows, std::move(primaryKey), std::move(columnNames), std::move(columnStatistics), std::move(indices));
}
void Column::serialize(utility::Serializer& serializer) const {
   serializer.writeProperty(1, columnName);
   serializer.writeProperty(2, logicalType);
   serializer.writeProperty(3, isNullable);
}
Column Column::deserialize(utility::Deserializer& deserializer) {
   auto columnName = deserializer.readProperty<std::string>(1);
   auto logicalType = deserializer.readProperty<Type>(2);
   auto isNullable = deserializer.readProperty<bool>(3);
   return Column(columnName, logicalType, isNullable);
}
void Catalog::persist() {
   if (shouldPersist) {
      if (!std::filesystem::exists(dbDir)) {
         throw std::runtime_error("Catalog: dbDir does not exist");
      }
      for (auto& entry : entries) {
         entry.second->flush();
      }
      lingodb::utility::FileByteWriter reader(dbDir + "/db.lingodb");
      lingodb::utility::Serializer serializer(reader);
      serializer.writeProperty(0, *this);
   }
}
std::shared_ptr<Catalog> Catalog::create(std::string dbDir, bool eagerLoading) {
   if (!std::filesystem::exists(dbDir)) {
      std::filesystem::create_directories(dbDir);
   }
   if (!std::filesystem::exists(dbDir + "/db.lingodb")) {
      auto res = std::make_shared<Catalog>();
      res->dbDir = dbDir;
      return res;
   } else {
      lingodb::utility::FileByteReader reader(dbDir + "/db.lingodb");
      lingodb::utility::Deserializer deserializer(reader);
      auto res = std::make_shared<Catalog>(deserializer.readProperty<Catalog>(0));
      res->dbDir = dbDir;
      for (auto& entry : res->entries) {
         entry.second->setDBDir(dbDir);
         entry.second->setCatalog(&*res);
      }
      if (eagerLoading) {
         for (auto& entry : res->entries) {
            entry.second->ensureFullyLoaded();
         }
      }
      return res;
   }
}
std::shared_ptr<Catalog> Catalog::createEmpty() {
   return std::make_shared<Catalog>();
}

void LingoDBTableCatalogEntry::flush() {
   impl->flush();
}
void LingoDBTableCatalogEntry::setShouldPersist(bool shouldPersist) {
   impl->setPersist(shouldPersist);
}
void LingoDBTableCatalogEntry::setDBDir(std::string dbDir) {
   impl->setDBDir(dbDir);
}
void LingoDBTableCatalogEntry::ensureFullyLoaded() {
   impl->ensureLoaded();
}

std::vector<std::pair<std::string, std::vector<std::string>>> TableCatalogEntry::getIndices() const {
   assert(catalog);
   std::vector<std::pair<std::string, std::vector<std::string>>> res;
   for (auto idx : indices) {
      if (auto indexEntry = catalog->getTypedEntry<IndexCatalogEntry>(idx)) {
         res.push_back({idx, indexEntry.value()->getIndexedColumns()});
      }
   }
   return res;
}

LingoDBHashIndexEntry::LingoDBHashIndexEntry(std::string name, std::string tableName, std::vector<std::string> indexedColumns, std::unique_ptr<lingodb::runtime::LingoDBHashIndex> impl) : IndexCatalogEntry(CatalogEntryType::LINGODB_HASH_INDEX_ENTRY, name, tableName, indexedColumns), impl(std::move(impl)) {}

void LingoDBHashIndexEntry::serializeEntry(lingodb::utility::Serializer& serializer) const {
   serializer.writeProperty(2, name);
   serializer.writeProperty(3, tableName);
   serializer.writeProperty(4, indexedColumns);
   serializer.writeProperty(5, impl);
}
std::shared_ptr<LingoDBHashIndexEntry> LingoDBHashIndexEntry::deserialize(lingodb::utility::Deserializer& deserializer) {
   auto name = deserializer.readProperty<std::string>(2);
   auto tableName = deserializer.readProperty<std::string>(3);
   auto indexedColumns = deserializer.readProperty<std::vector<std::string>>(4);
   auto rawIndex = deserializer.readProperty<std::unique_ptr<lingodb::runtime::LingoDBHashIndex>>(5);
   return std::make_shared<LingoDBHashIndexEntry>(name, tableName, indexedColumns, std::move(rawIndex));
}

void LingoDBHashIndexEntry::setCatalog(Catalog* catalog) {
   CatalogEntry::setCatalog(catalog);
   auto tableEntry = catalog->getTypedEntry<LingoDBTableCatalogEntry>(tableName).value();
   impl->setTable(tableEntry.get());
}

void LingoDBHashIndexEntry::setDBDir(std::string dbDir) {
   impl->setDBDir(dbDir);
}
void LingoDBHashIndexEntry::setShouldPersist(bool shouldPersist) {
   impl->setPersist(shouldPersist);
}
lingodb::runtime::Index& LingoDBHashIndexEntry::getIndex() {
   return *impl;
}

std::shared_ptr<LingoDBHashIndexEntry> LingoDBHashIndexEntry::createForPrimaryKey(std::string table, std::vector<std::string> primaryKey) {
   auto impl = std::make_unique<runtime::LingoDBHashIndex>(table + ".pk.hashidx", primaryKey);
   auto res = std::make_shared<LingoDBHashIndexEntry>(table + ".pk", table, primaryKey, std::move(impl));
   return res;
}
void LingoDBHashIndexEntry::flush() {
   impl->flush();
}
void LingoDBHashIndexEntry::ensureFullyLoaded() {
   impl->ensureLoaded();
}

} // namespace lingodb::catalog