#include "lingodb/catalog/TableCatalogEntry.h"

#include "lingodb/catalog/Defs.h"
#include "lingodb/catalog/IndexCatalogEntry.h"
#include "lingodb/runtime/storage/LingoDBTable.h"
#include "lingodb/utility/Serialization.h"

#include <cassert>

namespace lingodb::catalog {
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
} // namespace lingodb::catalog
