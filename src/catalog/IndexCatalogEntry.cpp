#include "lingodb/catalog/IndexCatalogEntry.h"

#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/runtime/LingoDBHashIndex.h"
#include "lingodb/utility/Serialization.h"

namespace lingodb::catalog {
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