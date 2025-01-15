#include "lingodb/catalog/Catalog.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/catalog/Types.h"
#include "lingodb/utility/Serialization.h" <

namespace lingodb::catalog {

std::unique_ptr<Catalog> Catalog::deserialize(lingodb::utility::Deserializer& deSerializer) {
   auto res = std::make_unique<Catalog>();
   res->entries = deSerializer.readProperty<std::unordered_map<std::string, std::unique_ptr<CatalogEntry>>>(1);
   return res;
}
void Catalog::serialize(lingodb::utility::Serializer& serializer) const {
   serializer.writeProperty(1, entries);
}

void CatalogEntry::serialize(lingodb::utility::Serializer& serializer) const {
   serializer.writeProperty(1, entryType);
   serializeEntry(serializer);
}
std::unique_ptr<CatalogEntry> CatalogEntry::deserialize(lingodb::utility::Deserializer& deserializer) {
   auto entryType = deserializer.readProperty<CatalogEntryType>(1);
   switch (entryType) {
      case CatalogEntryType::INVALID_ENTRY:
         return nullptr;
      case CatalogEntryType::TABLE_ENTRY:
         return TableCatalogEntry::deserialize(deserializer);
   }
}
void TableCatalogEntry::serializeEntry(lingodb::utility::Serializer& serializer) const {
   serializer.writeProperty(2, name);
   serializer.writeProperty(3, columns);
   serializer.writeProperty(4, primaryKey);
}
std::unique_ptr<TableCatalogEntry> TableCatalogEntry::deserialize(lingodb::utility::Deserializer& deserializer) {
   auto name = deserializer.readProperty<std::string>(2);
   auto columns = deserializer.readProperty<std::vector<TableColumn>>(3);
   auto primaryKey = deserializer.readProperty<std::vector<std::string>>(4);
   return std::make_unique<TableCatalogEntry>(name, columns, primaryKey);
}

void TableColumn::serialize(lingodb::utility::Serializer& serializer) const {
   serializer.writeProperty(1, columnName);
   serializer.writeProperty(2, type);
   serializer.writeProperty(3, isNullable);
}
TableColumn TableColumn::deserialize(lingodb::utility::Deserializer& deserializer) {
   auto columnName = deserializer.readProperty<std::string>(1);
   auto type = deserializer.readProperty<Type>(2);
   auto isNullable = deserializer.readProperty<bool>(3);
   return TableColumn(columnName, type, isNullable);
}
} // namespace lingodb::catalog