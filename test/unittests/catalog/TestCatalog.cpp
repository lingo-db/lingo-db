#include "catch2/catch_all.hpp"

#include "lingodb/catalog/Catalog.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/utility/Serialization.h"
using namespace lingodb::utility;
using namespace lingodb::catalog;
TEST_CASE("Catalog: serialize") {
   SimpleByteWriter writer;
   Serializer serializer(writer);
   auto catalog = std::make_unique<Catalog>();
   catalog->insertEntry(std::make_unique<TableCatalogEntry>("foo", std::vector<TableColumn>{TableColumn{"x", Type::int8()}}, std::vector<std::string>{"x"}));
   serializer.writeProperty(1, catalog);
   SimpleByteReader reader(writer.data(), writer.size());
   Deserializer deserializer(reader);
   auto catalog2 = deserializer.readProperty<std::unique_ptr<Catalog>>(1);
   auto entry = catalog->getEntry("foo");
   REQUIRE(entry);
   REQUIRE(entry.value()->getEntryType() == lingodb::catalog::CatalogEntry::CatalogEntryType::TABLE_ENTRY);
   auto tableEntry = reinterpret_cast<TableCatalogEntry*>(entry.value());
   REQUIRE(tableEntry->getName() == "foo");

}