#include "catch2/catch_all.hpp"

#include "lingodb/catalog/Catalog.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/runtime/storage/LingoDBTable.h"
#include "lingodb/utility/Serialization.h"
using namespace lingodb::utility;
using namespace lingodb::catalog;
TEST_CASE("Catalog: serialize") {
   SimpleByteWriter writer;
   Serializer serializer(writer);
   auto catalog = std::make_unique<Catalog>();
   catalog->insertEntry(std::make_unique<LingoDBTableCatalogEntry>("foo", std::vector<Column>{Column{"x", Type::int8(), false}}, std::vector<std::string>{"x"}, nullptr)); //todo
   serializer.writeProperty(1, catalog);
   SimpleByteReader reader(writer.data(), writer.size());
   Deserializer deserializer(reader);
   auto catalog2 = deserializer.readProperty<std::shared_ptr<Catalog>>(1);
   auto entry = catalog->getEntry("foo");
   REQUIRE(entry);
   REQUIRE(entry.value()->getEntryType() == lingodb::catalog::CatalogEntry::CatalogEntryType::LINGODB_TABLE_ENTRY);
   auto tableEntry = entry.value();
   REQUIRE(tableEntry->getName() == "foo");
}