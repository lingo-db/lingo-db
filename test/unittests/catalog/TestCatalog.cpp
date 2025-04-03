#include "catch2/catch_all.hpp"
#include "lingodb/catalog/Column.h"
#include "lingodb/catalog/Defs.h"
#include "lingodb/catalog/IndexCatalogEntry.h"
#include "lingodb/catalog/MetaData.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/catalog/Types.h"
#include "lingodb/utility/Serialization.h"

#include <filesystem>

#include <arrow/ipc/json_simple.h>
#include <arrow/ipc/reader.h>
using namespace lingodb::utility;
using namespace lingodb::catalog;
namespace fs = std::filesystem;
namespace {
auto createTableEntry() {
   CreateTableDef createTableDef;
   createTableDef.name = "test_table";
   createTableDef.columns = {Column("col1", Type::int8(), true), Column("col2", Type::stringType(), false)};
   createTableDef.primaryKey = {"col1"};
   return LingoDBTableCatalogEntry::createFromCreateTable(createTableDef);
}
auto createIndexEntry() {
   return LingoDBHashIndexEntry::createForPrimaryKey("test_table", {"col1"});
}
} // namespace

TEST_CASE("Catalog:InMemory") {
   auto catalog = Catalog::createEmpty();
   auto tableEntry = createTableEntry();
   catalog->insertEntry(tableEntry);
   auto indexEntry = createIndexEntry();
   catalog->insertEntry(indexEntry);
   if (auto entry = catalog->getTypedEntry<TableCatalogEntry>("test_table")) {
      REQUIRE(entry.value()->getName() == "test_table");
      REQUIRE(entry.value()->getColumns().size() == 2);
   } else {
      FAIL("Table entry not found");
   }
   if (auto entry = catalog->getTypedEntry<IndexCatalogEntry>("test_table.pk")) {
      REQUIRE(entry.value()->getName() == "test_table.pk");
      REQUIRE(entry.value()->getTableName() == "test_table");
      REQUIRE(entry.value()->getIndexedColumns().size() == 1);
   } else {
      FAIL("Index entry not found");
   }
   REQUIRE(catalog->getEntry("non_existent_entry") == std::nullopt);
   REQUIRE(catalog->getTypedEntry<IndexCatalogEntry>("test_table") == std::nullopt);
   REQUIRE(catalog->getTypedEntry<TableCatalogEntry>("test_table.pk") == std::nullopt);
}

TEST_CASE("Catalog:Persisted") {
   fs::path tempDir = fs::temp_directory_path() / "lingodb-test-dir";
   //if exists: delete
   if (fs::exists(tempDir)) {
      fs::remove_all(tempDir);
   }
   fs::create_directories(tempDir);

   auto catalog = Catalog::create(tempDir.string(), true);
   catalog->setShouldPersist(true);
   auto tableEntry = createTableEntry();
   catalog->insertEntry(tableEntry);
   auto indexEntry = createIndexEntry();
   catalog->insertEntry(indexEntry);
   catalog->persist();
   //load again
   auto catalog2 = Catalog::create(tempDir.string(), false);
   if (auto entry = catalog2->getTypedEntry<TableCatalogEntry>("test_table")) {
      REQUIRE(entry.value()->getName() == "test_table");
      REQUIRE(entry.value()->getColumns().size() == 2);
   } else {
      FAIL("Table entry not found");
   }
   if (auto entry = catalog2->getTypedEntry<IndexCatalogEntry>("test_table.pk")) {
      REQUIRE(entry.value()->getName() == "test_table.pk");
      REQUIRE(entry.value()->getTableName() == "test_table");
      REQUIRE(entry.value()->getIndexedColumns().size() == 1);
   } else {
      FAIL("Index entry not found");
   }
}