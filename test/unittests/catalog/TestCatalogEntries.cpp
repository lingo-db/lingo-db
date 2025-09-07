#include "catch2/catch_all.hpp"
#include "lingodb/catalog/Defs.h"
#include "lingodb/catalog/IndexCatalogEntry.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/catalog/Types.h"
#include "lingodb/utility/Serialization.h"

#include <arrow/builder.h>
#include <arrow/ipc/reader.h>
#include <lingodb/catalog/Column.h>
#include <lingodb/catalog/MetaData.h>
using namespace lingodb::utility;
using namespace lingodb::catalog;

TEST_CASE("TableCatalogEntry:CreateTableDef") {
   CreateTableDef createTableDef;
   createTableDef.name = "test_table";
   createTableDef.columns = {Column("col1", Type::int8(), true), Column("col2", Type::stringType(), false)};
   createTableDef.primaryKey = {"col1"};

   SimpleByteWriter writer;
   Serializer serializer(writer);
   serializer.writeProperty(1, createTableDef);

   SimpleByteReader reader(writer.data(), writer.size());
   Deserializer deserializer(reader);
   auto createTableDef2 = deserializer.readProperty<CreateTableDef>(1);

   REQUIRE(createTableDef.name == createTableDef2.name);
   REQUIRE(createTableDef.columns.size() == createTableDef2.columns.size());
   REQUIRE(createTableDef.primaryKey == createTableDef2.primaryKey);
}

TEST_CASE("TableCatalogEntry:CreateAndSerialize") {
   CreateTableDef createTableDef;
   createTableDef.name = "test_table";
   createTableDef.columns = {Column("col1", Type::int8(), true), Column("col2", Type::stringType(), false)};
   createTableDef.primaryKey = {"col1"};
   auto tableEntry = LingoDBTableCatalogEntry::createFromCreateTable(createTableDef);
   REQUIRE(tableEntry->getName() == "test_table");
   REQUIRE(tableEntry->getColumns().size() == 2);
   REQUIRE(tableEntry->getColumns()[0].getColumnName() == "col1");
   REQUIRE(tableEntry->getColumns()[0].getLogicalType().toString() == "int8");
   REQUIRE(tableEntry->getColumns()[1].getColumnName() == "col2");
   REQUIRE(tableEntry->getColumns()[1].getLogicalType().toString() == "string");
   REQUIRE(tableEntry->getPrimaryKey().size() == 1);
   REQUIRE(tableEntry->getPrimaryKey()[0] == "col1");
   SimpleByteWriter writer;
   Serializer serializer(writer);
   serializer.writeProperty(1, std::dynamic_pointer_cast<CatalogEntry>(tableEntry));
   SimpleByteReader reader(writer.data(), writer.size());
   Deserializer deserializer(reader);
   auto catalogEntry = deserializer.readProperty<std::shared_ptr<CatalogEntry>>(1);
   REQUIRE(catalogEntry->getEntryType() == CatalogEntry::CatalogEntryType::LINGODB_TABLE_ENTRY);
   auto tableEntry2 = std::dynamic_pointer_cast<LingoDBTableCatalogEntry>(catalogEntry);
   REQUIRE(tableEntry2 != nullptr);
   REQUIRE(tableEntry2->getName() == "test_table");
   REQUIRE(tableEntry2->getColumns().size() == 2);
   REQUIRE(tableEntry2->getColumns()[0].getColumnName() == "col1");
   REQUIRE(tableEntry2->getColumns()[0].getLogicalType().toString() == "int8");
   REQUIRE(tableEntry2->getColumns()[1].getColumnName() == "col2");
   REQUIRE(tableEntry2->getColumns()[1].getLogicalType().toString() == "string");
   REQUIRE(tableEntry2->getPrimaryKey().size() == 1);
   REQUIRE(tableEntry2->getPrimaryKey()[0] == "col1");
}

TEST_CASE("IndexCatalogEntry:CreateAndSerialize") {
   auto indexEntry = LingoDBHashIndexEntry::createForPrimaryKey("test_table", {"col1"});
   REQUIRE(indexEntry->getName() == "test_table.pk");
   REQUIRE(indexEntry->getTableName() == "test_table");
   REQUIRE(indexEntry->getIndexedColumns().size() == 1);
   REQUIRE(indexEntry->getIndexedColumns()[0] == "col1");
   SimpleByteWriter writer;
   Serializer serializer(writer);
   serializer.writeProperty(1, std::dynamic_pointer_cast<CatalogEntry>(indexEntry));
   SimpleByteReader reader(writer.data(), writer.size());
   Deserializer deserializer(reader);
   auto catalogEntry = deserializer.readProperty<std::shared_ptr<CatalogEntry>>(1);
   REQUIRE(catalogEntry->getEntryType() == CatalogEntry::CatalogEntryType::LINGODB_HASH_INDEX_ENTRY);
   auto indexEntry2 = std::dynamic_pointer_cast<LingoDBHashIndexEntry>(catalogEntry);
   REQUIRE(indexEntry2 != nullptr);
   REQUIRE(indexEntry2->getName() == "test_table.pk");
   REQUIRE(indexEntry2->getTableName() == "test_table");
   REQUIRE(indexEntry2->getIndexedColumns().size() == 1);
   REQUIRE(indexEntry2->getIndexedColumns()[0] == "col1");
}