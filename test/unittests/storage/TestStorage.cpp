#include "catch2/catch_all.hpp"
#include "lingodb/catalog/Column.h"
#include "lingodb/catalog/Defs.h"
#include "lingodb/catalog/IndexCatalogEntry.h"
#include "lingodb/catalog/MetaData.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/catalog/Types.h"
#include "lingodb/runtime/LingoDBHashIndex.h"
#include "lingodb/runtime/RelationHelper.h"
#include "lingodb/runtime/Session.h"
#include "lingodb/runtime/storage/Index.h"
#include "lingodb/runtime/storage/TableStorage.h"
#include "lingodb/scheduler/Tasks.h"
#include "lingodb/utility/Serialization.h"

#include <filesystem>

#include <arrow/ipc/json_simple.h>
#include <arrow/ipc/reader.h>
#include <arrow/table.h>

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
auto createTableData() {
   auto schema = arrow::schema({arrow::field("col1", arrow::int8()), arrow::field("col2", arrow::utf8())});
   auto x = arrow::ipc::internal::json::ArrayFromJSON(arrow::int8(), "[1, 2]").ValueOrDie();
   auto y = arrow::ipc::internal::json::ArrayFromJSON(arrow::utf8(), R"(["a", "b"])").ValueOrDie();
   return arrow::RecordBatch::Make(schema, 2, std::vector<std::shared_ptr<arrow::Array>>{x, y});
}
auto createTableDataAsTable() {
   return arrow::Table::FromRecordBatches({createTableData()}).ValueOrDie();
}

class MockTask : public lingodb::scheduler::Task {
   std::function<void()> job;

   public:
   MockTask(std::function<void()> job) : job(std::move(job)) {}
   bool allocateWork() override {
      if (workExhausted.exchange(true)) {
         return false;
      }
      return true;
   }
   void performWork() override {
      job();
   }
};
class MockTaskWithContext : public lingodb::scheduler::TaskWithContext {
   std::function<void()> job;

   public:
   MockTaskWithContext(lingodb::runtime::ExecutionContext* context, std::function<void()> job) : TaskWithContext(context), job(std::move(job)) {}
   bool allocateWork() override {
      if (workExhausted.exchange(true)) {
         return false;
      }
      return true;
   }
   void performWork() override {
      job();
   }
};

} // namespace

TEST_CASE("Storage") {
   auto scheduler = lingodb::scheduler::startScheduler();

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
   tableEntry->addIndex(indexEntry->getName());
   catalog->persist();
   //load again
   auto catalog2 = Catalog::create(tempDir.string(), true);
   catalog2->setShouldPersist(true);
   lingodb::scheduler::awaitEntryTask(std::make_unique<MockTask>([&]() {
      auto indexEntry2 = catalog2->getTypedEntry<IndexCatalogEntry>("test_table.pk");
      REQUIRE(indexEntry2 != std::nullopt);
      auto& index2 = indexEntry2.value()->getIndex();
      auto* hashIndex2 = dynamic_cast<lingodb::runtime::LingoDBHashIndex*>(&index2);
      REQUIRE(hashIndex2 != nullptr);
      lingodb::runtime::HashIndexAccess access2(*hashIndex2, {"col1"});
      auto* iter2 = access2.lookup(0);
      REQUIRE(!iter2->hasNext());
   }));

   lingodb::scheduler::awaitEntryTask(std::make_unique<MockTask>([&]() {
      auto tableData = createTableData();
      if (auto relation = catalog2->getTypedEntry<TableCatalogEntry>("test_table")) {
         auto startRowId = relation.value()->getTableStorage().nextRowId();
         relation.value()->getTableStorage().append({tableData});
         for (auto idx : relation.value()->getIndices()) {
            if (auto index = catalog2->getTypedEntry<IndexCatalogEntry>(idx.first)) {
               index.value()->getIndex().appendRows(startRowId, tableData);
            }
         }
      }
      catalog2->persist();
   }));
   lingodb::scheduler::awaitEntryTask(std::make_unique<MockTask>([&]() {
      //load again
      auto catalog3 = Catalog::create(tempDir.string(), true);
      auto indexEntry3 = catalog3->getTypedEntry<IndexCatalogEntry>("test_table.pk");
      REQUIRE(indexEntry3 != std::nullopt);
      auto& index3 = indexEntry3.value()->getIndex();
      auto* hashIndex3 = dynamic_cast<lingodb::runtime::LingoDBHashIndex*>(&index3);
      REQUIRE(hashIndex3 != nullptr);
      lingodb::runtime::HashIndexAccess access3(*hashIndex3, {"col1", "col2"});
      auto* iter3 = access3.lookup(-3797884931935089717);
      REQUIRE(iter3->hasNext());
      lingodb::runtime::BatchView batchView;
      iter3->consumeRecordBatch(&batchView);
      REQUIRE(batchView.length == 1);
      REQUIRE(!iter3->hasNext());
      REQUIRE(batchView.arrays[0]->offset == 0);
      REQUIRE(batchView.arrays[1]->offset == 0);
      REQUIRE(*reinterpret_cast<const int8_t*>(batchView.arrays[0]->buffers[1]) == 1);
      auto* strOffsets = reinterpret_cast<const int32_t*>(batchView.arrays[1]->buffers[1]);
      auto strLen = strOffsets[1] - strOffsets[0];
      auto* strData = reinterpret_cast<const char*>(batchView.arrays[1]->buffers[2]);
      auto str = std::string(&strData[strOffsets[0]], strLen);
      REQUIRE(str == "a");
   }));
}
TEST_CASE("Storage:RelationHelper") {
   auto scheduler = lingodb::scheduler::startScheduler();

   fs::path tempDir = fs::temp_directory_path() / "lingodb-test-dir";
   //if exists: delete
   if (fs::exists(tempDir)) {
      fs::remove_all(tempDir);
   }
   fs::create_directories(tempDir);
   {
      auto session = lingodb::runtime::Session::createSession(tempDir, true);
      auto context = session->createExecutionContext();
      CreateTableDef createTableDef;
      createTableDef.name = "test_table";
      createTableDef.columns = {Column("col1", Type::int8(), true), Column("col2", Type::stringType(), false)};
      createTableDef.primaryKey = {"col1"};
      lingodb::scheduler::awaitEntryTask(std::make_unique<MockTaskWithContext>(context.get(), [&]() {
         lingodb::runtime::RelationHelper::setPersist(true);
         lingodb::runtime::RelationHelper::createTable(lingodb::runtime::VarLen32::fromString(serializeToHexString(createTableDef)));
         lingodb::runtime::RelationHelper::appendToTable(*session, "test_table", createTableDataAsTable());
      }));
   }

   lingodb::scheduler::awaitEntryTask(std::make_unique<MockTask>([&]() {
      auto session = lingodb::runtime::Session::createSession(tempDir, true);
      auto context = session->createExecutionContext();

      auto catalog = session->getCatalog();
      auto tableEntry = catalog->getTypedEntry<TableCatalogEntry>("test_table");
      REQUIRE(tableEntry != std::nullopt);
      auto indexEntry = catalog->getTypedEntry<IndexCatalogEntry>("test_table.pk");
      REQUIRE(indexEntry != std::nullopt);
      lingodb::scheduler::awaitEntryTask(std::make_unique<MockTaskWithContext>(context.get(), [&]() {
         lingodb::runtime::RelationHelper::setPersist(true);
         auto* access3 = lingodb::runtime::RelationHelper::accessHashIndex(lingodb::runtime::VarLen32::fromString(R"({"type": "hash", "index": "test_table.pk", "relation": "test_table", "mapping": {"x":"col1", "y":"col2"} })"));
         auto* iter3 = access3->lookup(-3797884931935089717);
         REQUIRE(iter3->hasNext());
         lingodb::runtime::BatchView batchView;
         iter3->consumeRecordBatch(&batchView);
         REQUIRE(batchView.length == 1);
         REQUIRE(!iter3->hasNext());
         REQUIRE(batchView.arrays[0]->offset == 0);
         REQUIRE(batchView.arrays[1]->offset == 0);
         REQUIRE(*reinterpret_cast<const int8_t*>(batchView.arrays[0]->buffers[1]) == 1);
         auto* strOffsets = reinterpret_cast<const int32_t*>(batchView.arrays[1]->buffers[1]);
         auto strLen = strOffsets[1] - strOffsets[0];
         auto* strData = reinterpret_cast<const char*>(batchView.arrays[1]->buffers[2]);
         auto str = std::string(&strData[strOffsets[0]], strLen);
         REQUIRE(str == "a");
      }));
   }));
}
