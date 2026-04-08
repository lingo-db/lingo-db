#include "catch2/catch_all.hpp"
#include "lingodb/catalog/Column.h"
#include "lingodb/catalog/Defs.h"
#include "lingodb/catalog/IndexCatalogEntry.h"
#include "lingodb/catalog/MetaData.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/catalog/Types.h"
#include "lingodb/runtime/ExternalDataSourceProperty.h"
#include "lingodb/runtime/LingoDBHashIndex.h"
#include "lingodb/runtime/RelationHelper.h"
#include "lingodb/runtime/Session.h"
#include "lingodb/runtime/storage/Index.h"
#include "lingodb/runtime/storage/TableStorage.h"
#include "lingodb/execution/Execution.h"
#include "lingodb/scheduler/Tasks.h"
#include "lingodb/utility/Serialization.h"

#include <filesystem>

#include <arrow/builder.h>
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

   // Create int8 array [1, 2]
   arrow::Int8Builder intBuilder;
   REQUIRE(intBuilder.Append(1).ok());
   REQUIRE(intBuilder.Append(2).ok());
   auto x = intBuilder.Finish().ValueOrDie();

   // Create string array ["a", "b"]
   arrow::StringBuilder stringBuilder;
   REQUIRE(stringBuilder.Append("a").ok());
   REQUIRE(stringBuilder.Append("b").ok());
   auto y = stringBuilder.Finish().ValueOrDie();

   return arrow::RecordBatch::Make(schema, 2, std::vector<std::shared_ptr<arrow::Array>>{x, y});
}
auto createTableDataAsTable() {
   return arrow::Table::FromRecordBatches({createTableData()}).ValueOrDie();
}

auto createMultiColumnTableEntry() {
   CreateTableDef createTableDef;
   createTableDef.name = "test_table_multi";
   createTableDef.columns = {
      Column("c10", Type::charType(10), true),
      Column("i", Type::int32(), true),
      Column("u", Type::makeIntType(32, false), true),
      Column("s", Type::stringType(), true),
      Column("t", Type(LogicalTypeId::TIMESTAMP, std::make_shared<TimestampTypeInfo>(std::nullopt, TimestampTypeInfo::TimestampUnit::MILLIS)), true),
      Column("b", Type::boolean(), true),
      Column("c1", Type::charType(1), true),
      Column("im", Type::intervalMonths(), true),
      Column("idt", Type::intervalDaytime(), true),
      Column("d18", Type::decimal(18, 2), true),
      Column("d32", Type::decimal(32, 4), true),
      Column("dt", Type(LogicalTypeId::DATE, std::make_shared<DateTypeInfo>(DateTypeInfo::DateUnit::DAY)), true),
      Column("dtms", Type(LogicalTypeId::DATE, std::make_shared<DateTypeInfo>(DateTypeInfo::DateUnit::MILLIS)), true),
   };
   createTableDef.primaryKey = {"c10", "i", "u", "s", "t", "b", "c1", "im", "idt", "d18", "d32", "dt", "dtms"};
   return LingoDBTableCatalogEntry::createFromCreateTable(createTableDef);
}
auto createMultiColumnIndexEntry() {
   return LingoDBHashIndexEntry::createForPrimaryKey("test_table_multi", {"c10", "i", "u", "s", "t", "b", "c1", "im", "idt", "d18", "d32", "dt", "dtms"});
}
auto createMultiColumnTableData() {
   auto schema = arrow::schema({
      arrow::field("c10", arrow::utf8()),
      arrow::field("i", arrow::int32()),
      arrow::field("u", arrow::int32()),
      arrow::field("s", arrow::utf8()),
      arrow::field("t", arrow::timestamp(arrow::TimeUnit::MILLI)),
      arrow::field("b", arrow::boolean()),
      arrow::field("c1", arrow::fixed_size_binary(4)),
      arrow::field("im", arrow::month_interval()),
      arrow::field("idt", arrow::day_time_interval()),
      arrow::field("d18", arrow::decimal128(18, 2)),
      arrow::field("d32", arrow::decimal128(32, 4)),
      arrow::field("dt", arrow::date32()),
      arrow::field("dtms", arrow::date64()),
   });

   arrow::StringBuilder c10Builder;
   REQUIRE(c10Builder.Append("0123456789").ok());
   REQUIRE(c10Builder.Append("abcdefghij").ok());
   REQUIRE(c10Builder.AppendNull().ok());
   auto c10Arr = c10Builder.Finish().ValueOrDie();

   arrow::Int32Builder iBuilder;
   REQUIRE(iBuilder.Append(42).ok());
   REQUIRE(iBuilder.AppendNull().ok());
   REQUIRE(iBuilder.Append(-7).ok());
   auto iArr = iBuilder.Finish().ValueOrDie();

   arrow::Int32Builder uBuilder;
   REQUIRE(uBuilder.Append(7).ok());
   REQUIRE(uBuilder.Append(-2).ok());
   REQUIRE(uBuilder.AppendNull().ok());
   auto uArr = uBuilder.Finish().ValueOrDie();

   arrow::StringBuilder sBuilder;
   REQUIRE(sBuilder.Append("alpha").ok());
   REQUIRE(sBuilder.Append("betaggamaetanetalambda").ok());
   REQUIRE(sBuilder.AppendNull().ok());
   auto sArr = sBuilder.Finish().ValueOrDie();

   arrow::TimestampBuilder tBuilder(arrow::timestamp(arrow::TimeUnit::MILLI), arrow::default_memory_pool());
   REQUIRE(tBuilder.Append(static_cast<int64_t>(1710000000000000ll)).ok());
   REQUIRE(tBuilder.AppendNull().ok());
   REQUIRE(tBuilder.Append(static_cast<int64_t>(1710000000000123ll)).ok());
   auto tArr = tBuilder.Finish().ValueOrDie();

   arrow::BooleanBuilder bBuilder;
   REQUIRE(bBuilder.Append(true).ok());
   REQUIRE(bBuilder.AppendNull().ok());
   REQUIRE(bBuilder.Append(false).ok());
   auto bArr = bBuilder.Finish().ValueOrDie();

   arrow::FixedSizeBinaryBuilder c1Builder(arrow::fixed_size_binary(4), arrow::default_memory_pool());
   const uint8_t c1r0[] = {0, 0, 0, static_cast<uint8_t>('x')};
   const uint8_t c1r1[] = {155, 156, 157, 159};
   const uint8_t c1r2[] = {0, 0, 0, static_cast<uint8_t>('z')};
   REQUIRE(c1Builder.Append(c1r0).ok());
   REQUIRE(c1Builder.Append(c1r1).ok());
   REQUIRE(c1Builder.Append(c1r2).ok());
   auto c1Arr = c1Builder.Finish().ValueOrDie();

   arrow::MonthIntervalBuilder imBuilder(arrow::default_memory_pool());
   REQUIRE(imBuilder.Append(12).ok());
   REQUIRE(imBuilder.AppendNull().ok());
   REQUIRE(imBuilder.Append(-3).ok());
   auto imArr = imBuilder.Finish().ValueOrDie();

   arrow::DayTimeIntervalBuilder idtBuilder(arrow::default_memory_pool());
   REQUIRE(idtBuilder.Append(arrow::DayTimeIntervalType::DayMilliseconds{1, 250}).ok());
   REQUIRE(idtBuilder.AppendNull().ok());
   REQUIRE(idtBuilder.Append(arrow::DayTimeIntervalType::DayMilliseconds{-7, 0}).ok());
   auto idtArr = idtBuilder.Finish().ValueOrDie();

   arrow::Decimal128Builder d18Builder(arrow::decimal128(18, 2), arrow::default_memory_pool());
   REQUIRE(d18Builder.Append(arrow::Decimal128::FromString("1234567890123456.78").ValueOrDie()).ok());
   REQUIRE(d18Builder.AppendNull().ok());
   REQUIRE(d18Builder.Append(arrow::Decimal128::FromString("-1.00").ValueOrDie()).ok());
   auto d18Arr = d18Builder.Finish().ValueOrDie();

   arrow::Decimal128Builder d32Builder(arrow::decimal128(32, 4), arrow::default_memory_pool());
   REQUIRE(d32Builder.Append(arrow::Decimal128::FromString("1234567890123456789012345678.1234").ValueOrDie()).ok());
   REQUIRE(d32Builder.Append(arrow::Decimal128::FromString("-9999999999999999999999999999.0001").ValueOrDie()).ok());
   REQUIRE(d32Builder.AppendNull().ok());
   auto d32Arr = d32Builder.Finish().ValueOrDie();

   arrow::Date32Builder dtBuilder(arrow::default_memory_pool());
   REQUIRE(dtBuilder.Append(20000).ok());
   REQUIRE(dtBuilder.AppendNull().ok());
   REQUIRE(dtBuilder.Append(1).ok());
   auto dtArr = dtBuilder.Finish().ValueOrDie();

   arrow::Date64Builder dtmsBuilder(arrow::default_memory_pool());
   REQUIRE(dtmsBuilder.Append(1710000001000ll).ok());
   REQUIRE(dtmsBuilder.AppendNull().ok());
   REQUIRE(dtmsBuilder.Append(1).ok());
   auto dtmsArr = dtmsBuilder.Finish().ValueOrDie();

   return arrow::RecordBatch::Make(schema, 3, std::vector<std::shared_ptr<arrow::Array>>{c10Arr, iArr, uArr, sArr, tArr, bArr, c1Arr, imArr, idtArr, d18Arr, d32Arr, dtArr, dtmsArr});
}
auto createMultiColumnTableDataAsTable() {
   return arrow::Table::FromRecordBatches({createMultiColumnTableData()}).ValueOrDie();
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

TEST_CASE("Storage:HashIndexMultiColumnWithNulls") {
   auto scheduler = lingodb::scheduler::startScheduler();

   fs::path tempDir = fs::temp_directory_path() / "lingodb-test-dir-multi";
   if (fs::exists(tempDir)) {
      fs::remove_all(tempDir);
   }
   fs::create_directories(tempDir);

   auto catalog = Catalog::create(tempDir.string(), true);
   catalog->setShouldPersist(true);
   auto tableEntry = createMultiColumnTableEntry();
   catalog->insertEntry(tableEntry);
   auto indexEntry = createMultiColumnIndexEntry();
   catalog->insertEntry(indexEntry);
   tableEntry->addIndex(indexEntry->getName());
   catalog->persist();

   lingodb::scheduler::awaitEntryTask(std::make_unique<MockTask>([&]() {
      auto catalog2 = Catalog::create(tempDir.string(), true);
      catalog2->setShouldPersist(true);
      auto tableData = createMultiColumnTableData();
      if (auto relation = catalog2->getTypedEntry<TableCatalogEntry>("test_table_multi")) {
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
      auto catalog3 = Catalog::create(tempDir.string(), true);
      auto indexEntry3 = catalog3->getTypedEntry<IndexCatalogEntry>("test_table_multi.pk");
      REQUIRE(indexEntry3 != std::nullopt);
      auto& index3 = indexEntry3.value()->getIndex();
      auto* hashIndex3 = dynamic_cast<lingodb::runtime::LingoDBHashIndex*>(&index3);
      REQUIRE(hashIndex3 != nullptr);
      lingodb::runtime::HashIndexAccess access(*hashIndex3, {"c10", "i", "u", "s", "t", "b", "c1", "im", "idt", "d18", "d32", "dt", "dtms"});

      auto tmpSession = lingodb::runtime::Session::createSession();
      auto createTableDef = lingodb::catalog::CreateTableDef{"tmp", tableEntry->getColumns(), {}};
      tmpSession->getCatalog()->insertEntry(lingodb::catalog::LingoDBTableCatalogEntry::createFromCreateTable(createTableDef));
      tmpSession->getCatalog()->getTypedEntry<lingodb::catalog::LingoDBTableCatalogEntry>("tmp").value()->getTableStorage().append(createMultiColumnTableDataAsTable());

      std::string query = "select row_number() over() -1 as rowid, hash(c10,i,u,s,t,b,c1,im,idt,d18,d32,dt,dtms) as hash from tmp";
      auto queryExecutionConfig = lingodb::execution::createQueryExecutionConfig(lingodb::execution::ExecutionMode::SPEED, true);
      queryExecutionConfig->parallel = false;
      std::shared_ptr<arrow::Table> result;
      queryExecutionConfig->resultProcessor = lingodb::execution::createTableRetriever(result);
      auto executer = lingodb::execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig), *tmpSession);
      executer->fromData(query);
      lingodb::scheduler::awaitChildTask(std::make_unique<lingodb::execution::QueryExecutionTask>(std::move(executer)));

      auto asBatch = result->CombineChunksToBatch().ValueOrDie();
      auto hashColumn = std::static_pointer_cast<arrow::Int64Array>(asBatch->GetColumnByName("hash"));
      REQUIRE(hashColumn != nullptr);

      for (int64_t r = 0; r < asBatch->num_rows(); r++) {
         auto h = static_cast<size_t>(hashColumn->Value(r));
         auto* iter = access.lookup(h);
         REQUIRE(iter->hasNext());
      }
   }));
}
TEST_CASE("Storage:RelationHelper") {
   auto scheduler = lingodb::scheduler::startScheduler();
   REQUIRE(lingodb::scheduler::getNumWorkers() > 1);

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
         lingodb::runtime::ExternalDatasourceProperty externalDatasourceProperty{.tableName = "test_table", .mapping = {{"x", "col1"}, {"y", "col2"}}, .index = "test_table.pk", .indexType = "hash"};
         auto* access3 = lingodb::runtime::RelationHelper::accessHashIndex(lingodb::runtime::VarLen32::fromString(lingodb::utility::serializeToHexString(externalDatasourceProperty)));
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
