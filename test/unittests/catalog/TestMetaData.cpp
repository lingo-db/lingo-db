#include "catch2/catch_all.hpp"
#include "lingodb/catalog/Types.h"
#include "lingodb/utility/Serialization.h"

#include <arrow/builder.h>
#include <arrow/ipc/reader.h>
#include <lingodb/catalog/Column.h>
#include <lingodb/catalog/MetaData.h>
using namespace lingodb::utility;
using namespace lingodb::catalog;

TEST_CASE("MetaData:Column") {
   Column column("test", Type::int8(), true);
   Column column2("test2", Type::int64(), false);
   SimpleByteWriter writer;
   Serializer serializer(writer);
   serializer.writeProperty(1, column);
   serializer.writeProperty(2, column2);
   SimpleByteReader reader(writer.data(), writer.size());
   Deserializer deserializer(reader);
   auto column3 = deserializer.readProperty<Column>(1);
   auto column4 = deserializer.readProperty<Column>(2);
   REQUIRE(column.getColumnName() == column3.getColumnName());
   REQUIRE(column.getLogicalType().toString() == column3.getLogicalType().toString());
   REQUIRE(column.getIsNullable() == column3.getIsNullable());
   REQUIRE(column2.getColumnName() == column4.getColumnName());
   REQUIRE(column2.getLogicalType().toString() == column4.getLogicalType().toString());
   REQUIRE(column2.getIsNullable() == column4.getIsNullable());
}

TEST_CASE("MetaData:ColumnStatistics") {
   ColumnStatistics stats(10);
   ColumnStatistics stats2(std::nullopt);
   SimpleByteWriter writer;
   Serializer serializer(writer);
   serializer.writeProperty(1, stats);
   serializer.writeProperty(2, stats2);
   SimpleByteReader reader(writer.data(), writer.size());
   Deserializer deserializer(reader);
   auto stats3 = deserializer.readProperty<ColumnStatistics>(1);
   auto stats4 = deserializer.readProperty<ColumnStatistics>(2);
   REQUIRE(stats3.getNumDistinctValues().value() == 10);
   REQUIRE(!stats4.getNumDistinctValues());
}

TEST_CASE("MetaData:Sample") {
   // First create a sample (arrow record batch) with 2 rows and 2 columns
   auto schema = arrow::schema({arrow::field("a", arrow::int32()), arrow::field("b", arrow::float64())});

   // Create int32 array [1, 2]
   arrow::Int32Builder intBuilder;
   REQUIRE(intBuilder.Append(1).ok());
   REQUIRE(intBuilder.Append(2).ok());
   auto x = intBuilder.Finish().ValueOrDie();

   // Create float64 array [3.0, 4.0]
   arrow::DoubleBuilder doubleBuilder;
   REQUIRE(doubleBuilder.Append(3.0).ok());
   REQUIRE(doubleBuilder.Append(4.0).ok());
   auto y = doubleBuilder.Finish().ValueOrDie();
   auto batch = arrow::RecordBatch::Make(schema, 2, std::vector<std::shared_ptr<arrow::Array>>{x, y});
   Sample sample(batch);
   SimpleByteWriter writer;
   Serializer serializer(writer);
   serializer.writeProperty(1, sample);
   SimpleByteReader reader(writer.data(), writer.size());
   Deserializer deserializer(reader);
   auto sample2 = deserializer.readProperty<Sample>(1);
   REQUIRE(sample2.getSampleData()->num_rows() == 2);
}

class MockTableMetaDataProvider : public TableMetaDataProvider {
   Sample sample;
   size_t numRows;
   std::vector<std::string> primaryKey;
   std::vector<std::string> columnNames;
   std::vector<std::unique_ptr<ColumnStatistics>> columnStatistics;
   std::vector<std::pair<std::string, std::vector<std::string>>> indices;

   public:
   MockTableMetaDataProvider(Sample sample, size_t numRows, std::vector<std::string> primaryKey, std::vector<std::string> columnNames, std::vector<std::unique_ptr<ColumnStatistics>> columnStatistics, std::vector<std::pair<std::string, std::vector<std::string>>> indices)
      : sample(std::move(sample)), numRows(numRows), primaryKey(std::move(primaryKey)), columnNames(std::move(columnNames)), columnStatistics(std::move(columnStatistics)), indices(std::move(indices)) {}

   const Sample& getSample() const override { return sample; }
   size_t getNumRows() const override { return numRows; }
   std::vector<std::string> getPrimaryKey() const override { return primaryKey; }
   std::vector<std::string> getColumnNames() const override { return columnNames; }
   const ColumnStatistics& getColumnStatistics(std::string_view column) const override {
      for (size_t i = 0; i < columnNames.size(); i++) {
         if (columnNames[i] == column) {
            return *columnStatistics[i];
         }
      }
      throw std::runtime_error("MetaData: Column not found");
   }
   std::vector<std::pair<std::string, std::vector<std::string>>> getIndices() const override { return indices; }
};
TEST_CASE("MetaData:TableMetaDataProvider") {
   auto schema = arrow::schema({arrow::field("a", arrow::int32()), arrow::field("b", arrow::float64())});

   // Create int32 array [1, 2]
   arrow::Int32Builder intBuilder;
   REQUIRE(intBuilder.Append(1).ok());
   REQUIRE(intBuilder.Append(2).ok());
   auto x = intBuilder.Finish().ValueOrDie();

   // Create float64 array [3.0, 4.0]
   arrow::DoubleBuilder doubleBuilder;
   REQUIRE(doubleBuilder.Append(3.0).ok());
   REQUIRE(doubleBuilder.Append(4.0).ok());
   auto y = doubleBuilder.Finish().ValueOrDie();
   auto batch = arrow::RecordBatch::Make(schema, 2, std::vector<std::shared_ptr<arrow::Array>>{x, y});
   Sample sample(batch);
   size_t numRows = 2;
   std::vector<std::string> primaryKey{"a"};
   std::vector<std::string> columnNames{"a", "b"};
   std::vector<std::unique_ptr<ColumnStatistics>> columnStatistics;
   columnStatistics.push_back(std::make_unique<ColumnStatistics>(10));
   columnStatistics.push_back(std::make_unique<ColumnStatistics>(std::nullopt));
   std::vector<std::pair<std::string, std::vector<std::string>>> indices{{"index1", {"a"}}};

   auto metaData = std::make_shared<MockTableMetaDataProvider>(sample, numRows, primaryKey, columnNames, std::move(columnStatistics), indices);
   SimpleByteWriter writer;
   Serializer serializer(writer);
   serializer.writeProperty(1, metaData);
   SimpleByteReader reader(writer.data(), writer.size());
   Deserializer deserializer(reader);
   auto metaData2 = deserializer.readProperty<std::shared_ptr<TableMetaDataProvider>>(1);
   REQUIRE(metaData2->getNumRows() == numRows);
   REQUIRE(metaData2->getPrimaryKey() == primaryKey);
   REQUIRE(metaData2->getColumnNames() == columnNames);
}
