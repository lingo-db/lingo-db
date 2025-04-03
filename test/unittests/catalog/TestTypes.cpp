#include "catch2/catch_all.hpp"
#include "lingodb/catalog/Types.h"
#include "lingodb/utility/Serialization.h"
using namespace lingodb::utility;
using namespace lingodb::catalog;

namespace {
void testSerialization(const Type& type) {
   SimpleByteWriter writer;
   Serializer serializer(writer);
   serializer.writeProperty(1, type);
   SimpleByteReader reader(writer.data(), writer.size());
   Deserializer deserializer(reader);
   auto type2 = deserializer.readProperty<Type>(1);
   REQUIRE(type.toString() == type2.toString());
}
} // namespace

TEST_CASE("Types:bool") {
   auto type = Type::boolean();
   REQUIRE(type.getTypeId() == LogicalTypeId::BOOLEAN);
   REQUIRE(type.toString() == "bool");
   testSerialization(type);
}
TEST_CASE("Types:integer") {
   auto i8Type = Type::int8();
   auto i64Type = Type::int64();
   REQUIRE(i8Type.getTypeId() == LogicalTypeId::INT);
   REQUIRE(i8Type.toString() == "int8");
   REQUIRE(i64Type.getTypeId() == LogicalTypeId::INT);
   REQUIRE(i64Type.toString() == "int64");
   testSerialization(i8Type);
   testSerialization(i64Type);
}
TEST_CASE("Types:float") {
   auto f32Type = Type::f32();
   auto f64Type = Type::f64();
   REQUIRE(f32Type.getTypeId() == LogicalTypeId::FLOAT);
   REQUIRE(f32Type.toString() == "float");
   REQUIRE(f64Type.getTypeId() == LogicalTypeId::DOUBLE);
   REQUIRE(f64Type.toString() == "double");
   testSerialization(f32Type);
   testSerialization(f64Type);
}
TEST_CASE("Types:decimal") {
   auto type = Type::decimal(10, 2);
   REQUIRE(type.getTypeId() == LogicalTypeId::DECIMAL);
   REQUIRE(type.toString() == "decimal(10, 2)");
   testSerialization(type);
}

TEST_CASE("Types:string") {
   auto type = Type::stringType();
   REQUIRE(type.getTypeId() == LogicalTypeId::STRING);
   REQUIRE(type.toString() == "string");
   testSerialization(type);
}

TEST_CASE("Types:timestamp") {
   auto type = Type::timestamp();
   REQUIRE(type.getTypeId() == LogicalTypeId::TIMESTAMP);
   REQUIRE(type.toString() == "timestamp<ns>");
   testSerialization(type);
}
TEST_CASE("Types:interval") {
   auto daytimeType = Type::intervalDaytime();
   auto monthType = Type::intervalMonths();
   REQUIRE(daytimeType.getTypeId() == LogicalTypeId::INTERVAL);
   REQUIRE(daytimeType.toString() == "interval<daytime>");
   REQUIRE(monthType.getTypeId() == LogicalTypeId::INTERVAL);
   REQUIRE(monthType.toString() == "interval<month>");
   testSerialization(daytimeType);
   testSerialization(monthType);
}
