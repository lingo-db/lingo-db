#include "catch2/catch_all.hpp"
#include "lingodb/catalog/Types.h"
#include "lingodb/utility/Serialization.h"
using namespace lingodb::utility;
using namespace lingodb::catalog;
TEST_CASE("Types:serialization") {
   SimpleByteWriter writer;
   Serializer serializer(writer);
   auto i8Type = Type::int8();
   serializer.writeProperty(1, i8Type);
   SimpleByteReader reader(writer.data(), writer.size());
   Deserializer deserializer(reader);
   auto i8Type2 = deserializer.readProperty<Type>(1);
   REQUIRE(i8Type2.getTypeId()==LogicalTypeId::INT);
}