#include "catch2/catch_all.hpp"

#include "lingodb/utility/Serialization.h"
using namespace lingodb::utility;
namespace {
class TestObject {
   int value;

   public:
   TestObject(int value) : value(value) {}
   void serialize(Serializer& serializer) const {
      serializer.writeProperty(1, value);
   }
   static TestObject deserialize(Deserializer& deserializer) {
      auto val = deserializer.readProperty<int>(1);
      return TestObject(val);
   }
   int getValue() const {
      return value;
   }
};
} // namespace
TEST_CASE("Serialization:BasicTypes") {
   SimpleByteWriter writer;
   Serializer serializer(writer);
   serializer.writeProperty(1, true);
   serializer.writeProperty(2, 1);
   serializer.writeProperty(3, 1.0f);
   serializer.writeProperty(4, 1.0);
   serializer.writeProperty(5, std::string_view("Hello"));
   SimpleByteReader reader(writer.data(), writer.size());
   Deserializer deserializer(reader);
   REQUIRE(deserializer.readProperty<bool>(1) == true);
   REQUIRE(deserializer.readProperty<int>(2) == 1);
   REQUIRE(deserializer.readProperty<float>(3) == 1.0f);
   REQUIRE(deserializer.readProperty<double>(4) == 1.0);
   REQUIRE(deserializer.readProperty<std::string>(5) == "Hello");
}

//dumps bytes as hex string
void dumpBytes(const std::byte* data, size_t size) {
   for (size_t i = 0; i < size; i++) {
      printf("%02X ", static_cast<uint8_t>(data[i]));
   }
   printf("\n");
}

TEST_CASE("Serialization:CustomType") {
   SimpleByteWriter writer;
   Serializer serializer(writer);
   TestObject obj(42);
   serializer.writeProperty(1, obj);
   SimpleByteReader reader(writer.data(), writer.size());
   Deserializer deserializer(reader);
   auto obj2 = deserializer.readProperty<TestObject>(1);
   REQUIRE(obj2.getValue() == 42);
}

TEST_CASE("Serialization:UniquePtr") {
   SimpleByteWriter writer;
   Serializer serializer(writer);
   auto obj = std::make_unique<TestObject>(42);
   serializer.writeProperty(1, obj);
   SimpleByteReader reader(writer.data(), writer.size());
   Deserializer deserializer(reader);
   auto obj2 = deserializer.readProperty<std::unique_ptr<TestObject>>(1);
   REQUIRE(obj2->getValue() == 42);
}

TEST_CASE("Serialization:SharedPtr") {
   SimpleByteWriter writer;
   Serializer serializer(writer);
   auto obj = std::make_shared<TestObject>(42);
   serializer.writeProperty(1, obj);
   SimpleByteReader reader(writer.data(), writer.size());
   Deserializer deserializer(reader);
   auto obj2 = deserializer.readProperty<std::shared_ptr<TestObject>>(1);
   REQUIRE(obj2->getValue() == 42);
}
TEST_CASE("Serialization:Vector") {
   SimpleByteWriter writer;
   Serializer serializer(writer);
   std::vector<int> vec = {1, 2, 3, 4, 5};
   serializer.writeProperty(1, vec);
   SimpleByteReader reader(writer.data(), writer.size());
   Deserializer deserializer(reader);
   auto vec2 = deserializer.readProperty<std::vector<int>>(1);
   REQUIRE(vec2 == vec);
}
TEST_CASE("Serialization:vector<shared_ptr<Custom>>"){
        SimpleByteWriter writer;
        Serializer serializer(writer);
        std::vector<std::shared_ptr<TestObject>> vec;
        vec.push_back(std::make_shared<TestObject>(1));
        vec.push_back(std::make_shared<TestObject>(2));
        vec.push_back(std::make_shared<TestObject>(3));
        serializer.writeProperty(1, vec);
        SimpleByteReader reader(writer.data(), writer.size());
        Deserializer deserializer(reader);
        auto vec2 = deserializer.readProperty<std::vector<std::shared_ptr<TestObject>>>(1);
        REQUIRE(vec2.size() == 3);
        REQUIRE(vec2[0]->getValue() == 1);
        REQUIRE(vec2[1]->getValue() == 2);
        REQUIRE(vec2[2]->getValue() == 3);
}