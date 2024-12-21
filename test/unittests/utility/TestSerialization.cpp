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
TEST_CASE("Serialization:Pair") {
   SimpleByteWriter writer;
   Serializer serializer(writer);
   serializer.writeProperty(1, std::make_pair(1, 2));
   SimpleByteReader reader(writer.data(), writer.size());
   Deserializer deserializer(reader);
   auto pair = deserializer.readProperty<std::pair<int, int>>(1);
   REQUIRE(pair.first == 1);
   REQUIRE(pair.second == 2);
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
   serializer.writeProperty(2, std::unique_ptr<TestObject>());
   SimpleByteReader reader(writer.data(), writer.size());
   Deserializer deserializer(reader);
   auto obj2 = deserializer.readProperty<std::unique_ptr<TestObject>>(1);
   auto obj3 = deserializer.readProperty<std::unique_ptr<TestObject>>(2);
   REQUIRE(obj2->getValue() == 42);
   REQUIRE(!obj3);
}

TEST_CASE("Serialization:SharedPtr") {
   SimpleByteWriter writer;
   Serializer serializer(writer);
   auto obj = std::make_shared<TestObject>(42);
   serializer.writeProperty(1, obj);
   serializer.writeProperty(2, std::shared_ptr<TestObject>());
   SimpleByteReader reader(writer.data(), writer.size());
   Deserializer deserializer(reader);
   auto obj2 = deserializer.readProperty<std::shared_ptr<TestObject>>(1);
   auto obj3 = deserializer.readProperty<std::shared_ptr<TestObject>>(2);
   REQUIRE(obj2->getValue() == 42);
   REQUIRE(!obj3);
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
TEST_CASE("Serialization:vector<shared_ptr<Custom>>") {
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
//test serialization with enum classes based on uint8_t, uint16_t:
enum class EnumClassUint8 : uint8_t { A = 1,
                                      B = 2,
                                      C = 3 };
enum class EnumClassUint16 : uint16_t { A = 1,
                                        B = 2,
                                        C = 3 };
TEST_CASE("Serialization:EnumClassUint8") {
   SimpleByteWriter writer;
   Serializer serializer(writer);
   serializer.writeProperty(1, EnumClassUint8::A);
   serializer.writeProperty(2, EnumClassUint8::B);
   serializer.writeProperty(3, EnumClassUint8::C);
   SimpleByteReader reader(writer.data(), writer.size());
   Deserializer deserializer(reader);
   REQUIRE(deserializer.readProperty<EnumClassUint8>(1) == EnumClassUint8::A);
   REQUIRE(deserializer.readProperty<EnumClassUint8>(2) == EnumClassUint8::B);
   REQUIRE(deserializer.readProperty<EnumClassUint8>(3) == EnumClassUint8::C);
}
TEST_CASE("Serialization:EnumClassUint16") {
   SimpleByteWriter writer;
   Serializer serializer(writer);
   serializer.writeProperty(1, EnumClassUint16::A);
   serializer.writeProperty(2, EnumClassUint16::B);
   serializer.writeProperty(3, EnumClassUint16::C);
   SimpleByteReader reader(writer.data(), writer.size());
   Deserializer deserializer(reader);
   REQUIRE(deserializer.readProperty<EnumClassUint16>(1) == EnumClassUint16::A);
   REQUIRE(deserializer.readProperty<EnumClassUint16>(2) == EnumClassUint16::B);
   REQUIRE(deserializer.readProperty<EnumClassUint16>(3) == EnumClassUint16::C);
}

namespace {
class AbstractTestObject {
   protected:
   uint8_t type;
   AbstractTestObject(uint8_t type) : type(type) {}

   public:
   virtual void serializeChildClass(Serializer& serializer) const = 0;
   void serialize(Serializer& serializer) const {
      serializer.writeProperty(0, type);
      serializeChildClass(serializer);
   }
   static std::shared_ptr<AbstractTestObject> deserialize(Deserializer& deserializer);
   virtual int getValue() const = 0;
   virtual ~AbstractTestObject() = default;
};
class TestObject1 : public AbstractTestObject {
   int value;

   public:
   TestObject1(int value) : AbstractTestObject(0), value(value) {}
   void serializeChildClass(Serializer& serializer) const override {
      serializer.writeProperty(1, value);
   }
   static std::shared_ptr<AbstractTestObject> deserialize(Deserializer& deserializer) {
      auto value = deserializer.readProperty<int>(1);
      return std::make_shared<TestObject1>(value);
   }
   int getValue() const override {
      return value;
   }
};

std::shared_ptr<AbstractTestObject> AbstractTestObject::deserialize(lingodb::utility::Deserializer& deserializer) {
   auto type = deserializer.readProperty<uint8_t>(0);
   switch (type) {
      case 0: {
         return TestObject1::deserialize(deserializer);
      }
      default:
         throw std::runtime_error("Unknown type");
   }
}
} // namespace
TEST_CASE("Serialization:AbstractClass") {
   SimpleByteWriter writer;
   Serializer serializer(writer);
   std::shared_ptr<AbstractTestObject> obj = std::make_shared<TestObject1>(42);
   serializer.writeProperty(1, obj);
   SimpleByteReader reader(writer.data(), writer.size());
   Deserializer deserializer(reader);
   auto obj2 = deserializer.readProperty<std::shared_ptr<AbstractTestObject>>(1);
   REQUIRE(obj2->getValue() == 42);
}
TEST_CASE("Serialization:optional") {
   SimpleByteWriter writer;
   Serializer serializer(writer);
   std::optional<int> val = 42;
   serializer.writeProperty(1, val);
   SimpleByteReader reader(writer.data(), writer.size());
   Deserializer deserializer(reader);
   auto val2 = deserializer.readProperty<std::optional<int>>(1);
   REQUIRE(val2.has_value());
   REQUIRE(val2.value() == 42);
}
TEST_CASE("Serialization:unorderd_map") {
   SimpleByteWriter writer;
   Serializer serializer(writer);
   std::unordered_map<std::string, size_t> m;
   m.insert({"foo", 2});
   m.insert({"bar", 42});
   serializer.writeProperty(1, m);
   SimpleByteReader reader(writer.data(), writer.size());
   Deserializer deserializer(reader);
   auto m2 = deserializer.readProperty<std::unordered_map<std::string, size_t>>(1);
   REQUIRE(m.at("foo") == 2);
   REQUIRE(m.at("bar") == 42);
}

TEST_CASE("Serialization:hex_string") {
   auto hexString = lingodb::utility::serializeToHexString(std::string("Hello World!"));
   auto deSerialized = lingodb::utility::deserializeFromHexString<std::string>(hexString);
   REQUIRE(deSerialized == "Hello World!");
}