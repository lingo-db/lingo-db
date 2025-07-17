#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
namespace lingodb::ast {
enum class ConstantType : uint8_t {
   INT = 1,

   UINT = 2,
   FLOAT = 3,
   STRING = 4,
   INTERVAL = 5,
   NULL_P = 6,
   BOOLEAN = 7,

   INVALID = 99,

};
class Value {
   public:
   virtual ~Value() = default;
   explicit Value(ConstantType type) : type(type) {}
   ConstantType type;

   virtual std::string toString() = 0;
   size_t hash() {
      return std::hash<std::string>{}(toString());
   }
   bool operator==(Value& other) {
       return toString() == other.toString();
    }
};

class UnsignedIntValue : public Value {
   public:
   explicit UnsignedIntValue(size_t iVal) : Value(ConstantType::UINT), iVal(iVal) {}
   size_t iVal;
   std::string toString() override {
      return "uint:" + std::to_string(iVal);
   }

};

class IntValue : public Value {
   public:
   explicit IntValue(int iVal) : Value(ConstantType::INT), iVal(iVal) {}
   int iVal;
   std::string toString() override {
      return "int:" + std::to_string(iVal);
   }
};

class FloatValue : public Value {
   public:
   explicit FloatValue(std::string fVal) : Value(ConstantType::FLOAT), fVal(fVal) {}
   //TODO float or double?
   std::string fVal;
   std::string toString() override {
      return "float: " + fVal;
   }
};

class StringValue : public Value {
   public:
   explicit StringValue(std::string sVal) : Value(ConstantType::STRING), sVal(sVal) {}
   std::string sVal;
   std::string toString() override{
      return "string: " + sVal;
   }
};

class BoolValue : public Value {
   public:
   explicit BoolValue(bool bVal) : Value(ConstantType::BOOLEAN), bVal(bVal) {}
   bool bVal;
   std::string toString() override {
      return "bool:" + std::to_string(bVal);
   }
};

class NullValue : public Value {
   public:
   explicit NullValue() : Value(ConstantType::NULL_P) {}
   std::string toString() override{
      return "NULL";
   }
};

class Interval {
   public:
   int32_t months;
   int32_t days;
   int64_t micros;
};

class IntervalValue : public Value {
   public:
   explicit IntervalValue(Interval iVal) : Value(ConstantType::INTERVAL), iVal(iVal) {}
   Interval iVal;
   std::string toString() override{
      return "interval[...]";
   }
};

} // namespace lingodb::ast