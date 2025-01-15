#ifndef LINGODB_CATALOG_TYPES_H
#define LINGODB_CATALOG_TYPES_H

#include <memory>
#include <optional>
namespace lingodb::utility {
class Serializer;
class Deserializer;
} //end namespace lingodb::utility
namespace lingodb::catalog {
enum class LogicalTypeId : uint8_t {
   INVALID = 0,
   SQLNULL = 1,
   BOOLEAN = 2,
   INT = 3,
   FLOAT = 4,
   DOUBLE = 5,
   DECIMAL = 6,
   DATE = 7,
   TIMESTAMP = 8,
   INTERVAL = 9,
   DURATION = 10,
   CHAR = 11,
   STRING = 12,
};
class TypeInfo {
   protected:
   enum class TypeInfoType : uint8_t {
      IntegerInfo = 0,
      DecimalInfo = 1,
      StringInfo = 2,
      TimestampInfo = 3,
   };
   TypeInfoType infoType;
   TypeInfo(TypeInfoType infoType) : infoType(infoType) {}

   public:
   void serialize(utility::Serializer& serializer) const;
   virtual void serializeConcrete(utility::Serializer& serializer) const = 0;
   static std::shared_ptr<TypeInfo> deserialize(utility::Deserializer& deserializer);
   virtual ~TypeInfo() = default;
};
class Type {
   LogicalTypeId id;
   std::shared_ptr<TypeInfo> info;

   public:
   Type(LogicalTypeId id, std::shared_ptr<TypeInfo> info) : id(id), info(std::move(info)) {}
   std::string toString() const;
   LogicalTypeId getTypeId() { return id; }
   template <class T>
   std::shared_ptr<T> getInfo() {
      return std::dynamic_pointer_cast<T>(info);
   }
   void serialize(utility::Serializer& serializer) const;
   static Type deserialize(utility::Deserializer& deserializer);
   static Type makeIntType(size_t width, bool isSigned);
   static Type int8() { return makeIntType(8, true); }
   static Type int16() { return makeIntType(16, true); }
   static Type int32() { return makeIntType(32, true); }
   static Type int64() { return makeIntType(64, true); }
};
class IntTypeInfo : public TypeInfo {
   bool isSigned;
   size_t bitWidth;

   public:
   IntTypeInfo(bool isSigned, size_t bitWidth) : TypeInfo(TypeInfoType::IntegerInfo), isSigned(isSigned), bitWidth(bitWidth) {}
   void serializeConcrete(utility::Serializer& serializer) const override;
   static std::shared_ptr<IntTypeInfo> deserialize(utility::Deserializer& deserializer);
   std::string toString();
};
class DecimalTypeInfo : public TypeInfo {
   size_t precision;
   size_t scale;

   public:
   DecimalTypeInfo(size_t precision, size_t scale) : TypeInfo(TypeInfoType::DecimalInfo), precision(precision), scale(scale) {}
   void serializeConcrete(utility::Serializer& serializer) const override;
   static std::shared_ptr<DecimalTypeInfo> deserialize(utility::Deserializer& deserializer);
   std::string toString();
};
class StringTypeInfo : public TypeInfo {
   std::string collation;
   std::optional<size_t> maxLength;

   public:
   StringTypeInfo(std::string collation, std::optional<size_t> maxLength) : TypeInfo(TypeInfoType::StringInfo), collation(std::move(collation)), maxLength(maxLength) {}
   void serializeConcrete(utility::Serializer& serializer) const override;
   static std::shared_ptr<StringTypeInfo> deserialize(utility::Deserializer& deserializer);
   std::string toString();
};
class TimestampTypeInfo : public TypeInfo {
   enum class TimestampUnit : uint8_t {
      NANOS = 0,
      MICROS = 1,
      MILLIS = 2,
      SECONDS = 3,
   };
   std::optional<std::string> timezone;
   TimestampUnit unit;

   public:
   TimestampTypeInfo(std::optional<std::string> timezone, TimestampUnit unit) : TypeInfo(TypeInfoType::TimestampInfo), timezone(std::move(timezone)), unit(unit) {}
   void serializeConcrete(utility::Serializer& serializer) const override;
   static std::shared_ptr<TimestampTypeInfo> deserialize(utility::Deserializer& deserializer);
   std::string toString();
};
} //end namespace lingodb::catalog
#endif //LINGODB_CATALOG_TYPES_H
