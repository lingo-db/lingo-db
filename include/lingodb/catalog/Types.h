#ifndef LINGODB_CATALOG_TYPES_H
#define LINGODB_CATALOG_TYPES_H

#include <memory>
#include <mlir/Dialect/Func/Transforms/Passes.h.inc>
#include <optional>
#include <string>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
namespace lingodb::utility {
class Serializer;
class Deserializer;
} //end namespace lingodb::utility
namespace lingodb::catalog {
class MLIRTypeCreator;

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
   CHAR = 10,
   STRING = 11,
   NONE = 12,
   INDEX = 13,
};
class TypeInfo {
   protected:
   enum class TypeInfoType : uint8_t {
      IntegerInfo = 0,
      DecimalInfo = 1,
      StringInfo = 2,
      TimestampInfo = 3,
      CharInfo = 4,
      DateInfo = 5,
      IntervalInfo = 6,
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
   std::shared_ptr<catalog::MLIRTypeCreator> mlirTypeCreator;

   public:
   Type(LogicalTypeId id, std::shared_ptr<TypeInfo> info);
   std::string toString() const;
   LogicalTypeId getTypeId() const { return id; }
   template <class T>
   std::shared_ptr<T> getInfo() {
      return std::dynamic_pointer_cast<T>(info);
   }
   std::shared_ptr<catalog::MLIRTypeCreator> getMLIRTypeCreator() const { return mlirTypeCreator; }
   void serialize(utility::Serializer& serializer) const;
   static Type deserialize(utility::Deserializer& deserializer);
   static Type makeIntType(size_t width, bool isSigned);
   static Type boolean() { return Type(LogicalTypeId::BOOLEAN, nullptr); }
   static Type int8() { return makeIntType(8, true); }
   static Type int16() { return makeIntType(16, true); }
   static Type int32() { return makeIntType(32, true); }
   static Type int64() { return makeIntType(64, true); }
   static Type f32() { return Type(LogicalTypeId::FLOAT, nullptr); }
   static Type f64() { return Type(LogicalTypeId::DOUBLE, nullptr); }
   static Type decimal(size_t precision, size_t scale);
   static Type charType(size_t length);
   static Type stringType();
   static Type timestamp();
   static Type intervalDaytime();
   static Type intervalMonths();
   static Type noneType();
   static Type index();
};
class NullableType {
   public:
   NullableType(Type type);
   NullableType(Type type, bool isNullable);
   Type type;
   std::shared_ptr<NullableType> castType = nullptr;
   bool isNullable;
   bool useZeroInsteadOfNull = false;
   mlir::Type toMlirType(mlir::MLIRContext* context) const;
   mlir::Value castValueToThisType(mlir::OpBuilder& builder, mlir::Value valueToCast, bool valueNullable) const;
   mlir::Value castValue(mlir::OpBuilder& builder, mlir::Value valueToCast) const;
   bool isNumeric() const;

   bool operator==(NullableType&);
   bool operator!=(NullableType&);
};
class IntTypeInfo : public TypeInfo {
   bool isSigned;
   size_t bitWidth;

   public:
   IntTypeInfo(bool isSigned, size_t bitWidth) : TypeInfo(TypeInfoType::IntegerInfo), isSigned(isSigned), bitWidth(bitWidth) {}
   void serializeConcrete(utility::Serializer& serializer) const override;
   static std::shared_ptr<IntTypeInfo> deserialize(utility::Deserializer& deserializer);
   std::string toString();
   bool getIsSigned() { return isSigned; }
   size_t getBitWidth() { return bitWidth; }
};
class DecimalTypeInfo : public TypeInfo {
   size_t precision;
   size_t scale;

   public:
   DecimalTypeInfo(size_t precision, size_t scale) : TypeInfo(TypeInfoType::DecimalInfo), precision(precision), scale(scale) {}
   void serializeConcrete(utility::Serializer& serializer) const override;
   static std::shared_ptr<DecimalTypeInfo> deserialize(utility::Deserializer& deserializer);
   std::string toString();
   size_t getPrecision() { return precision; }
   size_t getScale() { return scale; }
};
class CharTypeInfo : public TypeInfo {
   size_t length;

   public:
   CharTypeInfo(size_t length) : TypeInfo(TypeInfoType::CharInfo), length(length) {}
   void serializeConcrete(utility::Serializer& serializer) const override;
   static std::shared_ptr<CharTypeInfo> deserialize(utility::Deserializer& deserializer);
   std::string toString();
   size_t getLength() { return length; }
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
   public:
   enum class TimestampUnit : uint8_t {
      SECONDS = 0,
      MILLIS = 1,
      MICROS = 2,
      NANOS = 3,

   };

   private:
   std::optional<std::string> timezone;
   TimestampUnit unit;

   public:
   TimestampTypeInfo(std::optional<std::string> timezone, TimestampUnit unit) : TypeInfo(TypeInfoType::TimestampInfo), timezone(std::move(timezone)), unit(unit) {}
   void serializeConcrete(utility::Serializer& serializer) const override;
   static std::shared_ptr<TimestampTypeInfo> deserialize(utility::Deserializer& deserializer);
   std::string toString();
   auto getTimezone() { return timezone; }
   auto getUnit() { return unit; }
};

class DateTypeInfo : public TypeInfo {
   public:
   enum class DateUnit : uint8_t {
      DAY = 0,
      MILLIS = 1,
   };

   private:
   DateUnit unit;

   public:
   DateTypeInfo(DateUnit unit) : TypeInfo(TypeInfoType::DateInfo), unit(unit) {}
   void serializeConcrete(utility::Serializer& serializer) const override;
   static std::shared_ptr<DateTypeInfo> deserialize(utility::Deserializer& deserializer);
   std::string toString();
   auto getUnit() { return unit; }
};
class IntervalTypeInfo : public TypeInfo {
   public:
   enum class IntervalUnit : uint8_t {
      MONTH = 0,
      DAYTIME = 1,
   };

   private:
   IntervalUnit unit;

   public:
   IntervalTypeInfo(IntervalUnit unit) : TypeInfo(TypeInfoType::IntervalInfo), unit(unit) {}
   void serializeConcrete(utility::Serializer& serializer) const override;
   static std::shared_ptr<IntervalTypeInfo> deserialize(utility::Deserializer& deserializer);
   std::string toString();
   auto getUnit() { return unit; }
};
} //end namespace lingodb::catalog
#endif //LINGODB_CATALOG_TYPES_H
