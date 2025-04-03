#include "lingodb/catalog/Types.h"
#include "lingodb/catalog/MLIRTypes.h"
#include "lingodb/utility/Serialization.h"

namespace lingodb::catalog {
Type::Type(lingodb::catalog::LogicalTypeId id, std::shared_ptr<TypeInfo> infoInput) : id(id), info(std::move(infoInput)) {
   switch (id) {
      case LogicalTypeId::INVALID:
         break;
      case LogicalTypeId::SQLNULL:
         break;
      case LogicalTypeId::BOOLEAN:
         mlirTypeCreator = lingodb::catalog::createBoolTypeCreator();
         break;
      case LogicalTypeId::INT:
         mlirTypeCreator = lingodb::catalog::createIntTypeCreator(std::dynamic_pointer_cast<IntTypeInfo>(info));
         break;
      case LogicalTypeId::FLOAT:
         mlirTypeCreator = lingodb::catalog::createFloatTypeCreator();
         break;
      case LogicalTypeId::DOUBLE:
         mlirTypeCreator = lingodb::catalog::createDoubleTypeCreator();
         break;
      case LogicalTypeId::DECIMAL:
         mlirTypeCreator = lingodb::catalog::createDecimalTypeCreator(std::dynamic_pointer_cast<DecimalTypeInfo>(info));
         break;
      case LogicalTypeId::DATE:
         mlirTypeCreator = lingodb::catalog::createDateTypeCreator(std::dynamic_pointer_cast<DateTypeInfo>(info));
         break;
      case LogicalTypeId::TIMESTAMP:
         mlirTypeCreator = lingodb::catalog::createTimestampTypeCreator(std::dynamic_pointer_cast<TimestampTypeInfo>(info));
         break;
      case LogicalTypeId::INTERVAL:
         mlirTypeCreator = lingodb::catalog::createIntervalTypeCreator(std::dynamic_pointer_cast<IntervalTypeInfo>(info));
         break;
      case LogicalTypeId::CHAR:
         mlirTypeCreator = lingodb::catalog::createCharTypeCreator(std::dynamic_pointer_cast<CharTypeInfo>(info));
         break;
      case LogicalTypeId::STRING:
         mlirTypeCreator = lingodb::catalog::createStringTypeCreator(std::dynamic_pointer_cast<StringTypeInfo>(info));
         break;
   }
}
void Type::serialize(utility::Serializer& serializer) const {
   serializer.writeProperty<LogicalTypeId>(0, id);
   serializer.writeProperty<std::shared_ptr<TypeInfo>>(1, info);
}
Type Type::deserialize(utility::Deserializer& deserializer) {
   auto id = deserializer.readProperty<LogicalTypeId>(0);
   auto info = deserializer.readProperty<std::shared_ptr<TypeInfo>>(1);
   return Type{id, info};
}
void TypeInfo::serialize(utility::Serializer& serializer) const {
   serializer.writeProperty(42, infoType);
   serializeConcrete(serializer);
}
std::shared_ptr<TypeInfo> TypeInfo::deserialize(utility::Deserializer& deserializer) {
   auto infoType = deserializer.readProperty<TypeInfoType>(42);
   switch (infoType) {
      case TypeInfoType::IntegerInfo:
         return IntTypeInfo::deserialize(deserializer);
      case TypeInfoType::DecimalInfo:
         return DecimalTypeInfo::deserialize(deserializer);
      case TypeInfoType::StringInfo:
         return StringTypeInfo::deserialize(deserializer);
      case TypeInfoType::TimestampInfo:
         return TimestampTypeInfo::deserialize(deserializer);
      case TypeInfoType::CharInfo:
         return CharTypeInfo::deserialize(deserializer);
      case TypeInfoType::DateInfo:
         return DateTypeInfo::deserialize(deserializer);
      case TypeInfoType::IntervalInfo:
         return IntervalTypeInfo::deserialize(deserializer);
   }
}
void IntTypeInfo::serializeConcrete(utility::Serializer& serializer) const {
   serializer.writeProperty(0, isSigned);
   serializer.writeProperty(1, bitWidth);
}
std::shared_ptr<IntTypeInfo> IntTypeInfo::deserialize(utility::Deserializer& deserializer) {
   auto isSigned = deserializer.readProperty<bool>(0);
   auto bitWidth = deserializer.readProperty<size_t>(1);
   return std::make_shared<IntTypeInfo>(isSigned, bitWidth);
}
void DecimalTypeInfo::serializeConcrete(utility::Serializer& serializer) const {
   serializer.writeProperty(0, precision);
   serializer.writeProperty(1, scale);
}
std::shared_ptr<DecimalTypeInfo> DecimalTypeInfo::deserialize(utility::Deserializer& deserializer) {
   auto precision = deserializer.readProperty<size_t>(0);
   auto scale = deserializer.readProperty<size_t>(1);
   return std::make_shared<DecimalTypeInfo>(precision, scale);
}
void StringTypeInfo::serializeConcrete(utility::Serializer& serializer) const {
   serializer.writeProperty(0, collation);
   serializer.writeProperty(1, maxLength);
}
std::shared_ptr<StringTypeInfo> StringTypeInfo::deserialize(utility::Deserializer& deserializer) {
   auto collation = deserializer.readProperty<std::string>(0);
   auto maxLength = deserializer.readProperty<std::optional<size_t>>(1);
   return std::make_shared<StringTypeInfo>(collation, maxLength);
}
void TimestampTypeInfo::serializeConcrete(utility::Serializer& serializer) const {
   serializer.writeProperty(0, timezone);
   serializer.writeProperty(1, unit);
}
std::shared_ptr<TimestampTypeInfo> TimestampTypeInfo::deserialize(utility::Deserializer& deserializer) {
   auto timezone = deserializer.readProperty<std::optional<std::string>>(0);
   auto unit = deserializer.readProperty<TimestampTypeInfo::TimestampUnit>(1);
   return std::make_shared<TimestampTypeInfo>(timezone, unit);
}
std::string Type::toString() const {
   switch (id) {
      case LogicalTypeId::INVALID:
         return "invalid";
      case LogicalTypeId::SQLNULL:
         return "null";
      case LogicalTypeId::BOOLEAN:
         return "bool";
      case LogicalTypeId::INT:
         return std::dynamic_pointer_cast<IntTypeInfo>(info)->toString();
      case LogicalTypeId::FLOAT:
         return "float";
      case LogicalTypeId::DOUBLE:
         return "double";
      case LogicalTypeId::DECIMAL:
         return std::dynamic_pointer_cast<DecimalTypeInfo>(info)->toString();
      case LogicalTypeId::DATE:
         return std::dynamic_pointer_cast<DateTypeInfo>(info)->toString();
      case LogicalTypeId::TIMESTAMP:
         return std::dynamic_pointer_cast<TimestampTypeInfo>(info)->toString();
      case LogicalTypeId::INTERVAL:
         return std::dynamic_pointer_cast<IntervalTypeInfo>(info)->toString();
      case LogicalTypeId::CHAR:
         return std::dynamic_pointer_cast<CharTypeInfo>(info)->toString();
      case LogicalTypeId::STRING:
         return std::dynamic_pointer_cast<StringTypeInfo>(info)->toString();
   }
}
std::string IntTypeInfo::toString() {
   std::string res = isSigned ? "int" : "uint";
   res += std::to_string(bitWidth);
   return res;
}
std::string DecimalTypeInfo::toString() {
   return "decimal(" + std::to_string(precision) + ", " + std::to_string(scale) + ")";
}
std::string StringTypeInfo::toString() {
   std::string res = "string";
   if (!collation.empty()) {
      res += "<COLLATE " + collation + ">";
   }

   if (maxLength) {
      res += "(" + std::to_string(*maxLength) + ")";
   }
   return res;
}
std::string TimestampTypeInfo::toString() {
   std::string res = "timestamp<";
   if (timezone) {
      res += *timezone + ", ";
   }
   switch (unit) {
      case TimestampUnit::NANOS:
         res += "ns";
         break;
      case TimestampUnit::MICROS:
         res += "us";
         break;
      case TimestampUnit::MILLIS:
         res += "ms";
         break;
      case TimestampUnit::SECONDS:
         res += "s";
         break;
   }
   res += ">";
   return res;
}
std::string CharTypeInfo::toString() {
   return "char(" + std::to_string(length) + ")";
}
std::shared_ptr<CharTypeInfo> CharTypeInfo::deserialize(utility::Deserializer& deserializer) {
   auto length = deserializer.readProperty<size_t>(0);
   return std::make_shared<CharTypeInfo>(length);
}
void CharTypeInfo::serializeConcrete(utility::Serializer& serializer) const {
   serializer.writeProperty(0, length);
}
std::shared_ptr<DateTypeInfo> DateTypeInfo::deserialize(utility::Deserializer& deserializer) {
   auto unit = deserializer.readProperty<DateTypeInfo::DateUnit>(0);
   return std::make_shared<DateTypeInfo>(unit);
}
void DateTypeInfo::serializeConcrete(utility::Serializer& serializer) const {
   serializer.writeProperty(0, unit);
}
std::string DateTypeInfo::toString() {
   std::string res = "date<";
   switch (unit) {
      case DateUnit::DAY:
         res += "day";
         break;
      case DateUnit::MILLIS:
         res += "ms";
         break;
   }
   res += ">";
   return res;
}
std::shared_ptr<IntervalTypeInfo> IntervalTypeInfo::deserialize(utility::Deserializer& deserializer) {
   auto unit = deserializer.readProperty<IntervalTypeInfo::IntervalUnit>(0);
   return std::make_shared<IntervalTypeInfo>(unit);
}
void IntervalTypeInfo::serializeConcrete(utility::Serializer& serializer) const {
   serializer.writeProperty(0, unit);
}
std::string IntervalTypeInfo::toString() {
   std::string res = "interval<";
   switch (unit) {
      case IntervalUnit::MONTH:
         res += "month";
         break;
      case IntervalUnit::DAYTIME:
         res += "daytime";
         break;
   }
   res += ">";
   return res;
}
Type Type::makeIntType(size_t width, bool isSigned) {
   return Type(LogicalTypeId::INT, std::make_shared<IntTypeInfo>(isSigned, width));
}

Type Type::decimal(size_t precision, size_t scale) {
   return Type(LogicalTypeId::DECIMAL, std::make_shared<DecimalTypeInfo>(precision, scale));
}
Type Type::charType(size_t length) {
   return Type(LogicalTypeId::CHAR, std::make_shared<CharTypeInfo>(length));
}
Type Type::stringType() {
   return Type(LogicalTypeId::STRING, std::make_shared<StringTypeInfo>("", std::nullopt));
}
Type Type::timestamp() {
   return Type(LogicalTypeId::TIMESTAMP, std::make_shared<TimestampTypeInfo>(std::nullopt, TimestampTypeInfo::TimestampUnit::NANOS));
}
Type Type::intervalDaytime() {
   return Type(LogicalTypeId::INTERVAL, std::make_shared<IntervalTypeInfo>(IntervalTypeInfo::IntervalUnit::DAYTIME));
}
Type Type::intervalMonths() {
   return Type(LogicalTypeId::INTERVAL, std::make_shared<IntervalTypeInfo>(IntervalTypeInfo::IntervalUnit::MONTH));
}
} //end namespace lingodb::catalog