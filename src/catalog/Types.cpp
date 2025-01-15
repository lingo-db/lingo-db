#include "lingodb/catalog/Types.h"
#include "lingodb/utility/Serialization.h"

namespace lingodb::catalog {
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
      default:
         throw std::runtime_error("not implemented");
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
         return "date";
      case LogicalTypeId::TIMESTAMP:
         return std::dynamic_pointer_cast<TimestampTypeInfo>(info)->toString();
      case LogicalTypeId::INTERVAL:
         return "interval";
      case LogicalTypeId::DURATION:
         return "duration";
      case LogicalTypeId::CHAR:
         return "char";
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
Type Type::makeIntType(size_t width, bool isSigned) {
   return Type(LogicalTypeId::INT, std::make_shared<IntTypeInfo>(width, isSigned));
}
} //end namespace lingodb::catalog