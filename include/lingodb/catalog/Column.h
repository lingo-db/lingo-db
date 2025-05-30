#ifndef LINGODB_CATALOG_COLUMN_H
#define LINGODB_CATALOG_COLUMN_H
#include "Types.h"

namespace lingodb::catalog {
class Column {
   std::string columnName;
   Type logicalType;
   bool isNullable;

   public:
   Column(std::string columnName, Type type, bool isNullable) : columnName(std::move(columnName)), logicalType(std::move(type)), isNullable(isNullable) {}

   const Type& getLogicalType() const { return logicalType; }
   std::string_view getColumnName() const { return columnName; }
   bool getIsNullable() const { return isNullable; }
   void serialize(utility::Serializer& serializer) const;
   static Column deserialize(utility::Deserializer& deserializer);
};

} // namespace lingodb::catalog
#endif //LINGODB_CATALOG_COLUMN_H
