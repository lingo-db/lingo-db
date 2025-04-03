#include "lingodb/catalog/Column.h"

#include "lingodb/utility/Serialization.h"

namespace lingodb::catalog {
void Column::serialize(utility::Serializer& serializer) const {
   serializer.writeProperty(1, columnName);
   serializer.writeProperty(2, logicalType);
   serializer.writeProperty(3, isNullable);
}
Column Column::deserialize(utility::Deserializer& deserializer) {
   auto columnName = deserializer.readProperty<std::string>(1);
   auto logicalType = deserializer.readProperty<Type>(2);
   auto isNullable = deserializer.readProperty<bool>(3);
   return Column(columnName, logicalType, isNullable);
}
} // namespace lingodb::catalog