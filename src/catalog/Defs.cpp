#include "lingodb/catalog/Defs.h"
#include "lingodb/utility/Serialization.h"
void lingodb::catalog::CreateTableDef::serialize(utility::Serializer& serializer) const {
   serializer.writeProperty(1, name);
   serializer.writeProperty(2, columns);
   serializer.writeProperty(3, primaryKey);
}

lingodb::catalog::CreateTableDef lingodb::catalog::CreateTableDef::deserialize(utility::Deserializer& deserializer) {
   auto name = deserializer.readProperty<std::string>(1);
   auto columns = deserializer.readProperty<std::vector<Column>>(2);
   auto primaryKey = deserializer.readProperty<std::vector<std::string>>(3);
   return CreateTableDef{name, columns, primaryKey};
}
