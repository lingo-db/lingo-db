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

void lingodb::catalog::CreateFunctionDef::serialize(utility::Serializer& serializer) const {
   serializer.writeProperty(1, name);
   serializer.writeProperty(2, language);
   serializer.writeProperty(3, code);
   serializer.writeProperty(4, returnType);
   serializer.writeProperty(5, argumentTypes);
}
lingodb::catalog::CreateFunctionDef lingodb::catalog::CreateFunctionDef::deserialize(utility::Deserializer& deserializer) {
   auto name = deserializer.readProperty<std::string>(1);
   auto language = deserializer.readProperty<std::string>(2);
   auto code = deserializer.readProperty<std::string>(3);
   auto returnType = deserializer.readProperty<Type>(4);
   auto argumentTypes = deserializer.readProperty<std::vector<Type>>(5);

   return CreateFunctionDef{name, language, code, returnType, argumentTypes};
}
