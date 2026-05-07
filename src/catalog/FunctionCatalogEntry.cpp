#include "lingodb/catalog/FunctionCatalogEntry.h"

#include "lingodb/utility/Serialization.h"

#include "lingodb/catalog/TableCatalogEntry.h"

#include <filesystem>

namespace lingodb::catalog {

void FunctionCatalogEntry::serializeEntry(lingodb::utility::Serializer& serializer) const {
   serializer.writeProperty(1, entryType);
   serializer.writeProperty(2, name);
   serializer.writeProperty(3, code);
   serializer.writeProperty(4, returnType);
   serializer.writeProperty(5, argumentTypes);
}
std::shared_ptr<FunctionCatalogEntry> FunctionCatalogEntry::deserialize(lingodb::utility::Deserializer& deserializer) {
   auto entryType = deserializer.readProperty<CatalogEntryType>(1);
   switch (entryType) {
      case CatalogEntryType::C_FUNCTION_ENTRY:
         return CFunctionCatalogEntry::deserialize(deserializer);
      case CatalogEntryType::PYTHON_FUNCTION_ENTRY:
         return PythonFunctionCatalogEntry::deserialize(deserializer);
      default:
         throw std::runtime_error("FunctionCatalogEntry::deserialize: not a scalar function entry");
   }
}

std::shared_ptr<FunctionCatalogEntry> CFunctionCatalogEntry::deserialize(lingodb::utility::Deserializer& deserializer) {
   auto name = deserializer.readProperty<std::string>(2);
   auto code = deserializer.readProperty<std::string>(3);
   auto returnType = deserializer.readProperty<Type>(4);
   auto argumentTypes = deserializer.readProperty<std::vector<Type>>(5);
   return std::make_shared<CFunctionCatalogEntry>(name, code, returnType, argumentTypes);
}

std::shared_ptr<FunctionCatalogEntry> PythonFunctionCatalogEntry::deserialize(lingodb::utility::Deserializer& deserializer) {
   auto name = deserializer.readProperty<std::string>(2);
   auto code = deserializer.readProperty<std::string>(3);
   auto returnType = deserializer.readProperty<Type>(4);
   auto argumentTypes = deserializer.readProperty<std::vector<Type>>(5);
   return std::make_shared<PythonFunctionCatalogEntry>(name, code, returnType, argumentTypes);
}

void TableFunctionCatalogEntry::serializeEntry(lingodb::utility::Serializer& serializer) const {
   serializer.writeProperty(1, entryType);
   serializer.writeProperty(2, name);
   serializer.writeProperty(3, language);
   serializer.writeProperty(4, code);
   serializer.writeProperty(5, inputTableName);
   serializer.writeProperty(6, inputColumns);
   serializer.writeProperty(7, argumentTypes);
   serializer.writeProperty(8, returnColumns);
}

std::shared_ptr<TableFunctionCatalogEntry> TableFunctionCatalogEntry::deserialize(lingodb::utility::Deserializer& deserializer) {
   auto name = deserializer.readProperty<std::string>(2);
   auto language = deserializer.readProperty<std::string>(3);
   auto code = deserializer.readProperty<std::string>(4);
   auto inputTableName = deserializer.readProperty<std::string>(5);
   auto inputColumns = deserializer.readProperty<std::vector<std::pair<std::string, Type>>>(6);
   auto argumentTypes = deserializer.readProperty<std::vector<Type>>(7);
   auto returnColumns = deserializer.readProperty<std::vector<std::pair<std::string, Type>>>(8);
   return std::make_shared<TableFunctionCatalogEntry>(name, language, code, inputTableName, inputColumns, argumentTypes, returnColumns);
}

void visitUDFFunctions(const std::function<void(std::string, void*)>& fn) {
   auto f = lingodb::catalog::FunctionCatalogEntry::getUdfFunctions();
   for (auto udf : f) {
      fn(udf.first, udf.second.addrPtr);
   }
}
} // namespace lingodb::catalog
