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
      case CatalogEntryType::HIPY_FUNCTION_ENTRY:
         return HiPyFunctionCatalogEntry::deserialize(deserializer);
      case CatalogEntryType::PYTHON_FUNCTION_ENTRY:
         return PythonFunctionCatalogEntry::deserialize(deserializer);
      default:
         throw std::runtime_error("Should not happen");
   }
}

std::shared_ptr<FunctionCatalogEntry> CFunctionCatalogEntry::deserialize(lingodb::utility::Deserializer& deserializer) {
   auto name = deserializer.readProperty<std::string>(2);
   auto code = deserializer.readProperty<std::string>(3);
   auto returnType = deserializer.readProperty<Type>(4);
   auto argumentTypes = deserializer.readProperty<std::vector<Type>>(5);
   return std::make_shared<CFunctionCatalogEntry>(name, code, returnType, argumentTypes);
}
std::shared_ptr<FunctionCatalogEntry> HiPyFunctionCatalogEntry::deserialize(lingodb::utility::Deserializer& deserializer) {
   auto name = deserializer.readProperty<std::string>(2);
   auto code = deserializer.readProperty<std::string>(3);
   auto returnType = deserializer.readProperty<Type>(4);
   auto argumentTypes = deserializer.readProperty<std::vector<Type>>(5);
   auto byteCode = deserializer.readProperty<std::string>(6);
   return std::make_shared<HiPyFunctionCatalogEntry>(name, code, returnType, argumentTypes, byteCode);
}
std::shared_ptr<FunctionCatalogEntry> PythonFunctionCatalogEntry::deserialize(lingodb::utility::Deserializer& deserializer){
   auto name = deserializer.readProperty<std::string>(2);
   auto code = deserializer.readProperty<std::string>(3);
   auto returnType = deserializer.readProperty<Type>(4);
   auto argumentTypes = deserializer.readProperty<std::vector<Type>>(5);
   return std::make_shared<CFunctionCatalogEntry>(name, code, returnType, argumentTypes);
}


void HiPyFunctionCatalogEntry::serializeEntry(lingodb::utility::Serializer& serializer) const {
   serializer.writeProperty(1, entryType);
   serializer.writeProperty(2, name);
   serializer.writeProperty(3, code);
   serializer.writeProperty(4, returnType);
   serializer.writeProperty(5, argumentTypes);
   serializer.writeProperty(6, byteCode);
}

void visitUDFFunctions(const std::function<void(std::string, void*)>& fn) {
   auto f = lingodb::catalog::FunctionCatalogEntry::getUdfFunctions();
   for (auto udf : f) {
      fn(udf.first, udf.second.addrPtr);
   }
}
} // namespace lingodb::catalog