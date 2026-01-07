#ifndef LINGODB_RUNTIME_EXTERNALDATASOURCEPROPERTY_H
#define LINGODB_RUNTIME_EXTERNALDATASOURCEPROPERTY_H
#include "lingodb/utility/Serialization.h"
#include "storage/TableStorage.h"
namespace lingodb::runtime {
struct ExternalDatasourceProperty {
   struct Mapping {
      std::string memberName;
      std::string identifier;
      void serialize(lingodb::utility::Serializer& serializer) const {
         serializer.writeProperty(0, memberName);
         serializer.writeProperty(1, identifier);
      }
      bool operator==(const Mapping& other) const {
         return other.memberName == memberName && other.identifier == identifier;
      }
      static Mapping deserialize(lingodb::utility::Deserializer& deserializer) {
         Mapping map{};
         map.memberName = deserializer.readProperty<std::string>(0);
         map.identifier = deserializer.readProperty<std::string>(1);
         return map;
      }
   };
   std::string tableName;
   std::vector<Mapping> mapping;
   std::vector<runtime::FilterDescription> filterDescriptions{};
   std::string index;
   std::string indexType;

   void serialize(lingodb::utility::Serializer& serializer) const {
      serializer.writeProperty(0, tableName);
      serializer.writeProperty(1, mapping);
      serializer.writeProperty(2, filterDescriptions);
      serializer.writeProperty(3, index);
      serializer.writeProperty(4, indexType);
   }
   bool operator==(const ExternalDatasourceProperty& other) const {
      return other.index == index && other.indexType == indexType && other.mapping == mapping && other.tableName == tableName &&
         other.filterDescriptions == filterDescriptions;
   }

   static ExternalDatasourceProperty deserialize(lingodb::utility::Deserializer& deserializer) {
      ExternalDatasourceProperty prop{};
      prop.tableName = deserializer.readProperty<std::string>(0);
      prop.mapping = deserializer.readProperty<std::vector<Mapping>>(1);
      prop.filterDescriptions = deserializer.readProperty<std::vector<runtime::FilterDescription>>(2);
      prop.index = deserializer.readProperty<std::string>(3);
      prop.indexType = deserializer.readProperty<std::string>(4);

      return prop;
   }
};
} // namespace lingodb::runtime

#endif // LINGODB_RUNTIME_EXTERNALDATASOURCEPROPERTY_H
