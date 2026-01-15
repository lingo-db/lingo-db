#ifndef LINGODB_RUNTIME_DATASOURCERESTRICTIONPROPERTY_H
#define LINGODB_RUNTIME_DATASOURCERESTRICTIONPROPERTY_H
#include "lingodb/runtime/storage/TableStorage.h"
#include "lingodb/utility/Serialization.h"
#include <llvm/ADT/Hashing.h>

namespace lingodb::runtime {
struct DatasourceRestrictionProperty {
   std::vector<runtime::FilterDescription> filterDescription;

   inline bool operator==(const DatasourceRestrictionProperty& other) const noexcept {
      return filterDescription == other.filterDescription;
   }

   void serialize(lingodb::utility::Serializer& serializer) const {
      serializer.writeProperty(0, filterDescription);
   }

   static DatasourceRestrictionProperty deserialize(lingodb::utility::Deserializer& deserializer) {
      std::vector<runtime::FilterDescription> filters = deserializer.readProperty<std::vector<runtime::FilterDescription>>(0);
      return DatasourceRestrictionProperty{.filterDescription = filters};
   }

   inline llvm::hash_code hash() const {
      return llvm::hash_value(utility::serializeToHexString(this));
   }
};
} // namespace lingodb::runtime
#endif // LINGODB_RUNTIME_DATASOURCERESTRICTIONPROPERTY_H
