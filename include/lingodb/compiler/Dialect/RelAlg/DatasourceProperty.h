#ifndef LINGODB_COMPILER_DIALECT_RELALG_DATASOURCEPROPERTY_H
#define LINGODB_COMPILER_DIALECT_RELALG_DATASOURCEPROPERTY_H
#include "lingodb/runtime/storage/TableStorage.h"
#include "lingodb/utility/Serialization.h"
#include <llvm/ADT/Hashing.h>

namespace lingodb {
struct DatasourceProperty {
   std::vector<runtime::FilterDescription> filterDescription;

   inline bool operator==(const DatasourceProperty& other) const noexcept {
      return filterDescription == other.filterDescription;
   }

   void serialize(lingodb::utility::Serializer& serializer) {
      serializer.writeProperty(0, filterDescription);
   }

   static DatasourceProperty deserialize(lingodb::utility::Deserializer& deserializer) {
      std::vector<runtime::FilterDescription> filters = deserializer.readProperty<std::vector<runtime::FilterDescription>>(0);
      return DatasourceProperty{.filterDescription = filters};
   }

   inline llvm::hash_code hash() const {
      DatasourceProperty tmp = *this;
      lingodb::utility::SimpleByteWriter writer{};
      lingodb::utility::Serializer s{writer};
      tmp.serialize(s);

      const std::byte* data = writer.data();
      size_t len = writer.size();

      std::string bytes;
      bytes.reserve(len);
      for (size_t i = 0; i < len; ++i) {
         bytes.push_back(static_cast<char>(std::to_integer<unsigned char>(data[i])));
      }

      return llvm::hash_value(bytes);
   }
};
} // namespace lingodb
#endif // LINGODB_COMPILER_DIALECT_RELALG_DATASOURCEPROPERTY_H
