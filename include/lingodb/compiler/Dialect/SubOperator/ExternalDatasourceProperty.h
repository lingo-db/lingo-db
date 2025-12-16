#ifndef LINGODB_EXTERNALDATASOURCEPROPERTY_H
#define LINGODB_EXTERNALDATASOURCEPROPERTY_H
#include "lingodb/utility/Serialization.h"
#include <llvm/ADT/Hashing.h>

namespace lingodb {
struct ExternalDatasourceProperty {
   std::string restrictions;

   inline bool operator==(const ExternalDatasourceProperty& other) const noexcept {
      return restrictions == other.restrictions;
   }

   void serialize(lingodb::utility::Serializer& serializer) {
   }

   inline llvm::hash_code hash() const {
      return llvm::hash_combine(restrictions);
   }
};
} // namespace lingodb
namespace llvm {
inline hash_code hash_value(const lingodb::ExternalDatasourceProperty& datasource) {
   return datasource.hash();
}
} // namespace llvm

#endif //LINGODB_EXTERNALDATASOURCEPROPERTY_H
