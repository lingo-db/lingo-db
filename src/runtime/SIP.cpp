#include "lingodb/runtime/SIP.h"

#include "lingodb/compiler/Dialect/SubOperator/MemberManager.h"

#include <functional>
#include <string>

namespace lingodb::runtime {
std::unordered_map<std::string, lingodb::runtime::HashIndexedView*> SIP::filters{};
std::shared_mutex SIP::filtersMutex{};
VarLen32 SIP::createSIP(lingodb::runtime::HashIndexedView* hash, VarLen32 sipName) {
   // Create a hash based on the DataSource pointer so we get a stable identifier
   // for this DataSource instance. Use uintptr_t to avoid narrowing.


   {
      std::unique_lock lock(filtersMutex);
      filters.emplace(sipName.str(), hash);
   }
   //std::cerr << "Creating SIP: " << sipName.str()  << " with: " << hash << std::endl;

   return VarLen32::fromString(sipName.str());
}

lingodb::runtime::HashIndexedView* SIP::getFilter(const std::string& sipName) {
   std::shared_lock lock(filtersMutex);
   auto it = filters.find(sipName);
   if (it == filters.end()) return nullptr;
   return it->second;
}
} // namespace lingodb::runtime