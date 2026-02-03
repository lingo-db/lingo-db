#include "lingodb/runtime/SIP.h"

#include "lingodb/compiler/Dialect/SubOperator/MemberManager.h"

#include <functional>
#include <string>

namespace lingodb::runtime {
std::unordered_map<std::string, lingodb::runtime::HashIndexedView*> SIP::filters{};
std::atomic<SIP::SIPNode*> SIP::sips{nullptr};
std::shared_mutex SIP::filtersMutex{};
lingodb::runtime::HashIndexedView* SIP::createSIP(lingodb::runtime::HashIndexedView* hash, uint8_t id) {
   // Create a hash based on the DataSource pointer so we get a stable identifier
   // for this DataSource instance. Use uintptr_t to avoid narrowing.
   SIPNode* newSIP = new SIPNode();
   newSIP->hashView = hash;
   newSIP->id = id;
   SIPNode* oldHead = sips.load();
   do {
      newSIP->next = oldHead;
   } while (!sips.compare_exchange_weak(oldHead, newSIP));

   return hash;
}

lingodb::runtime::HashIndexedView* SIP::getFilter(const uint8_t id) {
   auto current = sips.load(std::memory_order_relaxed);
   while (current->id != id && current->next != nullptr) {
      current = current->next;
   }
   assert(current);
   return current->hashView;

}
} // namespace lingodb::runtime