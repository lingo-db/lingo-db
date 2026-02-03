#ifndef LINGODB_RUNTIME_SIP_H
#define LINGODB_RUNTIME_SIP_H
#include "LazyJoinHashtable.h"
#include "lingodb/runtime/ExecutionContext.h"
#include "lingodb/runtime/helpers.h"
#include <shared_mutex>
namespace lingodb::runtime {
class SIP {

   public:
   struct SIPNode {
      lingodb::runtime::HashIndexedView* hashView;
      SIPNode* next;
      std::atomic<int8_t> skipState{0};
      uint8_t id;

#if DEBUG
      std::atomic<size_t> filteredCount;
      std::atomic<size_t> completeCount;
#endif
   };
   //virtual void iterate(bool parallel, std::vector<std::string> members, const std::function<void(BatchView*)>& cb) = 0;
   virtual ~SIP() {}
   static std::atomic<SIPNode*> sips;
   static std::unordered_map<std::string, lingodb::runtime::HashIndexedView*> filters;
   static std::shared_mutex filtersMutex;
   static lingodb::runtime::HashIndexedView* getFilter(const uint8_t id);
   static SIPNode* getFilterNode(const uint8_t id);
   static lingodb::runtime::HashIndexedView* createSIP(lingodb::runtime::HashIndexedView* hashView, uint8_t id);
   //static DataSource* getFromTable(ArrowTable* arrowTable, runtime::VarLen32 mappingVal,runtime::VarLen32 columnArray);
};

} // namespace lingodb::runtime

#endif // LINGODB_RUNTIME_SIP_H
