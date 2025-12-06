#ifndef LINGODB_SIP_H
#define LINGODB_SIP_H
#include "ArrowTable.h"
#include "ArrowView.h"
#include "LazyJoinHashtable.h"
#include "lingodb/runtime/ExecutionContext.h"
#include "lingodb/runtime/helpers.h"
#include <cstdint>
#include <shared_mutex>
#include <lingodb/runtime/DataSourceIteration.h>
namespace  lingodb::runtime {
class SIP {
   public:
   //virtual void iterate(bool parallel, std::vector<std::string> members, const std::function<void(BatchView*)>& cb) = 0;
   virtual ~SIP() {}
   static std::unordered_map<std::string,  lingodb::runtime::HashIndexedView*> filters;
   static std::shared_mutex filtersMutex;
   static lingodb::runtime::HashIndexedView* getFilter(const std::string& sipName);
   static VarLen32 createSIP(lingodb::runtime::HashIndexedView* hashView, VarLen32  sipName);
   //static DataSource* getFromTable(ArrowTable* arrowTable, runtime::VarLen32 mappingVal,runtime::VarLen32 columnArray);
};

} // namespace lingodb::runtime

#endif //LINGODB_SIP_H
