#ifndef DB_DIALECTS_RELATIONALATTRIBUTEMANAGER_H
#define DB_DIALECTS_RELATIONALATTRIBUTEMANAGER_H
#include "mlir/Dialect/RelAlg/IR/RelationalAttribute.h"
#include "mlir/Dialect/RelAlg/IR/RelationalAttributeDefAttr.h"
#include "mlir/Dialect/RelAlg/IR/RelationalAttributeRefAttr.h"

namespace mlir::relalg {
class RelationalAttributeManager {
   public:


   void setContext(MLIRContext* context);
   std::shared_ptr<RelationalAttribute> get(StringRef scope,StringRef attribute);
   RelationalAttributeDefAttr createDef(SymbolRefAttr name,Attribute fromExisting=Attribute());
   RelationalAttributeRefAttr createRef(SymbolRefAttr name);
   void setCurrentScope(StringRef scope){
      current_scope=scope;
   }
   private:
   MLIRContext* context;
   std::string current_scope;
   struct hash_pair {
      template<class T1, class T2>
      size_t operator()(const std::pair<T1, T2> &p) const {
         auto hash1 = std::hash<T1>{}(p.first);
         auto hash2 = std::hash<T2>{}(p.second);
         return hash1 ^ hash2;
      }
   };
   std::unordered_map<std::pair<std::string,std::string>, std::shared_ptr<RelationalAttribute>,hash_pair> attributes;

};
}

#endif //DB_DIALECTS_RELATIONALATTRIBUTEMANAGER_H
