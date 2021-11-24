#ifndef MLIR_DIALECT_RELALG_IR_RELATIONALATTRIBUTEMANAGER_H
#define MLIR_DIALECT_RELALG_IR_RELATIONALATTRIBUTEMANAGER_H
#include "mlir/Dialect/RelAlg/IR/RelationalAttribute.h"
#include "mlir/Dialect/RelAlg/IR/RelationalAttributeDefAttr.h"
#include "mlir/Dialect/RelAlg/IR/RelationalAttributeRefAttr.h"

namespace mlir::relalg {
class RelationalAttributeManager {
   public:
   void setContext(MLIRContext* context);
   std::shared_ptr<RelationalAttribute> get(StringRef scope, StringRef attribute);
   RelationalAttributeDefAttr createDef(SymbolRefAttr name, Attribute fromExisting = Attribute());
   RelationalAttributeDefAttr createDef(StringRef name, Attribute fromExisting = Attribute());

   RelationalAttributeRefAttr createRef(SymbolRefAttr name);
   RelationalAttributeRefAttr createRef(const RelationalAttribute* attr);

   RelationalAttributeRefAttr createRef(StringRef scope, StringRef name);
   std::pair<std::string,std::string> getName(const RelationalAttribute* attr);

   void setCurrentScope(StringRef scope) {
      currentScope = scope;
      if (!scopeUnifier.count(std::string(scope))) {
         scopeUnifier[std::string(scope)] = 0;
      }
   }
   std::string getUniqueScope(StringRef base) {
      if (scopeUnifier.count(std::string(base))) {
         scopeUnifier[std::string(base)] += 1;
         return std::string(base) + std::to_string(scopeUnifier[std::string(base)]);
      } else {
         scopeUnifier[std::string(base)] = 0;
         return std::string(base);
      }
   }

   private:
   MLIRContext* context;
   std::string currentScope;
   struct HashPair {
      template <class T1, class T2>
      size_t operator()(const std::pair<T1, T2>& p) const {
         auto hash1 = std::hash<T1>{}(p.first);
         auto hash2 = std::hash<T2>{}(p.second);
         return hash1 ^ hash2;
      }
   };
   std::unordered_map<std::pair<std::string, std::string>, std::shared_ptr<RelationalAttribute>, HashPair> attributes;
   std::unordered_map<const RelationalAttribute*, std::pair<std::string, std::string>> attributesRev;

   std::unordered_map<std::string, size_t> scopeUnifier;
};
} // namespace mlir::relalg

#endif // MLIR_DIALECT_RELALG_IR_RELATIONALATTRIBUTEMANAGER_H
