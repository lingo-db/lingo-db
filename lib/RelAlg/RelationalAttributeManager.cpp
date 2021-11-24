#include "mlir/Dialect/RelAlg/IR/RelationalAttributeManager.h"
namespace mlir::relalg {
void RelationalAttributeManager::setContext(MLIRContext* context) {
   this->context = context;
}
std::shared_ptr<RelationalAttribute> RelationalAttributeManager::get(StringRef scope, StringRef attribute) {
   auto pair = std::make_pair(std::string(scope), std::string(attribute));
   if (!attributes.count(pair)) {
      auto attr = std::make_shared<RelationalAttribute>();
      attributes[pair] = attr;
      attributesRev[attr.get()]=pair;
   }
   return attributes[pair];
}
RelationalAttributeDefAttr RelationalAttributeManager::createDef(SymbolRefAttr name, Attribute fromExisting) {
   auto attribute = get(currentScope, name.getRootReference().getValue());
   return mlir::relalg::RelationalAttributeDefAttr::get(context, name.getRootReference().getValue(), attribute, fromExisting);
}
RelationalAttributeDefAttr RelationalAttributeManager::createDef(StringRef name, Attribute fromExisting) {
   auto attribute = get(currentScope, name);
   return mlir::relalg::RelationalAttributeDefAttr::get(context, name, attribute, fromExisting);
}
RelationalAttributeRefAttr RelationalAttributeManager::createRef(SymbolRefAttr name) {
   assert(name.getNestedReferences().size() == 1);
   auto attribute = get(name.getRootReference().getValue(), name.getLeafReference().getValue());
   return relalg::RelationalAttributeRefAttr::get(context, name, attribute);
}
RelationalAttributeRefAttr RelationalAttributeManager::createRef(StringRef scope, StringRef name) {
   auto attribute = get(scope, name);
   std::vector<FlatSymbolRefAttr> nested;
   nested.push_back(FlatSymbolRefAttr::get(context, name));
   return relalg::RelationalAttributeRefAttr::get(context, SymbolRefAttr::get(context, scope, nested), attribute);
}
RelationalAttributeRefAttr RelationalAttributeManager::createRef(const RelationalAttribute* attr){
   auto [scope,name]=attributesRev[attr];
   return createRef(scope,name);
}

std::pair<std::string, std::string> RelationalAttributeManager::getName(const RelationalAttribute* attr) {
   return attributesRev[attr];
}
} // namespace mlir::relalg
