#include "mlir/Dialect/RelAlg/IR/RelationalAttributeManager.h"
namespace mlir::relalg {
void RelationalAttributeManager::setContext(MLIRContext* context) {
   this->context = context;
}
std::shared_ptr<RelationalAttribute> RelationalAttributeManager::get(StringRef scope, StringRef attribute) {
   auto pair = std::make_pair(std::string(scope), std::string(attribute));
   if (!attributes.count(pair)) {
      auto attr=std::make_shared<RelationalAttribute>();
      attributes[pair] = attr;
      attributes_rev[attr.get()]=pair;
   }
   return attributes[pair];
}
RelationalAttributeDefAttr RelationalAttributeManager::createDef(SymbolRefAttr name, Attribute fromExisting) {
   auto attribute = get(current_scope, name.getRootReference());
   return mlir::relalg::RelationalAttributeDefAttr::get(context, name.getRootReference(), attribute, fromExisting);
}
RelationalAttributeDefAttr RelationalAttributeManager::createDef(StringRef name, Attribute fromExisting) {
   auto attribute = get(current_scope, name);
   return mlir::relalg::RelationalAttributeDefAttr::get(context, name, attribute, fromExisting);
}
RelationalAttributeRefAttr RelationalAttributeManager::createRef(SymbolRefAttr name) {
   assert(name.getNestedReferences().size() == 1);
   auto attribute = get(name.getRootReference(), name.getLeafReference());
   return relalg::RelationalAttributeRefAttr::get(context, name, attribute);
}
RelationalAttributeRefAttr RelationalAttributeManager::createRef(StringRef scope, StringRef name) {
   auto attribute = get(scope, name);
   std::vector<FlatSymbolRefAttr> nested;
   nested.push_back(FlatSymbolRefAttr::get(context, name));
   return relalg::RelationalAttributeRefAttr::get(context, SymbolRefAttr::get(context, scope, nested), attribute);
}
RelationalAttributeRefAttr RelationalAttributeManager::createRef(RelationalAttribute* attr){
   auto [scope,name]=attributes_rev[attr];
   return createRef(scope,name);
}

}