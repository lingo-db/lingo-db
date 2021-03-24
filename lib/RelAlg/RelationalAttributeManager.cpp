#include "mlir/Dialect/RelAlg/IR/RelationalAttributeManager.h"
namespace mlir::relalg {
void RelationalAttributeManager::setContext(MLIRContext* context) {
   this->context = context;
}
std::shared_ptr<RelationalAttribute> RelationalAttributeManager::get(StringRef scope, StringRef attribute) {
   auto pair = std::make_pair(std::string(scope), std::string(attribute));
   if (!attributes.count(pair)) {
      attributes[pair] = std::make_shared<RelationalAttribute>();
   }
   return attributes[pair];
}
RelationalAttributeDefAttr RelationalAttributeManager::createDef(SymbolRefAttr name, Attribute fromExisting) {
   auto attribute = get(current_scope, name.getRootReference());
   return mlir::relalg::RelationalAttributeDefAttr::get(context, name.getRootReference(), attribute, fromExisting);
}
RelationalAttributeRefAttr RelationalAttributeManager::createRef(SymbolRefAttr name) {
   assert(name.getNestedReferences().size() == 1);
   auto attribute = get(name.getRootReference(), name.getLeafReference());
   return relalg::RelationalAttributeRefAttr::get(context, name, attribute);
}
}