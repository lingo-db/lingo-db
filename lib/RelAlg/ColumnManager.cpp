#include "mlir/Dialect/RelAlg/IR/ColumnManager.h"
namespace mlir::relalg {
void ColumnManager::setContext(MLIRContext* context) {
   this->context = context;
}
std::shared_ptr<Column> ColumnManager::get(StringRef scope, StringRef attribute) {
   auto pair = std::make_pair(std::string(scope), std::string(attribute));
   if (!attributes.count(pair)) {
      auto attr = std::make_shared<Column>();
      attributes[pair] = attr;
      attributesRev[attr.get()] = pair;
   }
   return attributes[pair];
}
ColumnDefAttr ColumnManager::createDef(SymbolRefAttr name, Attribute fromExisting) {
   assert(name.getNestedReferences().size() == 1);
   auto attribute = get(name.getRootReference().getValue(), name.getLeafReference().getValue());
   return mlir::relalg::ColumnDefAttr::get(context, name, attribute, fromExisting);
}
ColumnDefAttr ColumnManager::createDef(StringRef scope, StringRef name, Attribute fromExisting) {
   auto attribute = get(scope, name);
   std::vector<FlatSymbolRefAttr> nested;
   nested.push_back(FlatSymbolRefAttr::get(context, name));
   return mlir::relalg::ColumnDefAttr::get(context, SymbolRefAttr::get(context, scope, nested), attribute, fromExisting);
}
ColumnRefAttr ColumnManager::createRef(SymbolRefAttr name) {
   assert(name.getNestedReferences().size() == 1);
   auto attribute = get(name.getRootReference().getValue(), name.getLeafReference().getValue());
   return relalg::ColumnRefAttr::get(context, name, attribute);
}
ColumnRefAttr ColumnManager::createRef(StringRef scope, StringRef name) {
   auto attribute = get(scope, name);
   std::vector<FlatSymbolRefAttr> nested;
   nested.push_back(FlatSymbolRefAttr::get(context, name));
   return relalg::ColumnRefAttr::get(context, SymbolRefAttr::get(context, scope, nested), attribute);
}
ColumnRefAttr ColumnManager::createRef(const Column* attr) {
   auto [scope, name] = attributesRev[attr];
   return createRef(scope, name);
}
ColumnDefAttr ColumnManager::createDef(const Column* attr) {
   auto [scope, name] = attributesRev[attr];
   return createDef(scope, name);
}

std::pair<std::string, std::string> ColumnManager::getName(const Column* attr) {
   return attributesRev.at(attr);
}
} // namespace mlir::relalg
