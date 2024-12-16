#include "lingodb/compiler/Dialect/TupleStream/ColumnManager.h"
#include <iostream>
namespace lingodb::compiler::dialect::tuples {
using namespace mlir;
void ColumnManager::setContext(MLIRContext* context) {
   this->context = context;
}
std::shared_ptr<Column> ColumnManager::get(StringRef scope, StringRef attribute) {
   if (!scopeUnifier.contains(std::string(scope))) {
      scopeUnifier[std::string(scope)] = 0;
   }
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
   return ColumnDefAttr::get(context, name, attribute, fromExisting);
}
ColumnDefAttr ColumnManager::createDef(StringRef scope, StringRef name, Attribute fromExisting) {
   auto attribute = get(scope, name);
   std::vector<FlatSymbolRefAttr> nested;
   nested.push_back(FlatSymbolRefAttr::get(context, name));
   return ColumnDefAttr::get(context, SymbolRefAttr::get(context, scope, nested), attribute, fromExisting);
}
ColumnRefAttr ColumnManager::createRef(SymbolRefAttr name) {
   assert(name.getNestedReferences().size() == 1);
   auto attribute = get(name.getRootReference().getValue(), name.getLeafReference().getValue());
   return tuples::ColumnRefAttr::get(context, name, attribute);
}
ColumnRefAttr ColumnManager::createRef(StringRef scope, StringRef name) {
   auto attribute = get(scope, name);
   std::vector<FlatSymbolRefAttr> nested;
   nested.push_back(FlatSymbolRefAttr::get(context, name));
   return tuples::ColumnRefAttr::get(context, SymbolRefAttr::get(context, scope, nested), attribute);
}
ColumnRefAttr ColumnManager::createRef(const Column* attr) {
   auto [scope, name] = attributesRev[attr];
   return createRef(scope, name);
}
ColumnDefAttr ColumnManager::createDef(const Column* attr, Attribute fromExisting) {
   auto [scope, name] = attributesRev[attr];
   return createDef(scope, name, fromExisting);
}

std::pair<std::string, std::string> ColumnManager::getName(const Column* attr) {
   return attributesRev.at(attr);
}
} // namespace lingodb::compiler::dialect::tuples
