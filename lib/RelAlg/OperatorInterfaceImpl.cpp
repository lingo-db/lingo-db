#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
using namespace mlir::relalg;
std::vector<RelationalAttributeDefAttr> mlir::relalg::detail::getCreations(mlir::Operation* op) {
   std::vector<RelationalAttributeDefAttr> creations;
   op->walk([&](AddAttrOp attrOp) {
      creations.push_back(attrOp.attr());
   });
   return creations;
}

std::vector<RelationalAttributeRefAttr> mlir::relalg::detail::getUsages(mlir::Operation* op) {
   std::vector<RelationalAttributeRefAttr> usages;
   op->walk([&](GetAttrOp attrOp){
     usages.push_back(attrOp.attr());
   });
   return usages;
}
std::vector<RelationalAttributeRefAttr> AggregationOp::getAttributeUsage() {
   std::vector<RelationalAttributeRefAttr> usages;
   for(Attribute a:group_by_attrs()){
      usages.push_back(a.dyn_cast_or_null<RelationalAttributeRefAttr>());
   }
   auto real_usages=detail::getUsages(getOperation());
   usages.insert(usages.end(),real_usages.begin(),real_usages.end());
   return usages;
}

std::vector<RelationalAttributeDefAttr> BaseTableOp::getCreatedAttributes(){
   std::vector<RelationalAttributeDefAttr> creations;
   for (auto mapping : columns()) {
      auto [_, attr] = mapping;
      auto relation_def_attr = attr.dyn_cast_or_null<RelationalAttributeDefAttr>();
      creations.push_back(relation_def_attr);
   }
   return creations;
}
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.cpp.inc"
