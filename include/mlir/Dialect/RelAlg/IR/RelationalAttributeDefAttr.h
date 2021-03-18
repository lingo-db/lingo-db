//
// Created by michael on 13.03.21.
//

#ifndef MLIR_GOES_RELATIONAL_RELATIONALATTRIBUTEDEFATTR_H
#define MLIR_GOES_RELATIONAL_RELATIONALATTRIBUTEDEFATTR_H
#include "mlir/Dialect/RelAlg/IR/RelationalAttribute.h"
#include "mlir/Dialect/RelAlg/IR/RelationalAttributeAttrDefStorage.h"

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace relalg {

class RelationalAttributeDefAttr
    : public Attribute::AttrBase<RelationalAttributeDefAttr, Attribute,
                                 RelationalAttributeAttrDefStorage> {
public:
  using Base::Base;
  static RelationalAttributeDefAttr
  get(MLIRContext *context,StringRef attributeName,
      std::shared_ptr<RelationalAttribute> relationalAttribute,Attribute fromExisting) {
    return Base::get(context, attributeName,relationalAttribute,fromExisting);
  }
  const StringRef getName() {
    return getImpl()->name;
  }
  const RelationalAttribute &getRelationalAttribute() {
    return *getImpl()->relationalAttribute;
  }
  const Attribute getFromExisting(){
     return getImpl()->fromExisting;
  }
};
} // namespace relalg
} // namespace mlir

#endif // MLIR_GOES_RELATIONAL_RELATIONALATTRIBUTEDEFATTR_H
