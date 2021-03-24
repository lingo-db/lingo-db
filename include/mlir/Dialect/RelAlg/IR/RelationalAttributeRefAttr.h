//
// Created by michael on 13.03.21.
//

#ifndef MLIR_GOES_RELATIONAL_RELATIONALATTRIBUTEREFATTR_H
#define MLIR_GOES_RELATIONAL_RELATIONALATTRIBUTEREFATTR_H
#include "mlir/Dialect/RelAlg/IR/RelationalAttribute.h"
#include "mlir/Dialect/RelAlg/IR/RelationalAttributeAttrRefStorage.h"

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace relalg {

class RelationalAttributeRefAttr
    : public Attribute::AttrBase<RelationalAttributeRefAttr, Attribute,
                                 RelationalAttributeAttrRefStorage> {
public:
  using Base::Base;
  static RelationalAttributeRefAttr
  get(MLIRContext *context,SymbolRefAttr attributeName,
      std::shared_ptr<RelationalAttribute> relationalAttribute) {
    return Base::get(context, attributeName,relationalAttribute);
  }
  const SymbolRefAttr getName() {
    return getImpl()->name;
  }
  RelationalAttribute &getRelationalAttribute() {
    return *getImpl()->relationalAttribute;
  }
};
} // namespace relalg
} // namespace mlir

#endif // MLIR_GOES_RELATIONAL_RELATIONALATTRIBUTEREFATTR_H
