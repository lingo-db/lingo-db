
#ifndef MLIR_GOES_RELATIONAL_SORTSPECIFICATIONATTR_H
#define MLIR_GOES_RELATIONAL_SORTSPECIFICATIONATTR_H
#include "mlir/Dialect/RelAlg/IR/SortSpecificationAttrStorage.h"

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace relalg {

class SortSpecificationAttr
    : public Attribute::AttrBase<SortSpecificationAttr, Attribute,
       SortSpecificationAttrStorage> {
public:
  using Base::Base;
  static SortSpecificationAttr
  get(MLIRContext *context,RelationalAttributeRefAttr attr, SortSpec sortSpec) {
    return Base::get(context, attr,sortSpec);
  }
   RelationalAttributeRefAttr getAttr(){
      return getImpl()->attr;
   }
   SortSpec getSortSpec(){
      return getImpl()->sortSpec;
   }
};
} // namespace relalg
} // namespace mlir

#endif // MLIR_GOES_RELATIONAL_SORTSPECIFICATIONATTR_H
