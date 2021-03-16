#ifndef RelAlg_RelAlgOPS_H
#define RelAlg_RelAlgOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/DB/IR/DBType.h"
#include "mlir/Dialect/DB/IR/DBTypes.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOpsEnums.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgTypes.h"
#include "mlir/Dialect/RelAlg/IR/RelationalAttribute.h"
#include "mlir/Dialect/RelAlg/IR/RelationalAttributeDefAttr.h"
#include "mlir/Dialect/RelAlg/IR/RelationalAttributeRefAttr.h"


#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h.inc"

#endif// RelAlg_RelAlgOPS_H
