#pragma once

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>

#include "lingodb/compiler/Dialect/graphalg/GraphAlgEnumAttr.h.inc"

#define GET_ATTRDEF_CLASSES
#include "lingodb/compiler/Dialect/graphalg/GraphAlgAttr.h.inc"

namespace graphalg {

bool binaryOpIsCompare(BinaryOp op);

}
