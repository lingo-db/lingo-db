#pragma once

#include <mlir/IR/Types.h>

#include "lingodb/compiler/Dialect/garel/GARelAttr.h"

#define GET_TYPEDEF_CLASSES
#include "lingodb/compiler/Dialect/garel/GARelOpsTypes.h.inc"

namespace garel {

bool isColumnType(mlir::Type t);

} // namespace garel
