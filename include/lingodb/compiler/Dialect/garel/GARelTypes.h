#pragma once

#include <mlir/IR/Types.h>

#include "garel/GARelAttr.h"

#define GET_TYPEDEF_CLASSES
#include "garel/GARelOpsTypes.h.inc"

namespace garel {

bool isColumnType(mlir::Type t);

} // namespace garel
