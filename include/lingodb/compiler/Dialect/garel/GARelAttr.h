#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>

#include "lingodb/compiler/Dialect/garel/GARelEnumAttr.h.inc"

namespace garel {

/** Reference to a column inside of \c RelationType or \c TupleType. */
using ColumnIdx = std::int32_t;

} // namespace garel

#define GET_ATTRDEF_CLASSES
#include "lingodb/compiler/Dialect/garel/GARelAttr.h.inc"
