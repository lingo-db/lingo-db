#pragma once

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <lingodb/compiler/Dialect/graphalg/GraphAlgAttr.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgInterfaces.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgTypes.h>

#define GET_OP_CLASSES
#include "lingodb/compiler/Dialect/graphalg/GraphAlgOps.h.inc"

namespace graphalg {
} // namespace graphalg
