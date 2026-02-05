#pragma once

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Region.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "garel/GARelTypes.h"

#define GET_OP_CLASSES
#include "garel/GARelOps.h.inc"
