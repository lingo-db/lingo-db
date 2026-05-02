#include "lingodb/compiler/Dialect/PyInterp/PyInterpOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include <llvm/ADT/TypeSwitch.h>

using namespace mlir;

#define GET_OP_CLASSES
#include "lingodb/compiler/Dialect/PyInterp/PyInterpOps.cpp.inc"
