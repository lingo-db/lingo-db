#include "lingodb/compiler/Dialect/Arrow/IR/ArrowOps.h"
#include "lingodb/compiler/Dialect/Arrow/IR/ArrowDialect.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"

#include "mlir/IR/OpImplementation.h"
#include <unordered_set>

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Support/Debug.h>
#include <queue>
using namespace mlir;

#define GET_OP_CLASSES
#include "lingodb/compiler/Dialect/Arrow/IR/ArrowOps.cpp.inc"
