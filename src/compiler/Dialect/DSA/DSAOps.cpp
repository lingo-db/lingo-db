#include "lingodb/compiler/Dialect/DSA/IR/DSAOps.h"
#include "lingodb/compiler/Dialect/DSA/IR/DSADialect.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"

#include "mlir/IR/OpImplementation.h"
#include <unordered_set>

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Support/Debug.h>
#include <queue>
using namespace mlir;

#define GET_OP_CLASSES
#include "lingodb/compiler/Dialect/DSA/IR/DSAOps.cpp.inc"
#include "lingodb/compiler/Dialect/DSA/IR/DSAOpsInterfaces.cpp.inc"
