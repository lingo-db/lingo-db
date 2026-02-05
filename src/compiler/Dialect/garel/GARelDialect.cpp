#include "lingodb/compiler/Dialect/garel/GARelDialect.h"
#include "lingodb/compiler/Dialect/garel/GARelOps.h"

#include "lingodb/compiler/Dialect/garel/GARelOpsDialect.cpp.inc"

namespace garel {

void GARelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "lingodb/compiler/Dialect/garel/GARelOps.cpp.inc"
      >();
  registerAttributes();
  registerTypes();
}

} // namespace garel
