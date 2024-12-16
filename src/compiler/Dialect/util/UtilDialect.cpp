#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "mlir/IR/DialectImplementation.h"
#include <mlir/Transforms/InliningUtils.h>

struct UtilInlinerInterface : public mlir::DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;
   bool isLegalToInline(mlir::Operation*, mlir::Region*, bool, mlir::IRMapping&) const final override {
      return true;
   }
};
void lingodb::compiler::dialect::util::UtilDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "lingodb/compiler/Dialect/util/UtilOps.cpp.inc"
      >();
   addInterfaces<UtilInlinerInterface>();
   registerTypes();
   functionHelper = std::make_shared<FunctionHelper>();
}
#include "lingodb/compiler/Dialect/util/UtilOpsDialect.cpp.inc"
