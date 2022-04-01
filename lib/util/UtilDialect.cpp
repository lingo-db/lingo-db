#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/DialectImplementation.h"
#include <mlir/Transforms/InliningUtils.h>

struct UtilInlinerInterface : public mlir::DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;
   bool isLegalToInline(mlir::Operation*, mlir::Region*, bool, mlir::BlockAndValueMapping&) const final override {
      return true;
   }
};
void mlir::util::UtilDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/util/UtilOps.cpp.inc"
      >();
   addInterfaces<UtilInlinerInterface>();
   registerTypes();
   functionHelper = std::make_shared<FunctionHelper>();
}
#include "mlir/Dialect/util/UtilOpsDialect.cpp.inc"
