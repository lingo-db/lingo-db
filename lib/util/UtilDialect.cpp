#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/DialectImplementation.h"
#include <mlir/Transforms/InliningUtils.h>
using namespace mlir;
using namespace mlir::util;
struct UtilInlinerInterface : public DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;

   //===--------------------------------------------------------------------===//
   // Analysis Hooks
   //===--------------------------------------------------------------------===//

   /// All call operations within toy can be inlined.
   bool isLegalToInline(Operation* call, Operation* callable,
                        bool wouldBeCloned) const final {
      return true;
   }

   /// All operations within toy can be inlined.
   bool isLegalToInline(Operation*, Region*, bool,
                        BlockAndValueMapping&) const final {
      return true;
   }
   virtual bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                                BlockAndValueMapping &valueMapping) const {
      return true;
   }
};
void UtilDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/util/UtilOps.cpp.inc"
      >();
   addInterfaces<UtilInlinerInterface>();
   registerTypes();
}