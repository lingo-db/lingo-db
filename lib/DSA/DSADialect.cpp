#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/IR/DialectImplementation.h"
#include <mlir/Transforms/InliningUtils.h>
using namespace mlir;
using namespace mlir::dsa;
struct DSAInlinerInterface : public DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;
   bool isLegalToInline(Operation*, Region*, bool, BlockAndValueMapping&) const final override {
      return true;
   }
   virtual bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned, BlockAndValueMapping& valueMapping) const override {
      return true;
   }
};
void DSADialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/DSA/IR/DSAOps.cpp.inc"
      >();
   addInterfaces<DSAInlinerInterface>();
   registerTypes();
}
#include "mlir/Dialect/DSA/IR/DSAOpsDialect.cpp.inc"
