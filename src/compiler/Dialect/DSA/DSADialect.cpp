#include "lingodb/compiler/Dialect/DSA/IR/DSADialect.h"
#include "lingodb/compiler/Dialect/DSA/IR/DSAOps.h"
#include "mlir/IR/DialectImplementation.h"
#include <mlir/Transforms/InliningUtils.h>
using namespace mlir;

struct DSAInlinerInterface : public DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;
   bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const final override {
      return true;
   }
   virtual bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned, IRMapping& valueMapping) const override {
      return true;
   }
};
void lingodb::compiler::dialect::dsa::DSADialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "lingodb/compiler/Dialect/DSA/IR/DSAOps.cpp.inc"
      >();
   addInterfaces<DSAInlinerInterface>();
   registerTypes();
}
#include "lingodb/compiler/Dialect/DSA/IR/DSAOpsDialect.cpp.inc"
