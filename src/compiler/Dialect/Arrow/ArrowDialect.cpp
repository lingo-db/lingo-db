#include "lingodb/compiler/Dialect/Arrow/IR/ArrowDialect.h"
#include "lingodb/compiler/Dialect/Arrow/IR/ArrowOps.h"
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
void lingodb::compiler::dialect::arrow::ArrowDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "lingodb/compiler/Dialect/Arrow/IR/ArrowOps.cpp.inc"
      >();
   addInterfaces<DSAInlinerInterface>();
   registerTypes();
}
#include "lingodb/compiler/Dialect/Arrow/IR/ArrowOpsDialect.cpp.inc"
