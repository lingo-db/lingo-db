#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DB/IR/RuntimeFunctions.h"
#include "mlir/IR/DialectImplementation.h"
#include <mlir/Transforms/InliningUtils.h>
using namespace mlir;
using namespace mlir::db;
struct DBInlinerInterface : public DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;
   bool isLegalToInline(Operation*, Region*, bool, BlockAndValueMapping&) const final override {
      return true;
   }
   virtual bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned, BlockAndValueMapping& valueMapping) const override {
      return true;
   }
};
void DBDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/DB/IR/DBOps.cpp.inc"
      >();
   addInterfaces<DBInlinerInterface>();
   registerTypes();
   runtimeFunctionRegistry = mlir::db::RuntimeFunctionRegistry::getBuiltinRegistry(getContext());
}
#include "mlir/Dialect/DB/IR/DBOpsDialect.cpp.inc"
