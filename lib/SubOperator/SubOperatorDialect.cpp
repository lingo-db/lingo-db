

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/SubOperator/SubOperatorDialect.h"
#include "mlir/Dialect/SubOperator/SubOperatorOps.h"
#include "mlir/Transforms/FoldUtils.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::subop;

struct SubOperatorInlinerInterface : public mlir::DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;
   bool isLegalToInline(mlir::Operation*, mlir::Region*, bool, mlir::IRMapping&) const final override {
      return true;
   }
   virtual bool isLegalToInline(mlir::Region* dest, mlir::Region* src, bool wouldBeCloned, mlir::IRMapping& valueMapping) const override {
      return true;
   }
};
struct SubOpFoldInterface : public mlir::DialectFoldInterface {
   using DialectFoldInterface::DialectFoldInterface;

   bool shouldMaterializeInto(mlir::Region* region) const final {
      return true;
   }
};
void SubOperatorDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/SubOperator/SubOperatorOps.cpp.inc"
      >();
   registerTypes();
   registerAttrs();
   addInterfaces<SubOperatorInlinerInterface>();
   addInterfaces<SubOpFoldInterface>();
   getContext()->loadDialect<mlir::db::DBDialect>();
   getContext()->loadDialect<mlir::dsa::DSADialect>();
   getContext()->loadDialect<mlir::arith::ArithDialect>();
   getContext()->loadDialect<mlir::tuples::TupleStreamDialect>();
}

#include "mlir/Dialect/SubOperator/SubOperatorOpsDialect.cpp.inc"
