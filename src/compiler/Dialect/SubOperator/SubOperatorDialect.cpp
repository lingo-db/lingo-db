#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"

#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/DSA/IR/DSADialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace lingodb::compiler::dialect;

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
void subop::SubOperatorDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.cpp.inc"

      >();
   registerTypes();
   registerAttrs();
   addInterfaces<SubOperatorInlinerInterface>();
   addInterfaces<SubOpFoldInterface>();
   getContext()->loadDialect<db::DBDialect>();
   getContext()->loadDialect<dsa::DSADialect>();
   getContext()->loadDialect<mlir::arith::ArithDialect>();
   getContext()->loadDialect<mlir::index::IndexDialect>();
   getContext()->loadDialect<tuples::TupleStreamDialect>();
}

#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsDialect.cpp.inc"
