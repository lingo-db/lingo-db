#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <iostream>

#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <variant>
namespace {
mlir::arith::CmpIPredicateAttr convertToCmpIPred(mlir::OpBuilder, ::mlir::db::DBCmpPredicateAttr p) {
   using namespace mlir;
   switch (p.getValue()) {
      case db::DBCmpPredicate::eq:
         return mlir::arith::CmpIPredicateAttr::get(p.getContext(), arith::CmpIPredicate::eq);
      case db::DBCmpPredicate::neq:
         return mlir::arith::CmpIPredicateAttr::get(p.getContext(), arith::CmpIPredicate::ne);
      case db::DBCmpPredicate::lt:
         return mlir::arith::CmpIPredicateAttr::get(p.getContext(), arith::CmpIPredicate::slt);
      case db::DBCmpPredicate::gt:
         return mlir::arith::CmpIPredicateAttr::get(p.getContext(), arith::CmpIPredicate::sgt);
      case db::DBCmpPredicate::lte:
         return mlir::arith::CmpIPredicateAttr::get(p.getContext(), arith::CmpIPredicate::sle);
      case db::DBCmpPredicate::gte:
         return mlir::arith::CmpIPredicateAttr::get(p.getContext(), arith::CmpIPredicate::sge);
   }
   return mlir::arith::CmpIPredicateAttr::get(p.getContext(), arith::CmpIPredicate::sge);
}
mlir::arith::CmpFPredicateAttr convertToCmpFPred(mlir::OpBuilder, ::mlir::db::DBCmpPredicateAttr p) {
   using namespace mlir;
   switch (p.getValue()) {
      case db::DBCmpPredicate::eq:
         return mlir::arith::CmpFPredicateAttr::get(p.getContext(), arith::CmpFPredicate::OEQ);
      case db::DBCmpPredicate::neq:
         return mlir::arith::CmpFPredicateAttr::get(p.getContext(), arith::CmpFPredicate::ONE);
      case db::DBCmpPredicate::lt:
         return mlir::arith::CmpFPredicateAttr::get(p.getContext(), arith::CmpFPredicate::OLT);
      case db::DBCmpPredicate::gt:
         return mlir::arith::CmpFPredicateAttr::get(p.getContext(), arith::CmpFPredicate::OGT);
      case db::DBCmpPredicate::lte:
         return mlir::arith::CmpFPredicateAttr::get(p.getContext(), arith::CmpFPredicate::OLE);
      case db::DBCmpPredicate::gte:
         return mlir::arith::CmpFPredicateAttr::get(p.getContext(), arith::CmpFPredicate::OGE);
   }
   return mlir::arith::CmpFPredicateAttr::get(p.getContext(), arith::CmpFPredicate::OGE);
}
mlir::Attribute convertConst(mlir::Attribute attr, mlir::Value v) {
   using namespace mlir;
   std::variant<int64_t, double, std::string> parseArg;
   if (auto integerAttr = attr.dyn_cast_or_null<IntegerAttr>()) {
      if (v.getType().isIntOrIndex()) {
         return IntegerAttr::get(v.getType(), integerAttr.getInt());
      }
   } else if (auto floatAttr = attr.dyn_cast_or_null<FloatAttr>()) {
      if (v.getType().isa<mlir::FloatType>()) {
         return FloatAttr::get(v.getType(), floatAttr.getValueAsDouble());
      }
   }
   return attr;
}
#include "SimplifyToArith.inc"

//Pattern that optimizes the join order
class SimplifyToArith : public mlir::PassWrapper<SimplifyToArith, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "db-simplify-to-arith"; }

   public:
   void runOnOperation() override {
      //transform "standalone" aggregation functions
      {
         mlir::RewritePatternSet patterns(&getContext());
         patterns.insert<DBCmpToCmpI>(&getContext());
         patterns.insert<DBCmpToCmpF>(&getContext());
         patterns.insert<DBAddToAddI>(&getContext());
         patterns.insert<DBAddToAddF>(&getContext());
         patterns.insert<DBConstToConst>(&getContext());
         if (mlir::applyPatternsAndFoldGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
            assert(false && "should not happen");
         }
      }
   }
};
} // end anonymous namespace

namespace mlir::db {

std::unique_ptr<Pass> createSimplifyToArithPass() { return std::make_unique<SimplifyToArith>(); }

} // end namespace mlir::db