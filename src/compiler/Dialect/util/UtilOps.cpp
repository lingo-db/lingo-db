#include "lingodb/compiler/Dialect/util/UtilOps.h"

#include "lingodb/compiler/Dialect/util/UtilDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace lingodb::compiler::dialect;

::mlir::LogicalResult util::UnPackOp::verify() {
   util::UnPackOp& unPackOp = *this;
   if (auto tupleType = mlir::dyn_cast_or_null<mlir::TupleType>(unPackOp.getTuple().getType())) {
      if (tupleType.getTypes().size() != unPackOp.getVals().size()) {
         unPackOp.emitOpError("must unpack exactly as much as entries in tuple");
         unPackOp.dump();
         return failure();
      }
      for (size_t i = 0; i < tupleType.getTypes().size(); i++) {
         if (tupleType.getTypes()[i] != unPackOp.getVals()[i].getType()) {
            unPackOp.emitOpError("types must match during unpacking");
            unPackOp.dump();
            return failure();
         }
      }
   } else {
      unPackOp.emitOpError("must be tupletype");
      return failure();
   }
   return success();
}
::mlir::LogicalResult util::PackOp::verify() {
   util::PackOp& packOp = *this;
   if (auto tupleType = mlir::dyn_cast_or_null<mlir::TupleType>(packOp.getTuple().getType())) {
      if (tupleType.getTypes().size() != packOp.getVals().size()) {
         packOp.emitOpError("must unpack exactly as much as entries in tuple");
         packOp.dump();
         return failure();
      }
      for (size_t i = 0; i < tupleType.getTypes().size(); i++) {
         if (tupleType.getTypes()[i] != packOp.getVals()[i].getType()) {
            packOp.emitOpError("types must match during unpacking");
            packOp.dump();
            return failure();
         }
      }
   } else {
      packOp.emitOpError("must be tupletype");
      return failure();
   }
   return success();
}

LogicalResult util::UnPackOp::canonicalize(util::UnPackOp unPackOp, mlir::PatternRewriter& rewriter) {
   auto tuple = unPackOp.getTuple();
   if (auto* tupleCreationOp = tuple.getDefiningOp()) {
      if (auto packOp = mlir::dyn_cast_or_null<util::PackOp>(tupleCreationOp)) {
         rewriter.replaceOp(unPackOp.getOperation(), packOp.getVals());
         return success();
      }
   }
   std::vector<Value> vals;
   vals.reserve(unPackOp.getNumResults());
   for (unsigned i = 0; i < unPackOp.getNumResults(); i++) {
      auto ty = unPackOp.getResultTypes()[i];
      vals.push_back(rewriter.create<util::GetTupleOp>(unPackOp.getLoc(), ty, tuple, i));
   }
   rewriter.replaceOp(unPackOp.getOperation(), vals);
   return success();
}

LogicalResult util::GetTupleOp::canonicalize(util::GetTupleOp op, mlir::PatternRewriter& rewriter) {
   if (auto* tupleCreationOp = op.getTuple().getDefiningOp()) {
      if (auto packOp = mlir::dyn_cast_or_null<util::PackOp>(tupleCreationOp)) {
         rewriter.replaceOp(op.getOperation(), packOp.getOperand(op.getOffset()));
         return success();
      }
      if (auto selOp = mlir::dyn_cast_or_null<mlir::arith::SelectOp>(tupleCreationOp)) {
         auto sel1 = rewriter.create<util::GetTupleOp>(op.getLoc(), op.getVal().getType(), selOp.getTrueValue(), op.getOffset());
         auto sel2 = rewriter.create<util::GetTupleOp>(op.getLoc(), op.getVal().getType(), selOp.getFalseValue(), op.getOffset());
         rewriter.replaceOpWithNewOp<mlir::arith::SelectOp>(op, selOp.getCondition(), sel1, sel2);
         return success();
      }
      if (auto loadOp = mlir::dyn_cast_or_null<util::LoadOp>(tupleCreationOp)) {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPoint(loadOp);
         auto base = loadOp.getRef();
         if (auto idx = loadOp.getIdx()) {
            base = rewriter.create<util::ArrayElementPtrOp>(loadOp.getLoc(), base.getType(), base, idx);
         }

         auto elemTy = op.getResult().getType();
         auto elemRefTy = util::RefType::get(elemTy);
         auto tep = rewriter.create<util::TupleElementPtrOp>(loadOp.getLoc(), elemRefTy, base, op.getOffset());
         auto newLoad = rewriter.create<util::LoadOp>(loadOp.getLoc(), tep);
         rewriter.replaceOp(op.getOperation(), newLoad.getResult());
         return success();
      }
   }
   return failure();
}

LogicalResult util::StoreOp::canonicalize(util::StoreOp op, mlir::PatternRewriter& rewriter) {
   if (auto ty = mlir::dyn_cast_or_null<mlir::TupleType>(op.getVal().getType())) {
      auto base = op.getRef();
      if (auto idx = op.getIdx()) {
         base = rewriter.create<util::ArrayElementPtrOp>(op.getLoc(), base.getType(), base, idx);
      }
      for (size_t i = 0; i < ty.size(); i++) {
         auto elemRefTy = util::RefType::get(ty.getType(i));
         auto gt = rewriter.create<util::GetTupleOp>(op.getLoc(), ty.getType(i), op.getVal(), i);
         auto tep = rewriter.create<util::TupleElementPtrOp>(op.getLoc(), elemRefTy, base, i);
         rewriter.create<util::StoreOp>(op.getLoc(), gt, tep, Value());
      }
      rewriter.eraseOp(op);
      return mlir::success();
   }
   return failure();
}

::mlir::LogicalResult util::UndefOp::canonicalize(util::UndefOp op, ::mlir::PatternRewriter& rewriter) {
   if (auto ty = mlir::dyn_cast_or_null<mlir::TupleType>(op.getType())) {
      std::vector<mlir::Value> values;
      for (auto t : ty.getTypes()) {
         values.push_back(rewriter.create<util::UndefOp>(op->getLoc(), t));
      }
      rewriter.replaceOpWithNewOp<util::PackOp>(op, values);
      return mlir::success();
   }
   return failure();
}

void util::LoadOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>>& effects) {
   if (!getOperation()->hasAttr("nosideffect")) {
      effects.emplace_back(MemoryEffects::Read::get());
   }
}
::mlir::LogicalResult util::TupleElementPtrOp::verify() {
   util::TupleElementPtrOp& op = *this;
   auto resElementType = op.getType().getElementType();
   auto ptrTupleType = mlir::cast<mlir::TupleType>(mlir::cast<util::RefType>(op.getRef().getType()).getElementType());
   auto ptrElementType = ptrTupleType.getTypes()[op.getIdx()];
   if (resElementType != ptrElementType) {
      op.emitOpError("Element types do not match");
      mlir::OpPrintingFlags flags;
      flags.assumeVerified();
      op->print(llvm::outs(), flags);
      return mlir::failure();
   }
   return mlir::success();
}
#define GET_OP_CLASSES
#include "lingodb/compiler/Dialect/util/UtilOps.cpp.inc"
