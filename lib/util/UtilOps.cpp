#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/util/UtilDialect.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include <unordered_set>

using namespace mlir;

::mlir::LogicalResult mlir::util::UnPackOp::verify() {
   mlir::util::UnPackOp& unPackOp = *this;
   if (auto tupleType = unPackOp.tuple().getType().dyn_cast_or_null<mlir::TupleType>()) {
      if (tupleType.getTypes().size() != unPackOp.vals().size()) {
         unPackOp.emitOpError("must unpack exactly as much as entries in tuple");
         unPackOp.dump();
         return failure();
      }
      for (size_t i = 0; i < tupleType.getTypes().size(); i++) {
         if (tupleType.getTypes()[i] != unPackOp.vals()[i].getType()) {
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
::mlir::LogicalResult mlir::util::PackOp::verify() {
   mlir::util::PackOp& packOp = *this;
   if (auto tupleType = packOp.tuple().getType().dyn_cast_or_null<mlir::TupleType>()) {
      if (tupleType.getTypes().size() != packOp.vals().size()) {
         packOp.emitOpError("must unpack exactly as much as entries in tuple");
         packOp.dump();
         return failure();
      }
      for (size_t i = 0; i < tupleType.getTypes().size(); i++) {
         if (tupleType.getTypes()[i] != packOp.vals()[i].getType()) {
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

LogicalResult mlir::util::UnPackOp::canonicalize(mlir::util::UnPackOp unPackOp, mlir::PatternRewriter& rewriter) {
   auto tuple = unPackOp.tuple();
   if (auto* tupleCreationOp = tuple.getDefiningOp()) {
      if (auto packOp = mlir::dyn_cast_or_null<mlir::util::PackOp>(tupleCreationOp)) {
         rewriter.replaceOp(unPackOp.getOperation(), packOp.vals());
         return success();
      }
   }
   std::vector<Value> vals;
   vals.reserve(unPackOp.getNumResults());
   for (unsigned i = 0; i < unPackOp.getNumResults(); i++) {
      auto ty = unPackOp.getResultTypes()[i];
      vals.push_back(rewriter.create<mlir::util::GetTupleOp>(unPackOp.getLoc(), ty, tuple, i));
   }
   rewriter.replaceOp(unPackOp.getOperation(), vals);
   return success();
}

LogicalResult mlir::util::GetTupleOp::canonicalize(mlir::util::GetTupleOp op, mlir::PatternRewriter& rewriter) {
   if (auto* tupleCreationOp = op.tuple().getDefiningOp()) {
      if (auto packOp = mlir::dyn_cast_or_null<mlir::util::PackOp>(tupleCreationOp)) {
         rewriter.replaceOp(op.getOperation(), packOp.getOperand(op.offset()));
         return success();
      }
      if (auto selOp = mlir::dyn_cast_or_null<mlir::arith::SelectOp>(tupleCreationOp)) {
         auto sel1 = rewriter.create<mlir::util::GetTupleOp>(op.getLoc(), op.val().getType(), selOp.getTrueValue(), op.offset());
         auto sel2 = rewriter.create<mlir::util::GetTupleOp>(op.getLoc(), op.val().getType(), selOp.getFalseValue(), op.offset());
         rewriter.replaceOpWithNewOp<mlir::arith::SelectOp>(op, selOp.getCondition(), sel1, sel2);
         return success();
      }
      if (auto loadOp = mlir::dyn_cast_or_null<mlir::util::LoadOp>(tupleCreationOp)) {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPoint(loadOp);
         auto base = loadOp.ref();
         if (auto idx = loadOp.idx()) {
            base = rewriter.create<mlir::util::ArrayElementPtrOp>(loadOp.getLoc(), base.getType(), base, idx);
         }

         auto elemTy = op.getResult().getType();
         auto elemRefTy = mlir::util::RefType::get(elemTy);
         auto tep = rewriter.create<mlir::util::TupleElementPtrOp>(loadOp.getLoc(), elemRefTy, base, op.offset());
         auto newLoad = rewriter.create<mlir::util::LoadOp>(loadOp.getLoc(), tep);
         rewriter.replaceOp(op.getOperation(), newLoad.getResult());
         return success();
      }
   }
   return failure();
}

LogicalResult mlir::util::StoreOp::canonicalize(mlir::util::StoreOp op, mlir::PatternRewriter& rewriter) {
   if (auto ty = op.val().getType().dyn_cast_or_null<mlir::TupleType>()) {
      auto base = op.ref();
      if (auto idx = op.idx()) {
         base = rewriter.create<mlir::util::ArrayElementPtrOp>(op.getLoc(), base.getType(), base, idx);
      }
      for (size_t i = 0; i < ty.size(); i++) {
         auto elemRefTy = mlir::util::RefType::get(ty.getType(i));
         auto gt = rewriter.create<mlir::util::GetTupleOp>(op.getLoc(), ty.getType(i), op.val(), i);
         auto tep = rewriter.create<mlir::util::TupleElementPtrOp>(op.getLoc(), elemRefTy, base, i);
         rewriter.create<mlir::util::StoreOp>(op.getLoc(), gt, tep, Value());
      }
      rewriter.eraseOp(op);
      return mlir::success();
   }
   return failure();
}

void mlir::util::LoadOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>>& effects) {
   if (!getOperation()->hasAttr("nosideffect")) {
      effects.emplace_back(MemoryEffects::Read::get());
   }
}
::mlir::LogicalResult mlir::util::TupleElementPtrOp::verify() {
   mlir::util::TupleElementPtrOp& op = *this;
   auto resElementType = op.getType().getElementType();
   auto ptrTupleType = op.ref().getType().cast<mlir::util::RefType>().getElementType().cast<mlir::TupleType>();
   auto ptrElementType = ptrTupleType.getTypes()[op.idx()];
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
#include "mlir/Dialect/util/UtilOps.cpp.inc"
