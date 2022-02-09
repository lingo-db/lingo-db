#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include <unordered_set>

using namespace mlir;

::mlir::LogicalResult verify(mlir::util::UnPackOp unPackOp) {
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
::mlir::LogicalResult verify(mlir::util::PackOp packOp) {
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
      if (auto setOp = mlir::dyn_cast_or_null<mlir::util::SetTupleOp>(tupleCreationOp)) {
         if (auto* tupleCreationOp2 = setOp.tuple().getDefiningOp()) {
            if (auto packOp = mlir::dyn_cast_or_null<mlir::util::PackOp>(tupleCreationOp2)) {
               std::vector<Value> vals(packOp.vals().begin(), packOp.vals().end());
               vals[setOp.offset()] = setOp.val();
               rewriter.replaceOp(unPackOp.getOperation(), vals);
               return success();
            }
         }
      }
   }
   return failure();
}
void mlir::util::LoadOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>>& effects) {
   if (!getOperation()->hasAttr("nosideffect")) {
      effects.emplace_back(MemoryEffects::Read::get());
   }
}

#define GET_OP_CLASSES
#include "mlir/Dialect/util/UtilOps.cpp.inc"
