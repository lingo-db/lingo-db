#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/RuntimeFunctions.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include <unordered_set>

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Support/Debug.h>
#include <queue>
using namespace mlir;
bool mlir::db::CmpOp::isEqualityPred() { return predicate() == mlir::db::DBCmpPredicate::eq; }
bool mlir::db::CmpOp::isLessPred(bool eq) { return predicate() == (eq ? mlir::db::DBCmpPredicate::lte : mlir::db::DBCmpPredicate::lt); }
bool mlir::db::CmpOp::isGreaterPred(bool eq) { return predicate() == (eq ? mlir::db::DBCmpPredicate::gte : mlir::db::DBCmpPredicate::gt); }
mlir::Value mlir::db::CmpOp::getLeft() { return left(); }
mlir::Value mlir::db::CmpOp::getRight() { return right(); }
mlir::Type constructNullableBool(MLIRContext* context, ValueRange operands) {
   bool nullable = llvm::any_of(operands, [](auto operand) { return operand.getType().template isa<mlir::db::NullableType>(); });
   mlir::Type restype = IntegerType::get(context, 1);
   if (nullable) {
      restype = mlir::db::NullableType::get(context, restype);
   }
   return restype;
}
mlir::Type getBaseType(mlir::Type t) {
   if (auto nullableT = t.dyn_cast_or_null<mlir::db::NullableType>()) {
      return nullableT.getType();
   }
   return t;
}
bool isIntegerType(mlir::Type type, unsigned int width) {
   auto asStdInt = type.dyn_cast_or_null<mlir::IntegerType>();
   return asStdInt && asStdInt.getWidth() == width;
}
int getIntegerWidth(mlir::Type type, bool isUnSigned) {
   auto asStdInt = type.dyn_cast_or_null<mlir::IntegerType>();
   if (asStdInt && asStdInt.isUnsigned() == isUnSigned) {
      return asStdInt.getWidth();
   }
   return 0;
}
LogicalResult inferReturnType(MLIRContext* context, Optional<Location> location, ValueRange operands, SmallVectorImpl<Type>& inferredReturnTypes) {
   bool anyNullables = llvm::any_of(operands, [](Value v) { return v.getType().isa<mlir::db::NullableType>(); });
   Type baseType = getBaseType(operands[0].getType());
   if (anyNullables) {
      inferredReturnTypes.push_back(mlir::db::NullableType::get(context, baseType));
   } else {
      inferredReturnTypes.push_back(baseType);
   }
   return success();
}
LogicalResult mlir::db::CmpOp::inferReturnTypes(
   MLIRContext* context, Optional<Location> location, ValueRange operands,
   DictionaryAttr attributes, RegionRange regions,
   SmallVectorImpl<Type>& inferredReturnTypes) {
   inferredReturnTypes.assign({constructNullableBool(context, operands)});
   return success();
}

::mlir::LogicalResult verify(mlir::db::RuntimeCall runtimeCall) {
   auto reg = runtimeCall.getContext()->getLoadedDialect<mlir::db::DBDialect>()->getRuntimeFunctionRegistry();
   if (!reg->verify(runtimeCall.fn().str(), runtimeCall.args().getTypes(), runtimeCall.getNumResults() == 1 ? runtimeCall.getResultTypes()[0] : mlir::Type())) {
      runtimeCall->emitError("could not find matching runtime function");
      return failure();
   }
   return success();
}
bool mlir::db::RuntimeCall::canHandleInvalidValues() {
   auto reg = getContext()->getLoadedDialect<mlir::db::DBDialect>()->getRuntimeFunctionRegistry();
   if (auto* fn = reg->lookup(this->fn().str())) {
      return fn->nullHandleType == RuntimeFunction::HandlesInvalidVaues;
   }
   return false;
}
bool mlir::db::RuntimeCall::canHandleNulls() {
   auto reg = getContext()->getLoadedDialect<mlir::db::DBDialect>()->getRuntimeFunctionRegistry();
   if (auto* fn = reg->lookup(this->fn().str())) {
      return fn->nullHandleType == RuntimeFunction::HandlesNulls;
   }
   return false;
}

bool mlir::db::CmpOp::canHandleInvalidValues() {
   auto type = getBaseType(left().getType());
   if (type.isa<db::StringType>()) {
      return false;
   }
   return true;
}
bool mlir::db::CastOp::canHandleInvalidValues() {
   if (getBaseType(getResult().getType()).isa<db::StringType>() || getBaseType(val().getType()).isa<db::StringType>()) {
      return false;
   }
   return true;
}


LogicalResult mlir::db::OrOp::canonicalize(mlir::db::OrOp orOp, mlir::PatternRewriter& rewriter) {
   llvm::SmallDenseMap<mlir::Value, size_t> usage;
   for (auto val : orOp.vals()) {
      if (!val.getDefiningOp()) return failure();
      if (auto andOp = mlir::dyn_cast_or_null<mlir::db::AndOp>(val.getDefiningOp())) {
         llvm::SmallPtrSet<mlir::Value, 4> alreadyUsed;
         for (auto andOperand : andOp.vals()) {
            if (!alreadyUsed.contains(andOperand)) {
               usage[andOperand]++;
               alreadyUsed.insert(andOperand);
            }
         }
      } else {
         return failure();
      }
   }
   size_t totalAnds = orOp.vals().size();
   llvm::SmallPtrSet<mlir::Value, 4> extracted;
   std::vector<mlir::Value> newOrOperands;
   for (auto val : orOp.vals()) {
      if (auto andOp = mlir::dyn_cast_or_null<mlir::db::AndOp>(val.getDefiningOp())) {
         std::vector<mlir::Value> keep;
         for (auto andOperand : andOp.vals()) {
            if (usage[andOperand] == totalAnds) {
               extracted.insert(andOperand);
            } else {
               keep.push_back(andOperand);
            }
         }
         if (keep.size() != andOp.vals().size()) {
            if (keep.size()) {
               newOrOperands.push_back(rewriter.create<mlir::db::AndOp>(andOp->getLoc(), keep));
            }
         } else {
            newOrOperands.push_back(andOp);
         }
      }
   }
   std::vector<Value> extractedAsVec;
   extractedAsVec.insert(extractedAsVec.end(), extracted.begin(), extracted.end());
   if (!extracted.empty()) {
      Value newOrOp = rewriter.create<mlir::db::OrOp>(orOp->getLoc(), newOrOperands);
      extractedAsVec.push_back(newOrOp);
      rewriter.replaceOpWithNewOp<mlir::db::AndOp>(orOp, extractedAsVec);
      return success();
   }
   return failure();
}
LogicalResult mlir::db::AndOp::canonicalize(mlir::db::AndOp andOp, mlir::PatternRewriter& rewriter) {
   llvm::DenseSet<mlir::Value> rawValues;
   llvm::DenseMap<mlir::Value, std::vector<mlir::db::CmpOp>> cmps;
   std::queue<mlir::Value> queue;
   queue.push(andOp);
   while (!queue.empty()) {
      auto current = queue.front();
      queue.pop();
      if (auto* definingOp = current.getDefiningOp()) {
         if (auto nestedAnd = mlir::dyn_cast_or_null<mlir::db::AndOp>(definingOp)) {
            for (auto v : nestedAnd.vals()) {
               queue.push(v);
            }
         } else if (auto cmpOp = mlir::dyn_cast_or_null<mlir::db::CmpOp>(definingOp)) {
            cmps[cmpOp.left()].push_back(cmpOp);
            cmps[cmpOp.right()].push_back(cmpOp);
            rawValues.insert(current);
         } else {
            rawValues.insert(current);
         }
      } else {
         rawValues.insert(current);
      }
   }
   for (auto m : cmps) {
      mlir::Value lower, upper;
      mlir::db::CmpOp lowerCmp, upperCmp;
      mlir::Value current = m.getFirst();
      if (auto* definingOp = current.getDefiningOp()) {
         if (mlir::isa<mlir::db::ConstantOp>(definingOp)) {
            continue;
         }
      }
      for (auto cmp : m.second) {
         if (!rawValues.contains(cmp)) continue;
         switch (cmp.predicate()) {
            case DBCmpPredicate::lt:
            case DBCmpPredicate::lte:
               if (cmp.left() == current) {
                  upper = cmp.right();
                  upperCmp = cmp;
               } else {
                  lower = cmp.left();
                  lowerCmp = cmp;
               }
               break;
            case DBCmpPredicate::gt:
            case DBCmpPredicate::gte:
               if (cmp.left() == current) {
                  lower = cmp.right();
                  lowerCmp = cmp;
               } else {
                  upper = cmp.left();
                  upperCmp = cmp;
               }
               break;
            default: break;
         }
      }
      if (lower && upper && lower.getDefiningOp() && upper.getDefiningOp() && mlir::isa<mlir::db::ConstantOp>(lower.getDefiningOp()) && mlir::isa<mlir::db::ConstantOp>(upper.getDefiningOp())) {
         auto lowerInclusive = lowerCmp.predicate() == DBCmpPredicate::gte || lowerCmp.predicate() == DBCmpPredicate::lte;
         auto upperInclusive = upperCmp.predicate() == DBCmpPredicate::gte || upperCmp.predicate() == DBCmpPredicate::lte;
         mlir::Value between = rewriter.create<mlir::db::BetweenOp>(lowerCmp->getLoc(), current, lower, upper, lowerInclusive, upperInclusive);
         rawValues.erase(lowerCmp);
         rawValues.erase(upperCmp);
         rawValues.insert(between);
      }
   }
   if (rawValues.size() != andOp.vals().size()) {
      rewriter.replaceOpWithNewOp<mlir::db::AndOp>(andOp, std::vector<mlir::Value>(rawValues.begin(), rawValues.end()));
      return success();
   }
   return failure();
}
#define GET_OP_CLASSES
#include "mlir/Dialect/DB/IR/DBOps.cpp.inc"
#include "mlir/Dialect/DB/IR/DBOpsInterfaces.cpp.inc"
