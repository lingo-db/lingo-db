#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/DB/IR/RuntimeFunctions.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include <unordered_set>

#include "lingodb/compiler/mlir-support/parsing.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Support/Debug.h>
#include <queue>
using namespace mlir;
using namespace lingodb::compiler::dialect;

bool db::CmpOp::isEqualityPred(bool nullsAreEqual) { return getPredicate() == db::DBCmpPredicate::eq || (nullsAreEqual ? (getPredicate() == DBCmpPredicate::isa) : false); }
bool db::CmpOp::isUnequalityPred() { return getPredicate() == db::DBCmpPredicate::neq; }
bool db::CmpOp::isLessPred(bool eq) { return getPredicate() == (eq ? db::DBCmpPredicate::lte : db::DBCmpPredicate::lt); }
bool db::CmpOp::isGreaterPred(bool eq) { return getPredicate() == (eq ? db::DBCmpPredicate::gte : db::DBCmpPredicate::gt); }
mlir::Type getBaseType(mlir::Type t) {
   if (auto nullableT = mlir::dyn_cast_or_null<db::NullableType>(t)) {
      return nullableT.getType();
   }
   return t;
}
Type wrapNullableType(MLIRContext* context, Type type, ValueRange values) {
   if (llvm::any_of(values, [](Value v) { return mlir::isa<db::NullableType>(v.getType()); })) {
      return db::NullableType::get(type);
   }
   return type;
}
bool isIntegerType(mlir::Type type, unsigned int width) {
   auto asStdInt = mlir::dyn_cast_or_null<mlir::IntegerType>(type);
   return asStdInt && asStdInt.getWidth() == width;
}
int getIntegerWidth(mlir::Type type, bool isUnSigned) {
   auto asStdInt = mlir::dyn_cast_or_null<mlir::IntegerType>(type);
   if (asStdInt && asStdInt.isUnsigned() == isUnSigned) {
      return asStdInt.getWidth();
   }
   return 0;
}
namespace {

std::tuple<arrow::Type::type, uint32_t, uint32_t> convertTypeToArrow(mlir::Type type) {
   arrow::Type::type typeConstant = arrow::Type::type::NA;
   uint32_t param1 = 0, param2 = 0;
   if (isIntegerType(type, 1)) {
      typeConstant = arrow::Type::type::BOOL;
   } else if (auto intWidth = getIntegerWidth(type, false)) {
      switch (intWidth) {
         case 8: typeConstant = arrow::Type::type::INT8; break;
         case 16: typeConstant = arrow::Type::type::INT16; break;
         case 32: typeConstant = arrow::Type::type::INT32; break;
         case 64: typeConstant = arrow::Type::type::INT64; break;
      }
   } else if (auto uIntWidth = getIntegerWidth(type, true)) {
      switch (uIntWidth) {
         case 8: typeConstant = arrow::Type::type::UINT8; break;
         case 16: typeConstant = arrow::Type::type::UINT16; break;
         case 32: typeConstant = arrow::Type::type::UINT32; break;
         case 64: typeConstant = arrow::Type::type::UINT64; break;
      }
   } else if (auto decimalType = mlir::dyn_cast_or_null<db::DecimalType>(type)) {
      typeConstant = arrow::Type::type::DECIMAL128;
      param1 = decimalType.getP();
      param2 = decimalType.getS();
   } else if (auto floatType = mlir::dyn_cast_or_null<mlir::FloatType>(type)) {
      switch (floatType.getWidth()) {
         case 16: typeConstant = arrow::Type::type::HALF_FLOAT; break;
         case 32: typeConstant = arrow::Type::type::FLOAT; break;
         case 64: typeConstant = arrow::Type::type::DOUBLE; break;
      }
   } else if (auto stringType = mlir::dyn_cast_or_null<db::StringType>(type)) {
      typeConstant = arrow::Type::type::STRING;
   } else if (auto dateType = mlir::dyn_cast_or_null<db::DateType>(type)) {
      if (dateType.getUnit() == db::DateUnitAttr::day) {
         typeConstant = arrow::Type::type::DATE32;
      } else {
         typeConstant = arrow::Type::type::DATE64;
      }
   } else if (auto charType = mlir::dyn_cast_or_null<db::CharType>(type)) {
      typeConstant = arrow::Type::type::STRING;
      param1 = charType.getBytes();
   } else if (auto intervalType = mlir::dyn_cast_or_null<db::IntervalType>(type)) {
      if (intervalType.getUnit() == db::IntervalUnitAttr::months) {
         typeConstant = arrow::Type::type::INTERVAL_MONTHS;
      } else {
         typeConstant = arrow::Type::type::INTERVAL_DAY_TIME;
      }
   } else if (auto timestampType = mlir::dyn_cast_or_null<db::TimestampType>(type)) {
      typeConstant = arrow::Type::type::TIMESTAMP;
      param1 = static_cast<uint32_t>(timestampType.getUnit());
   }
   assert(typeConstant != arrow::Type::type::NA);
   return {typeConstant, param1, param2};
}
mlir::Type getAdaptedDecimalTypeAfterMulDiv(mlir::MLIRContext* context, int precision, int scale) {
   int beforeComma = precision - scale;
   if (beforeComma > 32 && scale > 6) {
      return db::DecimalType::get(context, 38, 6);
   }
   if (beforeComma > 32 && scale <= 6) {
      return db::DecimalType::get(context, 38, scale);
   }
   return db::DecimalType::get(context, std::min(precision, 38), std::min(scale, 38 - beforeComma));
}
} // namespace
OpFoldResult db::ConstantOp::fold(db::ConstantOp::FoldAdaptor adaptor) {
   auto type = getType();
   auto [arrowType, param1, param2] = convertTypeToArrow(type);
   std::variant<int64_t, double, std::string> parseArg;
   if (auto integerAttr = mlir::dyn_cast_or_null<IntegerAttr>(getValue())) {
      parseArg = integerAttr.getInt();
   } else if (auto floatAttr = mlir::dyn_cast_or_null<FloatAttr>(getValue())) {
      parseArg = floatAttr.getValueAsDouble();
   } else if (auto stringAttr = mlir::dyn_cast_or_null<StringAttr>(getValue())) {
      parseArg = stringAttr.str();
   } else {
      return {};
   }
   auto parseResult = support::parse(parseArg, arrowType, param1, param2);
   if (auto decimalType = mlir::dyn_cast_or_null<db::DecimalType>(type)) {
      auto [low, high] = support::parseDecimal(std::get<std::string>(parseResult), decimalType.getS());
      std::vector<uint64_t> parts = {low, high};
      return IntegerAttr::get(mlir::IntegerType::get(getContext(), 128), mlir::APInt(128, parts));
   } else if (auto integerType = mlir::dyn_cast_or_null<mlir::IntegerType>(type)) {
      return IntegerAttr::get(integerType, std::get<int64_t>(parseResult));
   } else if (mlir::isa<mlir::FloatType>(type)) {
      return FloatAttr::get(type, std::get<double>(parseResult));
   } else if (mlir::isa<db::StringType>(type)) {
      std::string str = std::get<std::string>(parseResult);
      return mlir::StringAttr::get(getContext(), str);
   } else if (mlir::isa<db::CharType>(type)) {
      std::string str = std::get<std::string>(parseResult);
      return mlir::StringAttr::get(getContext(), str);
   } else if (mlir::isa<db::IntervalType, db::DateType, db::TimestampType>(type)) {
      return mlir::IntegerAttr::get(mlir::IntegerType::get(getContext(), 64), std::get<int64_t>(parseResult));
   } else {
      type.dump();
   }
   return {};
}

::mlir::OpFoldResult db::AddOp::fold(db::AddOp::FoldAdaptor adaptor) {
   auto left = mlir::dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getLeft());
   auto right = mlir::dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getRight());
   if (left && right && left.getType() == right.getType()) {
      return IntegerAttr::get(left.getType(), left.getValue() + right.getValue());
   }
   return {};
}
::mlir::OpFoldResult db::SubOp::fold(db::SubOp::FoldAdaptor adaptor) {
   auto left = mlir::dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getLeft());
   auto right = mlir::dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getRight());
   if (left && right && left.getType() == right.getType()) {
      return IntegerAttr::get(left.getType(), left.getValue() - right.getValue());
   }
   return {};
}

::mlir::OpFoldResult db::CastOp::fold(db::CastOp::FoldAdaptor adaptor) {
   auto scalarSourceType = getVal().getType();
   auto scalarTargetType = getType();
   if (mlir::isa<db::StringType>(scalarSourceType) || mlir::isa<db::StringType>(scalarTargetType)) return {};
   if (mlir::isa<db::NullableType>(scalarSourceType)) return {};
   if (scalarSourceType == scalarTargetType) {
      return adaptor.getVal();
   }
   if (!adaptor.getVal()) return {};
   if (getIntegerWidth(scalarSourceType, false)) {
      auto intVal = mlir::cast<mlir::IntegerAttr>(adaptor.getVal()).getInt();
      if (mlir::isa<FloatType>(scalarTargetType)) {
         return mlir::FloatAttr::get(scalarTargetType, (double) intVal);
      } else if (auto decimalTargetType = mlir::dyn_cast_or_null<db::DecimalType>(scalarTargetType)) {
         auto [low, high] = support::getDecimalScaleMultiplier(decimalTargetType.getS());
         std::vector<uint64_t> parts = {low, high};
         return IntegerAttr::get(IntegerType::get(getContext(), 128), APInt(128, intVal) * APInt(128, parts));
      } else if (getIntegerWidth(scalarTargetType, false)) {
         return {};
      }
   } else if (auto floatType = mlir::dyn_cast_or_null<FloatType>(scalarSourceType)) {
      if (getIntegerWidth(scalarTargetType, false)) {
         return mlir::IntegerAttr::get(scalarTargetType, mlir::cast<mlir::FloatAttr>(adaptor.getVal()).getValueAsDouble());
      } else if (auto decimalTargetType = mlir::dyn_cast_or_null<db::DecimalType>(scalarTargetType)) {
         return {};
      }
   } else if (auto decimalSourceType = mlir::dyn_cast_or_null<db::DecimalType>(scalarSourceType)) {
      if (auto decimalTargetType = mlir::dyn_cast_or_null<db::DecimalType>(scalarTargetType)) {
         auto sourceScale = decimalSourceType.getS();
         auto targetScale = decimalTargetType.getS();
         if (sourceScale == targetScale) {
            return adaptor.getVal();
         }
         return {};
      } else if (mlir::isa<FloatType>(scalarTargetType)) {
         return {};
      } else if (getIntegerWidth(scalarTargetType, false)) {
         return {};
      }
   }
   return {};
}

::mlir::LogicalResult db::RuntimeCall::fold(db::RuntimeCall::FoldAdaptor adaptor, ::llvm::SmallVectorImpl<::mlir::OpFoldResult>& results) {
   auto reg = getContext()->getLoadedDialect<db::DBDialect>()->getRuntimeFunctionRegistry();
   auto* fn = reg->lookup(getFn().str());
   if (!fn) { return failure(); }
   if (!fn->foldFn) return failure();
   auto foldFn = fn->foldFn.value();
   return foldFn(getOperandTypes(), adaptor.getOperands(), results);
}
namespace {
LogicalResult inferReturnType(MLIRContext* context, std::optional<Location> location, ValueRange operands, SmallVectorImpl<Type>& inferredReturnTypes) {
   Type baseTypeLeft = getBaseType(operands[0].getType());
   Type baseTypeRight = getBaseType(operands[1].getType());
   Type baseType = baseTypeLeft;
   if (mlir::isa<db::DecimalType>(baseTypeLeft)) {
      auto a = mlir::cast<db::DecimalType>(baseTypeLeft);
      auto b = mlir::cast<db::DecimalType>(baseTypeRight);
      auto hidig = std::max(a.getP() - a.getS(), b.getP() - b.getS());
      auto maxs = std::max(a.getS(), b.getS());
      // Addition is super-type of both, with larger precision for carry.
      // TODO: actually add carry precision (+1).
      baseType = db::DecimalType::get(a.getContext(), hidig + maxs, maxs);
   }
   inferredReturnTypes.push_back(wrapNullableType(context, baseType, operands));
   return success();
}
LogicalResult inferMulReturnType(MLIRContext* context, std::optional<Location> location, ValueRange operands, SmallVectorImpl<Type>& inferredReturnTypes) {
   Type baseTypeLeft = getBaseType(operands[0].getType());
   Type baseTypeRight = getBaseType(operands[1].getType());
   Type baseType = baseTypeLeft;
   if (mlir::isa<db::DecimalType>(baseTypeLeft)) {
      auto a = mlir::cast<db::DecimalType>(baseTypeLeft);
      auto b = mlir::cast<db::DecimalType>(baseTypeRight);
      auto sump = a.getP() + b.getP();
      auto sums = a.getS() + b.getS();
      baseType = getAdaptedDecimalTypeAfterMulDiv(context, sump, sums);
   }
   inferredReturnTypes.push_back(wrapNullableType(context, baseType, operands));
   return success();
}
LogicalResult inferDivReturnType(MLIRContext* context, std::optional<Location> location, ValueRange operands, SmallVectorImpl<Type>& inferredReturnTypes) {
   Type baseTypeLeft = getBaseType(operands[0].getType());
   Type baseTypeRight = getBaseType(operands[1].getType());
   Type baseType = baseTypeLeft;
   if (mlir::isa<db::DecimalType>(baseTypeLeft)) {
      auto a = mlir::dyn_cast<db::DecimalType>(baseTypeLeft);
      auto b = mlir::dyn_cast<db::DecimalType>(baseTypeRight);
      baseType = getAdaptedDecimalTypeAfterMulDiv(context, a.getP() - a.getS() + b.getS() + std::max(6, a.getS() + b.getP()), std::max(6, a.getS() + b.getP()));
   }
   inferredReturnTypes.push_back(wrapNullableType(context, baseType, operands));
   return success();
}

LogicalResult inferRemReturnType(MLIRContext* context, std::optional<Location> location, ValueRange operands, SmallVectorImpl<Type>& inferredReturnTypes) {
   Type baseTypeLeft = getBaseType(operands[0].getType());
   Type baseTypeRight = getBaseType(operands[1].getType());
   Type baseType = baseTypeLeft;
   if (mlir::isa<db::DecimalType>(baseTypeLeft)) {
      auto a = mlir::dyn_cast<db::DecimalType>(baseTypeLeft);
      auto b = mlir::dyn_cast<db::DecimalType>(baseTypeRight);
      baseType = db::DecimalType::get(a.getContext(), std::min(a.getP() - a.getS(), b.getP() - b.getS()) + std::max(a.getS(), b.getS()), std::max(a.getS(), b.getS()));
   }
   inferredReturnTypes.push_back(wrapNullableType(context, baseType, operands));
   return success();
}
} // namespace
::mlir::LogicalResult db::RuntimeCall::verify() {
   db::RuntimeCall& runtimeCall = *this;
   auto reg = runtimeCall.getContext()->getLoadedDialect<db::DBDialect>()->getRuntimeFunctionRegistry();
   if (!reg->verify(runtimeCall.getFn().str(), runtimeCall.getArgs().getTypes(), runtimeCall.getNumResults() == 1 ? runtimeCall.getResultTypes()[0] : mlir::Type())) {
      runtimeCall->emitError("could not find matching runtime function");
      return failure();
   }
   return success();
}
bool db::RuntimeCall::supportsInvalidValues() {
   auto reg = getContext()->getLoadedDialect<db::DBDialect>()->getRuntimeFunctionRegistry();
   if (auto* fn = reg->lookup(this->getFn().str())) {
      return fn->nullHandleType == RuntimeFunction::HandlesInvalidVaues;
   }
   return false;
}
bool db::RuntimeCall::needsNullWrap() {
   auto reg = getContext()->getLoadedDialect<db::DBDialect>()->getRuntimeFunctionRegistry();
   if (auto* fn = reg->lookup(this->getFn().str())) {
      return fn->nullHandleType != RuntimeFunction::HandlesNulls;
   }
   return false;
}

bool db::CmpOp::supportsInvalidValues() {
   auto type = getBaseType(getLeft().getType());
   if (mlir::isa<db::StringType>(type)) {
      return false;
   }
   return true;
}
bool db::CastOp::supportsInvalidValues() {
   if (mlir::isa<db::StringType>(getBaseType(getResult().getType())) || mlir::isa<db::StringType>(getBaseType(getVal().getType()))) {
      return false;
   }
   return true;
}

LogicalResult db::OrOp::canonicalize(db::OrOp orOp, mlir::PatternRewriter& rewriter) {
   llvm::SmallDenseMap<mlir::Value, size_t> usage;
   for (auto val : orOp.getVals()) {
      if (!val.getDefiningOp()) return failure();
      if (auto andOp = mlir::dyn_cast_or_null<db::AndOp>(val.getDefiningOp())) {
         llvm::SmallPtrSet<mlir::Value, 4> alreadyUsed;
         for (auto andOperand : andOp.getVals()) {
            if (!alreadyUsed.contains(andOperand)) {
               usage[andOperand]++;
               alreadyUsed.insert(andOperand);
            }
         }
      } else {
         return failure();
      }
   }
   size_t totalAnds = orOp.getVals().size();
   llvm::SmallPtrSet<mlir::Value, 4> extracted;
   std::vector<mlir::Value> newOrOperands;
   for (auto val : orOp.getVals()) {
      if (auto andOp = mlir::dyn_cast_or_null<db::AndOp>(val.getDefiningOp())) {
         std::vector<mlir::Value> keep;
         for (auto andOperand : andOp.getVals()) {
            if (usage[andOperand] == totalAnds) {
               extracted.insert(andOperand);
            } else {
               keep.push_back(andOperand);
            }
         }
         if (keep.size() != andOp.getVals().size()) {
            if (keep.size()) {
               newOrOperands.push_back(rewriter.create<db::AndOp>(andOp->getLoc(), keep));
            }
         } else {
            newOrOperands.push_back(andOp);
         }
      }
   }
   std::vector<Value> extractedAsVec;
   extractedAsVec.insert(extractedAsVec.end(), extracted.begin(), extracted.end());
   if (!extracted.empty()) {
      if (newOrOperands.size() == 1) {
         extractedAsVec.push_back(newOrOperands[0]);
      } else if (newOrOperands.size() > 1) {
         Value newOrOp = rewriter.create<db::OrOp>(orOp->getLoc(), newOrOperands);
         extractedAsVec.push_back(newOrOp);
      }
      rewriter.replaceOpWithNewOp<db::AndOp>(orOp, extractedAsVec);
      return success();
   } else if (newOrOperands.size() == 1) {
      rewriter.replaceOp(orOp, newOrOperands[0]);
   }
   return failure();
}
LogicalResult db::AndOp::canonicalize(db::AndOp andOp, mlir::PatternRewriter& rewriter) {
   llvm::DenseSet<mlir::Value> rawValues;
   llvm::DenseMap<mlir::Value, std::vector<db::CmpOp>> cmps;
   std::queue<mlir::Value> queue;
   queue.push(andOp);
   while (!queue.empty()) {
      auto current = queue.front();
      queue.pop();
      if (auto* definingOp = current.getDefiningOp()) {
         if (auto nestedAnd = mlir::dyn_cast_or_null<db::AndOp>(definingOp)) {
            for (auto v : nestedAnd.getVals()) {
               queue.push(v);
            }
         } else if (auto cmpOp = mlir::dyn_cast_or_null<db::CmpOp>(definingOp)) {
            cmps[cmpOp.getLeft()].push_back(cmpOp);
            cmps[cmpOp.getRight()].push_back(cmpOp);
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
      db::CmpOp lowerCmp, upperCmp;
      mlir::Value current = m.getFirst();
      if (auto* definingOp = current.getDefiningOp()) {
         if (mlir::isa<db::ConstantOp>(definingOp)) {
            continue;
         }
      }
      for (auto cmp : m.second) {
         if (!rawValues.contains(cmp)) continue;
         switch (cmp.getPredicate()) {
            case DBCmpPredicate::lt:
            case DBCmpPredicate::lte:
               if (cmp.getLeft() == current) {
                  upper = cmp.getRight();
                  upperCmp = cmp;
               } else {
                  lower = cmp.getLeft();
                  lowerCmp = cmp;
               }
               break;
            case DBCmpPredicate::gt:
            case DBCmpPredicate::gte:
               if (cmp.getLeft() == current) {
                  lower = cmp.getRight();
                  lowerCmp = cmp;
               } else {
                  upper = cmp.getLeft();
                  upperCmp = cmp;
               }
               break;
            default: break;
         }
      }
      if (lower && upper && lower.getDefiningOp() && upper.getDefiningOp() && mlir::isa<db::ConstantOp>(lower.getDefiningOp()) && mlir::isa<db::ConstantOp>(upper.getDefiningOp())) {
         auto lowerInclusive = lowerCmp.getPredicate() == DBCmpPredicate::gte || lowerCmp.getPredicate() == DBCmpPredicate::lte;
         auto upperInclusive = upperCmp.getPredicate() == DBCmpPredicate::gte || upperCmp.getPredicate() == DBCmpPredicate::lte;
         mlir::Value between = rewriter.create<db::BetweenOp>(lowerCmp->getLoc(), current, lower, upper, lowerInclusive, upperInclusive);
         rawValues.erase(lowerCmp);
         rawValues.erase(upperCmp);
         rawValues.insert(between);
      }
   }
   if (rawValues.size() == 1) {
      rewriter.replaceOp(andOp, *rawValues.begin());
      return success();
   }
   if (rawValues.size() != andOp.getVals().size()) {
      rewriter.replaceOpWithNewOp<db::AndOp>(andOp, std::vector<mlir::Value>(rawValues.begin(), rawValues.end()));
      return success();
   }
   return failure();
}
#define GET_OP_CLASSES
#include "lingodb/compiler/Dialect/DB/IR/DBOps.cpp.inc"
#include "lingodb/compiler/Dialect/DB/IR/DBOpsInterfaces.cpp.inc"
