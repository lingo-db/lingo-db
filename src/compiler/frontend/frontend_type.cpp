#include "lingodb/compiler/frontend/frontend_type.h"

#include "lingodb/catalog/Types.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnCreationAnalysis.h"

namespace lingodb {
NullableType::NullableType(catalog::Type type) : type(type), isNullable(false) {
}
NullableType::NullableType(catalog::Type type, bool isNullable) : type(type), isNullable(isNullable) {
}
mlir::Type NullableType::toMlirType(mlir::MLIRContext* context) const {
   mlir::Type t = type.getMLIRTypeCreator()->createType(context);
   assert((this->type.getTypeId() == catalog::LogicalTypeId::NONE && isNullable) || this->type.getTypeId() != catalog::LogicalTypeId::NONE);
   if (isNullable) {
      return compiler::dialect::db::NullableType::get(context, t);
   }
   return t;
}
mlir::Value NullableType::castValueToThisType(mlir::OpBuilder& builder, mlir::Value valueToCast, bool valueNullable) const {
   auto type = toMlirType(builder.getContext());
   if (!isNullable && valueNullable) {
      //If type is not nullable but value is
      type = compiler::dialect::db::NullableType::get(builder.getContext(), type);
   }

   bool onlyTargetIsNullable = !valueNullable && isNullable;
   if (valueToCast.getType() == type) { return valueToCast; }

   if (auto* defOp = valueToCast.getDefiningOp()) {
      if (auto constOp = mlir::dyn_cast_or_null<compiler::dialect::db::ConstantOp>(defOp)) {
         if (!mlir::isa<compiler::dialect::db::NullableType>(type)) {
            constOp.getResult().setType(type);
            return constOp;
         }
      }
      if (auto nullOp = mlir::dyn_cast_or_null<compiler::dialect::db::NullOp>(defOp)) {
         auto t2 = mlir::cast<compiler::dialect::db::NullableType>(type);
         nullOp.getResult().setType(t2);
         return nullOp;
      }
   }

   if (valueToCast.getType() == getBaseType(type)) {
      return builder.create<compiler::dialect::db::AsNullableOp>(builder.getUnknownLoc(), type, valueToCast);
   }

   if (onlyTargetIsNullable) {
      mlir::Value casted = builder.create<compiler::dialect::db::CastOp>(builder.getUnknownLoc(), getBaseType(type), valueToCast);
      return builder.create<compiler::dialect::db::AsNullableOp>(builder.getUnknownLoc(), type, casted);
   } else {
      return builder.create<compiler::dialect::db::CastOp>(builder.getUnknownLoc(), type, valueToCast);
   }
}
mlir::Value NullableType::castValue(mlir::OpBuilder& builder, mlir::Value valueToCast) const {
   if (castType == nullptr) {
      return valueToCast;
   }
   auto type = castType->toMlirType(builder.getContext());
   if (!castType->isNullable && isNullable) {
      //If type is not nullable but value is
      type = compiler::dialect::db::NullableType::get(builder.getContext(), type);
   }

   bool onlyTargetIsNullable = !isNullable && castType->isNullable;
   if (valueToCast.getType() == type) { return valueToCast; }

   if (auto* defOp = valueToCast.getDefiningOp()) {
      if (auto constOp = mlir::dyn_cast_or_null<compiler::dialect::db::ConstantOp>(defOp)) {
         if (!mlir::isa<compiler::dialect::db::NullableType>(type)) {
            constOp.getResult().setType(type);
            return constOp;
         }
      }
      if (auto nullOp = mlir::dyn_cast_or_null<compiler::dialect::db::NullOp>(defOp)) {
         auto t2 = mlir::cast<compiler::dialect::db::NullableType>(type);
         nullOp.getResult().setType(t2);
         return nullOp;
      }
   }

   if (valueToCast.getType() == getBaseType(type)) {
      return builder.create<compiler::dialect::db::AsNullableOp>(builder.getUnknownLoc(), type, valueToCast);
   }

   if (onlyTargetIsNullable) {
      mlir::Value casted = builder.create<compiler::dialect::db::CastOp>(builder.getUnknownLoc(), getBaseType(type), valueToCast);
      return builder.create<compiler::dialect::db::AsNullableOp>(builder.getUnknownLoc(), type, casted);
   } else {
      return builder.create<compiler::dialect::db::CastOp>(builder.getUnknownLoc(), type, valueToCast);
   }
}

bool NullableType::isNumeric() const {
   return type.getTypeId() == catalog::LogicalTypeId::DOUBLE || type.getTypeId() == catalog::LogicalTypeId::DECIMAL || type.getTypeId() == catalog::LogicalTypeId::INT;
}
bool NullableType::operator==(NullableType& other) {
   return this->type.getTypeId() == other.type.getTypeId() && this->isNullable == other.isNullable;
}
bool NullableType::operator!=(NullableType& other) {
   if (this->isNullable != other.isNullable || this->type.getTypeId() != other.type.getTypeId()) {
      return true;
   }
   switch (this->type.getTypeId()) {
      case catalog::LogicalTypeId::INT: {
         auto info = this->type.getInfo<catalog::IntTypeInfo>();
         auto info2 = other.type.getInfo<catalog::IntTypeInfo>();
         return info->getBitWidth() != info2->getBitWidth() || info->getIsSigned() != info2->getIsSigned();
      }
      case catalog::LogicalTypeId::DECIMAL: {
         auto info = this->type.getInfo<catalog::DecimalTypeInfo>();
         auto info2 = other.type.getInfo<catalog::DecimalTypeInfo>();
         return info->getPrecision() != info2->getPrecision() || info->getScale() != info2->getScale();
      }
      default: return false;
   }
}
}