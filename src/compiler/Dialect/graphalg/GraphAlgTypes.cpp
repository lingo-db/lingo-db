#include <llvm/ADT/APInt.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Support/LogicalResult.h>

#include <lingodb/compiler/Dialect/graphalg/GraphAlgAttr.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgDialect.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgInterfaces.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgTypes.h>
#include <lingodb/compiler/Dialect/graphalg/SemiringTypes.h>

#define GET_TYPEDEF_CLASSES
#include "lingodb/compiler/Dialect/graphalg/GraphAlgOpsTypes.cpp.inc"

namespace graphalg {

MatrixType MatrixType::scalarOf(mlir::Type semiring) {
   auto* ctx = semiring.getContext();
   auto dim1 = DimAttr::getOne(ctx);
   return MatrixType::get(ctx, dim1, dim1, semiring);
}

bool MatrixType::isScalar() const { return isRowVector() && isColumnVector(); }

bool MatrixType::isRowVector() const { return getRows().isOne(); }

bool MatrixType::isColumnVector() const { return getCols().isOne(); }

bool MatrixType::isBoolean() const {
   return getSemiring() == SemiringTypes::forBool(getContext());
}

std::pair<DimAttr, DimAttr> MatrixType::getDims() const {
   return {getRows(), getCols()};
}

MatrixType MatrixType::asScalar() { return scalarOf(getSemiring()); }

MatrixType MatrixType::withSemiring(mlir::Type semiring) {
   return MatrixType::get(getContext(), getRows(), getCols(), semiring);
}

mlir::LogicalResult
MatrixType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                   graphalg::DimAttr rows, graphalg::DimAttr cols,
                   mlir::Type semiring) {
   if (!llvm::isa<SemiringTypeInterface>(semiring)) {
      return emitError() << semiring << " is not a semiring";
   }

   return mlir::success();
}

mlir::TypedAttr TropI64Type::addIdentity() const {
   return TropInfAttr::get(getContext(), *this);
}

mlir::TypedAttr TropI64Type::add(mlir::TypedAttr lhs,
                                 mlir::TypedAttr rhs) const {
   if (llvm::isa<TropInfAttr>(lhs)) {
      return rhs;
   } else if (llvm::isa<TropInfAttr>(rhs)) {
      return lhs;
   }

   auto lhsVal = llvm::cast<TropIntAttr>(lhs).getValue().getValue();
   auto rhsVal = llvm::cast<TropIntAttr>(rhs).getValue().getValue();
   auto resVal = llvm::APIntOps::smin(lhsVal, rhsVal);
   auto intType = SemiringTypes::forInt(getContext());
   return TropIntAttr::get(getContext(), *this,
                           mlir::IntegerAttr::get(intType, resVal));
}

mlir::TypedAttr TropI64Type::mulIdentity() const {
   return TropIntAttr::get(
      getContext(), *this,
      mlir::IntegerAttr::get(SemiringTypes::forInt(getContext()), 0));
}

mlir::TypedAttr TropI64Type::mul(mlir::TypedAttr lhs,
                                 mlir::TypedAttr rhs) const {
   if (llvm::isa<TropInfAttr>(lhs)) {
      return lhs;
   } else if (llvm::isa<TropInfAttr>(rhs)) {
      return rhs;
   }

   auto lhsVal = llvm::cast<TropIntAttr>(lhs).getValue().getValue();
   auto rhsVal = llvm::cast<TropIntAttr>(rhs).getValue().getValue();
   auto resVal = lhsVal + rhsVal;
   auto intType = SemiringTypes::forInt(getContext());
   return TropIntAttr::get(getContext(), *this,
                           mlir::IntegerAttr::get(intType, resVal));
}

mlir::TypedAttr TropF64Type::addIdentity() const {
   return TropInfAttr::get(getContext(), *this);
}

mlir::TypedAttr TropF64Type::add(mlir::TypedAttr lhs,
                                 mlir::TypedAttr rhs) const {
   if (llvm::isa<TropInfAttr>(lhs)) {
      return rhs;
   } else if (llvm::isa<TropInfAttr>(rhs)) {
      return lhs;
   }

   auto lhsVal = llvm::cast<TropFloatAttr>(lhs).getValue().getValue();
   auto rhsVal = llvm::cast<TropFloatAttr>(rhs).getValue().getValue();
   auto resVal = lhsVal < rhsVal ? lhsVal : rhsVal;
   auto realType = SemiringTypes::forReal(getContext());
   return TropFloatAttr::get(getContext(), *this,
                             mlir::FloatAttr::get(realType, resVal));
}

mlir::TypedAttr TropF64Type::mulIdentity() const {
   return TropFloatAttr::get(
      getContext(), *this,
      mlir::FloatAttr::get(SemiringTypes::forReal(getContext()), 0));
}

mlir::TypedAttr TropF64Type::mul(mlir::TypedAttr lhs,
                                 mlir::TypedAttr rhs) const {
   if (llvm::isa<TropInfAttr>(lhs)) {
      return lhs;
   } else if (llvm::isa<TropInfAttr>(rhs)) {
      return rhs;
   }

   auto lhsVal = llvm::cast<TropFloatAttr>(lhs).getValue().getValue();
   auto rhsVal = llvm::cast<TropFloatAttr>(rhs).getValue().getValue();
   auto resVal = lhsVal + rhsVal;
   auto realType = SemiringTypes::forReal(getContext());
   return TropFloatAttr::get(getContext(), *this,
                             mlir::FloatAttr::get(realType, resVal));
}

mlir::TypedAttr TropMaxI64Type::addIdentity() const {
   return TropInfAttr::get(getContext(), *this);
}

mlir::TypedAttr TropMaxI64Type::add(mlir::TypedAttr lhs,
                                    mlir::TypedAttr rhs) const {
   if (llvm::isa<TropInfAttr>(lhs)) {
      return rhs;
   } else if (llvm::isa<TropInfAttr>(rhs)) {
      return lhs;
   }

   auto lhsVal = llvm::cast<TropIntAttr>(lhs).getValue().getValue();
   auto rhsVal = llvm::cast<TropIntAttr>(rhs).getValue().getValue();
   auto resVal = llvm::APIntOps::smax(lhsVal, rhsVal);
   auto intType = SemiringTypes::forInt(getContext());
   return TropIntAttr::get(getContext(), *this,
                           mlir::IntegerAttr::get(intType, resVal));
}

mlir::TypedAttr TropMaxI64Type::mulIdentity() const {
   return TropIntAttr::get(
      getContext(), *this,
      mlir::IntegerAttr::get(SemiringTypes::forInt(getContext()), 0));
}

mlir::TypedAttr TropMaxI64Type::mul(mlir::TypedAttr lhs,
                                    mlir::TypedAttr rhs) const {
   if (llvm::isa<TropInfAttr>(lhs)) {
      return lhs;
   } else if (llvm::isa<TropInfAttr>(rhs)) {
      return rhs;
   }

   auto lhsVal = llvm::cast<TropIntAttr>(lhs).getValue().getValue();
   auto rhsVal = llvm::cast<TropIntAttr>(rhs).getValue().getValue();
   auto resVal = lhsVal + rhsVal;
   auto intType = SemiringTypes::forInt(getContext());
   return TropIntAttr::get(getContext(), *this,
                           mlir::IntegerAttr::get(intType, resVal));
}

namespace {

struct IntegerSemiringInterface
   : public SemiringTypeInterface::ExternalModel<IntegerSemiringInterface,
                                                 mlir::IntegerType> {
   static inline mlir::TypedAttr addIdentity(::mlir::Type type) {
      auto* ctx = type.getContext();
      if (type == SemiringTypes::forBool(ctx)) {
         return mlir::BoolAttr::get(ctx, false);
      } else {
         return mlir::IntegerAttr::get(type, 0);
      }
   }

   static inline mlir::TypedAttr add(::mlir::Type type, mlir::TypedAttr lhs,
                                     mlir::TypedAttr rhs) {
      auto* ctx = type.getContext();
      if (type == SemiringTypes::forBool(ctx)) {
         auto lhsVal = llvm::cast<mlir::BoolAttr>(lhs).getValue();
         auto rhsVal = llvm::cast<mlir::BoolAttr>(rhs).getValue();
         return mlir::BoolAttr::get(ctx, lhsVal || rhsVal);
      } else {
         auto lhsVal = llvm::cast<mlir::IntegerAttr>(lhs).getValue();
         auto rhsVal = llvm::cast<mlir::IntegerAttr>(rhs).getValue();
         return mlir::IntegerAttr::get(type, lhsVal + rhsVal);
      }
   }

   static inline mlir::TypedAttr mulIdentity(::mlir::Type type) {
      auto* ctx = type.getContext();
      if (type == SemiringTypes::forBool(ctx)) {
         return mlir::BoolAttr::get(ctx, true);
      } else {
         return mlir::IntegerAttr::get(type, 1);
      }
   }

   static inline mlir::TypedAttr mul(::mlir::Type type, mlir::TypedAttr lhs,
                                     mlir::TypedAttr rhs) {
      auto* ctx = type.getContext();
      if (type == SemiringTypes::forBool(ctx)) {
         auto lhsVal = llvm::cast<mlir::BoolAttr>(lhs).getValue();
         auto rhsVal = llvm::cast<mlir::BoolAttr>(rhs).getValue();
         return mlir::BoolAttr::get(ctx, lhsVal && rhsVal);
      } else {
         auto lhsVal = llvm::cast<mlir::IntegerAttr>(lhs).getValue();
         auto rhsVal = llvm::cast<mlir::IntegerAttr>(rhs).getValue();
         return mlir::IntegerAttr::get(type, lhsVal * rhsVal);
      }
   }
};

struct F64SemiringInterface
   : public SemiringTypeInterface::ExternalModel<F64SemiringInterface,
                                                 mlir::Float64Type> {
   static inline mlir::TypedAttr addIdentity(::mlir::Type type) {
      return mlir::FloatAttr::get(type, 0);
   }

   static inline mlir::TypedAttr add(::mlir::Type type, mlir::TypedAttr lhs,
                                     mlir::TypedAttr rhs) {
      auto v = llvm::cast<mlir::FloatAttr>(lhs).getValue() +
         llvm::cast<mlir::FloatAttr>(rhs).getValue();
      return mlir::FloatAttr::get(type, v);
   }

   static inline mlir::TypedAttr mulIdentity(::mlir::Type type) {
      return mlir::FloatAttr::get(type, 1);
   }

   static inline mlir::TypedAttr mul(::mlir::Type type, mlir::TypedAttr lhs,
                                     mlir::TypedAttr rhs) {
      auto v = llvm::cast<mlir::FloatAttr>(lhs).getValue() *
         llvm::cast<mlir::FloatAttr>(rhs).getValue();
      return mlir::FloatAttr::get(type, v);
   }
};

} // namespace

// Need to define this here to avoid depending on IPRTypes in
// IPRDialect and creating a cycle.
void GraphAlgDialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "lingodb/compiler/Dialect/graphalg/GraphAlgOpsTypes.cpp.inc"
      >();

   mlir::IntegerType::attachInterface<IntegerSemiringInterface>(*getContext());
   mlir::Float64Type::attachInterface<F64SemiringInterface>(*getContext());
}

} // namespace graphalg
