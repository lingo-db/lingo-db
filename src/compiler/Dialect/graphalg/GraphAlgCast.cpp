#include <llvm/ADT/APSInt.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>

#include <lingodb/compiler/Dialect/graphalg/GraphAlgAttr.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgCast.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgInterfaces.h>
#include <lingodb/compiler/Dialect/graphalg/SemiringTypes.h>

namespace graphalg {

static mlir::FloatAttr convertIntegerAttrToFloat(mlir::IntegerAttr attr) {
   llvm::APFloat realVal(llvm::APFloat::IEEEdouble());
   realVal.convertFromAPInt(attr.getValue(),
                            /*isSigned=*/true,
                            llvm::RoundingMode::NearestTiesToEven);
   return mlir::FloatAttr::get(SemiringTypes::forReal(attr.getContext()),
                               realVal);
}

static mlir::TypedAttr intToReal(mlir::TypedAttr attr,
                                 SemiringTypeInterface inRing,
                                 SemiringTypeInterface outRing) {
   return convertIntegerAttrToFloat(llvm::cast<mlir::IntegerAttr>(attr));
}

static mlir::TypedAttr tropIntToTropReal(mlir::TypedAttr attr,
                                         SemiringTypeInterface inRing,
                                         SemiringTypeInterface outRing) {
   auto* ctx = attr.getContext();
   if (llvm::isa<TropInfAttr>(attr)) {
      return TropInfAttr::get(ctx, outRing);
   }

   auto intAttr = llvm::cast<TropIntAttr>(attr).getValue();
   return TropFloatAttr::get(ctx, outRing, convertIntegerAttrToFloat(intAttr));
}

static mlir::IntegerAttr truncateFloatAttrToInteger(mlir::FloatAttr attr) {
   auto realVal = attr.getValue();
   llvm::APSInt intVal(/*BitWidth=*/64, /*isUnsigned=*/false);
   bool isExact;
   realVal.convertToInteger(intVal, llvm::RoundingMode::TowardZero, &isExact);
   return mlir::IntegerAttr::get(SemiringTypes::forInt(attr.getContext()),
                                 intVal);
}

static mlir::TypedAttr realToInt(mlir::TypedAttr attr,
                                 SemiringTypeInterface inRing,
                                 SemiringTypeInterface outRing) {
   return truncateFloatAttrToInteger(llvm::cast<mlir::FloatAttr>(attr));
}

static mlir::TypedAttr tropRealtoTropInt(mlir::TypedAttr attr,
                                         SemiringTypeInterface inRing,
                                         SemiringTypeInterface outRing) {
   auto* ctx = attr.getContext();
   if (llvm::isa<TropInfAttr>(attr)) {
      return TropInfAttr::get(ctx, outRing);
   }

   auto floatAttr = llvm::cast<TropFloatAttr>(attr).getValue();
   return TropIntAttr::get(ctx, outRing, truncateFloatAttrToInteger(floatAttr));
}

void GraphAlgCast::registerCasts(mlir::MLIRContext* ctx) {
   auto boolType = SemiringTypes::forBool(ctx);
   auto intType = SemiringTypes::forInt(ctx);
   auto realType = SemiringTypes::forReal(ctx);
   auto tropIntType = SemiringTypes::forTropInt(ctx);
   auto tropRealType = SemiringTypes::forTropReal(ctx);
   // Note: trop_max_int is not (yet) part of the specification.
   auto tropMaxIntType = SemiringTypes::forTropMaxInt(ctx);
   std::array<mlir::Type, 6> allTypes{
      boolType,
      intType,
      realType,
      tropIntType,
      tropRealType,
      tropMaxIntType,
   };

   // All types can be cast to themselves
   for (auto t : allTypes) {
      _casts[{t, t}] = [](mlir::TypedAttr attr, SemiringTypeInterface inRing,
                          SemiringTypeInterface outRing) { return attr; };
   }

   // Booleans can be cast to and from all semirings
   for (auto t : allTypes) {
      // Cast to bool
      _casts[{t, boolType}] =
         [](mlir::TypedAttr attr, SemiringTypeInterface inRing,
            SemiringTypeInterface outRing) -> mlir::TypedAttr {
         return mlir::BoolAttr::get(attr.getContext(),
                                    attr != inRing.addIdentity());
      };

      // Cast from bool
      _casts[{boolType, t}] = [](mlir::TypedAttr attr,
                                 SemiringTypeInterface inRing,
                                 SemiringTypeInterface outRing) {
         auto boolAttr = llvm::cast<mlir::BoolAttr>(attr);
         return boolAttr.getValue() ? outRing.mulIdentity() : outRing.addIdentity();
      };
   }

   _casts[{intType, realType}] = intToReal;
   _casts[{tropIntType, tropRealType}] = tropIntToTropReal;
   _casts[{realType, intType}] = realToInt;
   _casts[{tropRealType, tropIntType}] = tropRealtoTropInt;

   // Note: casts below are not part of the specification (yet)
   _casts[{intType, tropMaxIntType}] =
      [](mlir::TypedAttr attr, SemiringTypeInterface inRing,
         SemiringTypeInterface outRing) -> mlir::TypedAttr {
      auto intAttr = llvm::cast<mlir::IntegerAttr>(attr);
      if (intAttr == inRing.addIdentity()) {
         // Maintain additive identity property.
         return outRing.addIdentity();
      }

      return TropIntAttr::get(attr.getContext(), outRing, intAttr);
   };

   _casts[{tropMaxIntType, intType}] =
      [](mlir::TypedAttr attr, SemiringTypeInterface inRing,
         SemiringTypeInterface outRing) -> mlir::TypedAttr {
      if (attr == inRing.addIdentity()) {
         // Maintain additive identity property
         return outRing.addIdentity();
      }

      auto intAttr = llvm::cast<TropIntAttr>(attr);
      return intAttr.getValue();
   };
}

bool GraphAlgCast::isCastLegal(mlir::Type from, mlir::Type to) const {
   return _casts.contains({from, to});
}

mlir::TypedAttr GraphAlgCast::castAttribute(mlir::TypedAttr attr,
                                            mlir::Type to) const {
   auto from = attr.getType();
   auto func = _casts.lookup({from, to});
   if (func) {
      return func(attr, llvm::cast<SemiringTypeInterface>(from),
                  llvm::cast<SemiringTypeInterface>(to));
   }

   return nullptr;
}

} // namespace graphalg
