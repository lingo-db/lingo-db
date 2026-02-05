#pragma once

#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>

namespace graphalg {

class SemiringTypeInterface;

/** Implements the rules for casting between different semirings. */
class GraphAlgCast {
   private:
   using CastFunction = mlir::TypedAttr (*)(mlir::TypedAttr,
                                            SemiringTypeInterface,
                                            SemiringTypeInterface);

   llvm::SmallDenseMap<std::pair<mlir::Type, mlir::Type>, CastFunction> _casts;

   public:
   /**
   * Initializes the set of valid casts.
   *
   * Should be called once before using the other methods of this class.
   */
   void registerCasts(mlir::MLIRContext* ctx);

   /**
   * Whether a value in semiring \c from can be cast to a value in semiring
   * \c to.
   */
   bool isCastLegal(mlir::Type from, mlir::Type to) const;

   /**
   * Casts \c attr to the target type \c to.
   *
   * NOTE: returns \c nullptr on failure.
   */
   mlir::TypedAttr castAttribute(mlir::TypedAttr attr, mlir::Type to) const;
};

} // namespace graphalg
