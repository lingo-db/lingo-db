#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LogicalResult.h>

#include <lingodb/compiler/Dialect/graphalg/GraphAlgAttr.h>
#include <lingodb/compiler/Dialect/graphalg/analysis/DenseAnalysis.h>

#include "lingodb/compiler/Dialect/graphalg/GraphAlgTypeInterfaces.h.inc"

namespace graphalg {

#include "lingodb/compiler/Dialect/graphalg/GraphAlgOpInterfaces.h.inc"

/**
 * Marks operations as belonging to the 'Core' subset of the graphalg language.
 *
 * All non-Core ops are desugared to Core ops in the \c GraphAlgToCore pass.
 */
template <typename T>
class IsCore : public mlir::OpTrait::TraitBase<T, IsCore> {};

mlir::LogicalResult verifySameOperandsAndResultSemiring(mlir::Operation* op);

template <typename ConcreteType>
class SameOperandsAndResultSemiring
   : public mlir::OpTrait::TraitBase<ConcreteType,
                                     SameOperandsAndResultSemiring> {
   public:
   static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
      return verifySameOperandsAndResultSemiring(op);
   }
};

void markAllDense(llvm::ArrayRef<DensityLattice*> results,
                  llvm::MutableArrayRef<mlir::ChangeResult>
                     changedResults);

/** Op results are always dense matrices */
template <typename T>
class DenseResult : public mlir::OpTrait::TraitBase<T, DenseResult> {
   public:
   void inferDensity(llvm::ArrayRef<const DensityLattice*> operands,
                     llvm::ArrayRef<DensityLattice*>
                        results,
                     llvm::MutableArrayRef<mlir::ChangeResult>
                        changedResults) {
      markAllDense(results, changedResults);
   }
};

void markDenseIfAllInputsDense(
   llvm::ArrayRef<const DensityLattice*> operands,
   llvm::ArrayRef<DensityLattice*>
      results,
   llvm::MutableArrayRef<mlir::ChangeResult>
      changedResults);

/** Op results are dense iff ALL inputs are dense */
template <typename T>
class PropagatesDense : public mlir::OpTrait::TraitBase<T, PropagatesDense> {
   public:
   void inferDensity(llvm::ArrayRef<const DensityLattice*> operands,
                     llvm::ArrayRef<DensityLattice*>
                        results,
                     llvm::MutableArrayRef<mlir::ChangeResult>
                        changedResults) {
      markDenseIfAllInputsDense(operands, results, changedResults);
   }
};

} // namespace graphalg
