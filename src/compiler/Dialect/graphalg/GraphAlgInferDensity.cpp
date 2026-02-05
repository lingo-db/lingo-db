#include "lingodb/compiler/Dialect/graphalg/GraphAlgOps.h"
#include "lingodb/compiler/Dialect/graphalg/analysis/DenseAnalysis.h"

namespace graphalg {

// Used in DenseResult trait
void markAllDense(llvm::ArrayRef<DensityLattice*> results,
                  llvm::MutableArrayRef<mlir::ChangeResult>
                     changedResults) {
   for (auto [i, r] : llvm::enumerate(results)) {
      changedResults[i] |= r->setDense(true);
   }
}

static bool isDense(const DensityLattice* l) { return l->isDense(); }

// Used in PropagatesDense trait
void markDenseIfAllInputsDense(
   llvm::ArrayRef<const DensityLattice*> operands,
   llvm::ArrayRef<DensityLattice*>
      results,
   llvm::MutableArrayRef<mlir::ChangeResult>
      changedResults) {
   auto allDense = llvm::all_of(operands, isDense);
   for (auto [i, r] : llvm::enumerate(results)) {
      changedResults[i] |= r->setDense(allDense);
   }
}

void ElementWiseAddOp::inferDensity(
   llvm::ArrayRef<const DensityLattice*> operands,
   llvm::ArrayRef<DensityLattice*>
      results,
   llvm::MutableArrayRef<mlir::ChangeResult>
      changedResults) {
   auto anyDense = llvm::any_of(operands, isDense);
   changedResults[0] |= results[0]->setDense(anyDense);
}

} // namespace graphalg
