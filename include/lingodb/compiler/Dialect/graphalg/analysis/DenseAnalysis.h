#pragma once

#include <llvm/Support/raw_ostream.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace graphalg {

class Density {
   friend class DensityLattice;

   private:
   enum State {
      /**
     * Initial state where nothing is known about the density of the value.
     */
      UNKNOWN,
      /** The value may or may not be sparse. */
      MAYBE_SPARSE,
      /** The value is known to be dense. */
      DENSE,
   };
   State _state;

   explicit Density(State state) : _state(state) {}

   public:
   explicit Density() : _state(UNKNOWN) {}

   bool operator==(const Density& o) const { return _state == o._state; }

   bool operator!=(const Density& o) const { return !(*this == o); }

   void print(llvm::raw_ostream& os) const;

   static Density join(const Density& lhs, const Density& rhs) {
      if (lhs._state == UNKNOWN) {
         return rhs;
      } else if (rhs._state == UNKNOWN) {
         return lhs;
      } else if (lhs._state == DENSE && rhs._state == DENSE) {
         return Density(DENSE);
      } else {
         return Density(MAYBE_SPARSE);
      }
   }
};

class DensityLattice : public mlir::dataflow::Lattice<Density> {
   private:
   mlir::ChangeResult set(Density::State state) {
      if (getValue()._state == state) {
         return mlir::ChangeResult::NoChange;
      }

      getValue()._state = state;
      return mlir::ChangeResult::Change;
   }

   public:
   using Lattice::Lattice;

   bool isDense() const { return getValue()._state == Density::DENSE; }

   mlir::ChangeResult setDense(bool dense) {
      return set(dense ? Density::DENSE : Density::MAYBE_SPARSE);
   }
};

/** Tracks whether matrices are known to be dense. */
class DenseAnalysis
   : public mlir::dataflow::SparseForwardDataFlowAnalysis<DensityLattice> {
   public:
   using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

   mlir::LogicalResult
   visitOperation(mlir::Operation* op,
                  llvm::ArrayRef<const DensityLattice*>
                     operands,
                  llvm::ArrayRef<DensityLattice*>
                     results) override;
   void
   visitNonControlFlowArguments(mlir::Operation* op,
                                const mlir::RegionSuccessor& successor,
                                llvm::ArrayRef<DensityLattice*>
                                   argLattices,
                                unsigned firstIndex) override;

   void setToEntryState(DensityLattice* lattice) override;
};

class RunDenseAnalysis {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RunDenseAnalysis)

   private:
   mlir::DataFlowSolver _solver;
   bool _failed;

   public:
   RunDenseAnalysis(mlir::func::FuncOp op);
   mlir::LogicalResult status() const { return mlir::failure(_failed); }

   const DensityLattice* getFor(mlir::Value val) const;
};

} // namespace graphalg
