#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/Pass.h>

#include <lingodb/compiler/Dialect/graphalg/GraphAlgInterfaces.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgOps.h>
#include <lingodb/compiler/Dialect/graphalg/analysis/DenseAnalysis.h>
#include <lingodb/compiler/Dialect/graphalg/util/PrintDataFlowLatticeTestPass.h>

namespace graphalg {

void Density::print(llvm::raw_ostream& os) const {
   switch (_state) {
      case UNKNOWN:
         os << "unknown";
         return;
      case MAYBE_SPARSE:
         os << "maybe_sparse";
         return;
      case DENSE:
         os << "dense";
         return;
   }
}

mlir::LogicalResult
DenseAnalysis::visitOperation(mlir::Operation* op,
                              llvm::ArrayRef<const DensityLattice*>
                                 operands,
                              llvm::ArrayRef<DensityLattice*>
                                 results) {
   llvm::SmallVector<mlir::ChangeResult> changedResults(results.size());
   if (auto iface = llvm::dyn_cast<InferDensityInterface>(op)) {
      iface.inferDensity(operands, results, changedResults);
      for (auto [result, changed] : llvm::zip_equal(results, changedResults)) {
         propagateIfChanged(result, changed);
      }

      return mlir::success();
   }

   // Assume potentially sparse.
   for (auto* r : results) {
      propagateIfChanged(r, r->setDense(false));
   }

   return mlir::success();
}

void DenseAnalysis::visitNonControlFlowArguments(
   mlir::Operation* op, const mlir::RegionSuccessor& successor,
   llvm::ArrayRef<DensityLattice*> argLattices, unsigned firstIndex) {
   if (llvm::isa<ForConstOp, ForDimOp>(op)) {
      // Iteration counter is dense.
      assert(firstIndex == 1);
      auto arg = argLattices[0];
      propagateIfChanged(arg, arg->setDense(true));
      return;
   }

   // Assume potentially sparse.
   for (auto* a : argLattices.take_front(firstIndex)) {
      propagateIfChanged(a, a->setDense(false));
   }
}

void DenseAnalysis::setToEntryState(DensityLattice* lattice) {
   lattice->getValue() = Density();
}

RunDenseAnalysis::RunDenseAnalysis(mlir::func::FuncOp op) {
   // Run dead code analysis to mark nested regions as live
   _solver.load<mlir::dataflow::DeadCodeAnalysis>();
   // Dead code analysis has an accidental dependency on this analysis.
   _solver.load<mlir::dataflow::SparseConstantPropagation>();
   _solver.load<DenseAnalysis>();
   _failed = mlir::failed(_solver.initializeAndRun(op));
}

const DensityLattice* RunDenseAnalysis::getFor(mlir::Value val) const {
   return _solver.lookupState<DensityLattice>(val);
}

namespace {

struct TestDensePass
   : public mlir::PassWrapper<
        TestDensePass,
        PrintDataFlowLatticeTestPass<RunDenseAnalysis, mlir::func::FuncOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDensePass)

   llvm::StringRef getArgument() const final { return "test-print-dense"; }
   llvm::StringRef getDescription() const final {
      return "Print the contents of a density analysis.";
   }
};

} // namespace

void registerTestDensePass() { mlir::PassRegistration<TestDensePass>(); }

} // namespace graphalg
