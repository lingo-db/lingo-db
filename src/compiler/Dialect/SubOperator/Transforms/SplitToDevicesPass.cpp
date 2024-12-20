
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnCreationAnalysis.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnUsageAnalysis.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/SubOpDependencyAnalysis.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <queue>
namespace {
using namespace lingodb::compiler::dialect;
/*
   This pass performs:
      1. Device assignment for execution steps.
      2. Insertion of state context changes (data movement ops) between CPU and GPU steps.

   Assumptions:
      - Query IR contains subop::ExecutionGroupOp with subop::ExecutionStepOp.
      - The pass is relevant only if devices (e.g., GPU) are enabled.

   Reasoning & Workflow:
      1. CPU steps produce states that may be used by GPU and CPU steps. If a CPU step result that is a state is used by a GPU step, 
         insert a context switch to convert the resulting state to a GPU device pointer by allocating and sending to GPU before terminator.
      2. GPU steps consume input states (device ptrs) and runtime constructs (data source).
         All state movement is managed by CPU steps, table data movement is managed by the data source (host launches kernels with gpu steps callbacks).
      3. To form the result table, states modified by GPU steps must be switched back to CPU context. 
         If a CPU step uses a state from a GPU step, insert a context switch at the beginning of the CPU step.
*/

class SplitToDevicesPass : public mlir::PassWrapper<SplitToDevicesPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitToDevicesPass)
   virtual llvm::StringRef getArgument() const override { return "subop-split-to-devices"; }

   void decideExecutionStepDevice(subop::ExecutionStepOp step) {
      bool shouldRunOnCpu{false};
      if (!step.isOnCPU()) { return; }
      // assert(step.isOnCPU() && "Expected all steps marked to be executed on CPU before deciding device.");
      // Subops that return simple states, management handles or are involved in result construction must be run on CPU
      for (auto& op : step.getSubOps().front().getOperations()) {
         shouldRunOnCpu |= (isa<subop::GenericCreateOp>(&op) ||
                            isa<subop::CreateFrom>(&op) || // Result construction
                            isa<subop::CreateSimpleStateOp>(&op) ||
                            isa<subop::GenericCreateOp>(&op) || // GrowingBuffer, CreateHashIndexedViewLowering,  creation
                            isa<subop::GetExternalOp>(&op) // Data source retrieval
         );
         const bool materializeResultTable = (isa<subop::MaterializeOp>(&op) && mlir::isa<subop::ResultTableType>(op.getOperand(1).getType()));
         shouldRunOnCpu |= materializeResultTable;
      }
      if (!shouldRunOnCpu) {
         step.setDeviceType(subop::DeviceType::GPU);
      }
   }

   void runOnOperation() override {
      subop::ColumnUsageAnalysis usedColumns(getOperation());
      subop::ColumnCreationAnalysis createdColumns(getOperation());
      std::vector<mlir::Operation*> opsToErase;

      // Phase 1: decide on step executor device
      getOperation()->walk([&](subop::ExecutionGroupOp executionGroupOp) {
         executionGroupOp->walk<mlir::WalkOrder::PreOrder>([&](subop::ExecutionStepOp executionStepOp) {
            decideExecutionStepDevice(executionStepOp);
         });
      });
      // Phase 2: add data movement for steps that change context
      auto* ctxt = getOperation()->getContext();
      mlir::OpBuilder builder(ctxt);
      llvm::DenseSet<mlir::Value> gpuStates;
      // Adjust CPU results: if a result state is used by a GPU step, then it should be a device ptr (GPU steps do not move data nor produce states). Move to GPU.
      getOperation()->walk([&](subop::ExecutionGroupOp executionGroupOp) {
         executionGroupOp->walk<mlir::WalkOrder::PreOrder>([&](subop::ExecutionStepOp executionStepOp) {
            if (executionStepOp.isOnCPU()) {
               // A data source should not be moved to GPU.
               bool returnsExternalTable{false};
               for (auto res : executionStepOp.getResults()) {
                  if (isa<subop::TableType>(res.getType())) {
                     returnsExternalTable = true;
                     break;
                  }
               }
               if (!returnsExternalTable) {
                  auto returnOp = mlir::cast<subop::ExecutionStepReturnOp>(executionStepOp.getSubOps().front().getTerminator());
                  builder.setInsertionPoint(returnOp);
                  for (auto [terminatorIn, stepOut] : llvm::zip(returnOp->getOperands(), executionStepOp->getResults())) {
                     for (auto* stateUser : stepOut.getUsers()) {
                        if (auto stateUserStep = mlir::dyn_cast<subop::ExecutionStepOp>(stateUser)) { // can also be the last step, the user will be group terminator
                           if (stateUserStep.isOnGPU()) {
                              mlir::Operation* changedContextOp = builder.create<subop::StateContextSwitchOp>(returnOp->getLoc(), terminatorIn.getType(), terminatorIn, subop::DataMovementDirection::toDevice);
                              terminatorIn.replaceAllUsesExcept(changedContextOp->getResult(0), changedContextOp);
                              gpuStates.insert(stepOut);
                              break; // we need at least one GPU user
                           }
                        }
                     }
                  }
               }
            }
         });
      });
      // Adjust CPU steps' inputs: if any input state was used by a GPU step, then it is a device ptr (GPU steps do not move data nor produce states). Move to CPU.
      getOperation()->walk([&](subop::ExecutionGroupOp executionGroupOp) {
         executionGroupOp->walk<mlir::WalkOrder::PreOrder>([&](subop::ExecutionStepOp executionStepOp) {
            if (executionStepOp.isOnCPU()) {
               builder.setInsertionPointToStart(&executionStepOp.getSubOps().front());
               for (auto [subOpOperand, blockArg] : llvm::zip(executionStepOp.getOperands(), executionStepOp.getSubOps().front().getArguments())) { // assume direct mapping
                  if (gpuStates.contains(subOpOperand)) {
                     mlir::Operation* changedContextOp = builder.create<subop::StateContextSwitchOp>(builder.getUnknownLoc(), blockArg.getType(), blockArg, subop::DataMovementDirection::fromDevice);
                     blockArg.replaceAllUsesExcept(changedContextOp->getResult(0), changedContextOp);
                  }
               }
            }
         });
      });
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
subop::createSplitToDevicesPass() { return std::make_unique<SplitToDevicesPass>(); }