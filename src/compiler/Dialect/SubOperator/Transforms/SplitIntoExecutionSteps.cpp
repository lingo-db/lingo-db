#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/SubOpDependencyAnalysis.h"

#include "llvm/Support/Debug.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

#include <queue>
namespace {
using namespace lingodb::compiler::dialect;

class SplitIntoExecutionSteps : public mlir::PassWrapper<SplitIntoExecutionSteps, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitIntoExecutionSteps)
   virtual llvm::StringRef getArgument() const override { return "subop-split-into-steps"; }

   void runOnOperation() override {
      // Step 1: split into different streams
      getOperation()->walk([&](subop::ExecutionGroupOp executionGroup) { // we walk over execution groups (a.k.a queries)
         llvm::DenseMap<mlir::Operation*, std::vector<mlir::Operation*>> steps; // maps step to its ops
         llvm::DenseMap<mlir::Operation*, mlir::Operation*> opToStep; // reverse mapping, each op is mapped to a step
         for (mlir::Operation& op : executionGroup.getSubOps().front()) { // for each operation of an execution group
            if (mlir::isa<subop::ExecutionGroupReturnOp>(op)) {
               continue;
            }
            mlir::Operation* beforeInStream = nullptr;
            for (auto operand : op.getOperands()) {
               if (mlir::isa<tuples::TupleStreamType>(operand.getType())) {
                  if (auto* producer = operand.getDefiningOp()) {
                     assert(!beforeInStream);
                     beforeInStream = producer;
                  }
               }
            }
            if (beforeInStream) { // if there is a producer, we have a dependency
               steps[opToStep[beforeInStream]].push_back(&op); // map the current op to the parent step of a producer
               opToStep[&op] = opToStep[beforeInStream]; // and map the producer to the current op

            } else { // if the op doesn't take tuple streams with a producer, the op is itself a producer
               opToStep[&op] = &op;
               steps[&op].push_back(&op);
            }
         }
         // Step 2: collect required/produced state for each step
         // -> also deal with GetLocal operations (that do belong to the same step that accesses the state)
         llvm::DenseMap<mlir::Operation*, std::vector<std::tuple<mlir::Value, mlir::Value>>> requiredState;
         llvm::DenseMap<mlir::Operation*, std::vector<mlir::Value>> producedState;
         enum Kind {
            READ,
            WRITE
         };

         llvm::DenseMap<subop::Member, std::vector<std::tuple<subop::SubOperator, mlir::Operation*, Kind>>> memberUsage;
         for (auto& step : steps) {
            for (auto* op : step.second) {
               for (auto result : op->getResults()) {
                  if (!mlir::isa<tuples::TupleStreamType>(result.getType())) {
                     producedState[step.first].push_back(result);
                  }
               }
               op->walk([&](mlir::Operation* nestedOp) {
                  if (subop::SubOperator potentialSubOp = mlir::dyn_cast_or_null<subop::SubOperator>(nestedOp)) {
                     auto readMembers = potentialSubOp.getReadMembers();
                     auto writtenMembers = potentialSubOp.getWrittenMembers();
                     for (auto member : readMembers) {
                        memberUsage[member].push_back({potentialSubOp, op, READ});
                     }
                     for (auto member : writtenMembers) {
                        memberUsage[member].push_back({potentialSubOp, op, WRITE});
                     }
                  }
                  for (auto operand : nestedOp->getOperands()) {
                     //todo:: refine
                     if (auto* producer = operand.getDefiningOp()) {
                        if (producer->getBlock() == op->getBlock()) {
                           if (mlir::isa<tuples::TupleStreamType>(operand.getType())) {
                              continue;
                           }
                           requiredState[step.first].push_back({operand, operand}); // and it is not a tuple stream, then we require these states for the step.
                        }
                     }
                  }
               });
            }
         }
         llvm::DenseMap<mlir::Operation*, std::unordered_set<mlir::Operation*>> dependencies; // tracks which steps depend on other steps
         for (auto& [step, vals] : requiredState) { // for each required state/operand of some operation
            for (auto val : vals) {
               if (auto* producer = std::get<0>(val).getDefiningOp()) {
                  if (producer->getBlock() == step->getBlock()) { // if a state is produced in the same block as the step
                     auto* producerStep = opToStep[producer];
                     if (producerStep != step) { // and the producer of the state is not the current step
                        dependencies[step].insert(producerStep); // insert dependency of the current step on the producer step. (e.g., hashIndexedView arg of probe depends on the build step)
                     }
                  }
               }
            }
         }

         // Step 3: determine correct order of steps
         for (auto [member, ops] : memberUsage) {
            for (size_t i = 0; i < ops.size(); i++) {
               for (size_t j = i + 1; j < ops.size(); j++) { // for each pair of subops that access the same member
                  auto* pipelineOp1 = std::get<1>(ops[i]);
                  auto* pipelineOp2 = std::get<1>(ops[j]);
                  auto kind1 = std::get<2>(ops[i]);
                  auto kind2 = std::get<2>(ops[j]);
                  auto addConflict = [&]() {
                     auto* step1 = opToStep[pipelineOp1]; // get pipeline start
                     auto* step2 = opToStep[pipelineOp2];
                     assert(step1);
                     assert(step2);
                     if (step1 == step2) {
                        return;
                     }
                     if (pipelineOp1->isBeforeInBlock(pipelineOp2)) { // if operation1 comes before operation2, operation2 depends on operation1
                        dependencies[step2].insert(step1);
                     } else {
                        dependencies[step1].insert(step2);
                     }
                  };
                  if (kind1 == WRITE && kind2 == WRITE) { // if we have write after write dependency, we add a conflict.
                     addConflict();
                  }
                  if ((kind1 == WRITE && kind2 == READ) || (kind1 == READ && kind2 == WRITE)) {
                     addConflict();
                  }
               }
            }
         }

         // Step 4: create ExecutionStepOps in correct order and handle states
         llvm::DenseMap<mlir::Operation*, size_t> dependCount; // each pipeline has a dependency count
         std::queue<mlir::Operation*> queue; // first come steps with none dependencies, each "resolved step order" reduces dependency count of dependent steps
         llvm::DenseMap<mlir::Value, mlir::Value> stateMapping; // map value from original producers to the producer in the current step

         for (auto& [step, ops] : steps) {
            dependCount[step] = dependencies[step].size();
            if (dependCount[step] == 0) { // if no dependencies step is immediately added to queue
               queue.push(step);
            }
         }
         llvm::DenseMap<mlir::Operation*, std::unordered_set<mlir::Operation*>> inverseDependencies;
         for (auto& [step, stepDependsOn] : dependencies) {
            for (auto* dominantStep : stepDependsOn) {
               inverseDependencies[dominantStep].insert(step); // map a step to steps that should finish before
            }
         }

         while (!queue.empty()) {
            auto* currRoot = queue.front();
            queue.pop();
            for (auto* otherRoot : inverseDependencies[currRoot]) { // for each reverse dependency of the step. (e.g., for each probe that depends on a build)
               if (dependCount[otherRoot] > 0 && otherRoot != currRoot) {
                  dependCount[otherRoot]--; // decrease the dependency count (e.g., for the first step any depending step will have one less dependency)
                  if (dependCount[otherRoot] == 0) {
                     queue.push(otherRoot); // if a step has no more dependencies, then it is next to be processed
                  }
               }
            }
            std::vector<mlir::Type> returnTypes;
            for (auto produced : producedState[currRoot]) { // Collect result (state) types of a step.
               returnTypes.push_back(produced.getType());
            }
            mlir::OpBuilder outerBuilder(&getContext());
            outerBuilder.setInsertionPoint(executionGroup.getSubOps().front().getTerminator());
            std::vector<mlir::Value> inputs;
            std::vector<mlir::Value> blockArgs;
            llvm::SmallVector<bool> threadLocal;
            auto* block = new mlir::Block;
            for (auto [required, local] : requiredState[currRoot]) { // for states that step ops require
               assert(stateMapping.count(required)); // this means that we must have seen them before (i.e., produced by steps with no requirements)
               inputs.push_back(stateMapping[required]); // declare them as inputs, mapping here means that we "inline" the result of a step that produced the state
               blockArgs.push_back(block->addArgument(local.getType(), local.getLoc())); // we have 1-1 mapping between required states and block args
               threadLocal.push_back(false);
            }
            mlir::OpBuilder builder(&getContext());
            builder.setInsertionPointToStart(block);
            for (auto* op : steps[currRoot]) { // fill the step's block with all ops that belong to a step (cleanup the previous op location)
               op->remove();
               for (auto [o, n] : llvm::zip(requiredState[currRoot], blockArgs)) {
                  auto [required, local] = o;
                  local.replaceUsesWithIf(n, [&](mlir::OpOperand& operand) { // replace state uses within a step with a block argument in descendant ops
                     return op->isAncestor(operand.getOwner());
                  });
               }
               builder.insert(op);
            }
            builder.create<subop::ExecutionStepReturnOp>(currRoot->getLoc(), producedState[currRoot]);
            auto executionStepOp = outerBuilder.create<subop::ExecutionStepOp>(currRoot->getLoc(), returnTypes, inputs, outerBuilder.getBoolArrayAttr(threadLocal));
            executionStepOp.getSubOps().getBlocks().push_back(block);
            for (auto [s1, s2] : llvm::zip(producedState[currRoot], executionStepOp.getResults())) { // map produced values to the new results
               stateMapping[s1] = s2;
            }
         }
         for (auto [root, c] : dependCount) {
            if (c != 0) {
               root->dump();
               llvm::dbgs() << "dependencies:\n";
               for (auto* dep : dependencies[root]) {
                  if (dependCount[dep] > 0) {
                     dep->dump();
                  }
               }
               llvm::dbgs() << "-----------------------------------------------\n";
            }
         }
         auto returnOp = mlir::cast<subop::ExecutionGroupReturnOp>(executionGroup.getSubOps().front().getTerminator());
         std::vector<mlir::Value> returnValues;
         for (auto result : returnOp.getInputs()) {
            returnValues.push_back(stateMapping[result]);
         }
         returnOp->setOperands(returnValues);
      });
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
subop::createSplitIntoExecutionStepsPass() { return std::make_unique<SplitIntoExecutionSteps>(); }
