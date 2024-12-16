#include "llvm/Support/Debug.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/SubOpDependencyAnalysis.h"
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
      getOperation()->walk([&](subop::ExecutionGroupOp executionGroup) {

         std::unordered_map<mlir::Operation*, std::vector<mlir::Operation*>> steps;
         std::unordered_map<mlir::Operation*, mlir::Operation*> opToStep;
         for (mlir::Operation& op : executionGroup.getSubOps().front()) {
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
            if (beforeInStream) {
               steps[opToStep[beforeInStream]].push_back(&op);
               opToStep[&op] = opToStep[beforeInStream];

            } else {
               opToStep[&op] = &op;
               steps[&op].push_back(&op);
            }
         }
         // Step 2: collect required/produced state for each step
         // -> also deal with GetLocal operations (that do belong to the same step that accesses the state)
         std::unordered_map<mlir::Operation*, std::vector<std::tuple<mlir::Value, mlir::Value>>> requiredState;
         std::unordered_map<mlir::Operation*, std::vector<mlir::Value>> producedState;
         enum Kind {
            READ,
            WRITE
         };

         std::unordered_map<std::string, std::vector<std::tuple<subop::SubOperator, mlir::Operation*, Kind>>> memberUsage;
         for (auto& step : steps) {
            for (auto* op : step.second) {
               for (auto result : op->getResults()) {
                  if (!mlir::isa<tuples::TupleStreamType>(result.getType())) {
                     producedState[step.first].push_back(result);
                  }
               }
               op->walk([&](mlir::Operation* nestedOp) {
                  if (subop::SubOperator potentialSubOp = mlir::dyn_cast_or_null<subop::SubOperator>(nestedOp)) {
                     for (auto member : potentialSubOp.getReadMembers()) {
                        memberUsage[member].push_back({potentialSubOp, op, READ});
                     }
                     for (auto member : potentialSubOp.getWrittenMembers()) {
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
                           requiredState[step.first].push_back({operand, operand});
                        }
                     }
                  }
               });
            }
         }
         std::unordered_map<mlir::Operation*, std::unordered_set<mlir::Operation*>> dependencies;
         for (auto& [step, vals] : requiredState) {
            for (auto val : vals) {
               if (auto* producer = std::get<0>(val).getDefiningOp()) {
                  if (producer->getBlock() == step->getBlock()) {
                     auto* producerStep = opToStep[producer];
                     if (producerStep != step) {
                        dependencies[step].insert(producerStep);
                     }
                  }
               }
            }
         }

         // Step 3: determine correct order of steps
         for (auto [member, ops] : memberUsage) {
            for (size_t i = 0; i < ops.size(); i++) {
               for (size_t j = i + 1; j < ops.size(); j++) {
                  auto* pipelineOp1 = std::get<1>(ops[i]);
                  auto* pipelineOp2 = std::get<1>(ops[j]);
                  auto kind1 = std::get<2>(ops[i]);
                  auto kind2 = std::get<2>(ops[j]);
                  auto addConflict = [&]() {
                     auto* step1 = opToStep[pipelineOp1];
                     auto* step2 = opToStep[pipelineOp2];
                     assert(step1);
                     assert(step2);
                     if (step1 == step2) {
                        return;
                     }
                     if (pipelineOp1->isBeforeInBlock(pipelineOp2)) {
                        dependencies[step2].insert(step1);
                     } else {
                        dependencies[step1].insert(step2);
                     }
                  };
                  if (kind1 == WRITE && kind2 == WRITE) {
                     addConflict();
                  }
                  if ((kind1 == WRITE && kind2 == READ) || (kind1 == READ && kind2 == WRITE)) {
                     addConflict();
                  }
               }
            }
         }

         // Step 4: create ExecutionStepOps in correct order and handle states
         std::unordered_map<mlir::Operation*, size_t> dependCount;
         std::queue<mlir::Operation*> queue;
         llvm::DenseMap<mlir::Value, mlir::Value> stateMapping;

         for (auto& [step, ops] : steps) {
            dependCount[step] = dependencies[step].size();
            if (dependCount[step] == 0) {
               queue.push(step);
            }
         }
         std::unordered_map<mlir::Operation*, std::unordered_set<mlir::Operation*>> inverseDependencies;
         for (auto& [a, b] : dependencies) {
            for (auto* c : b) {
               inverseDependencies[c].insert(a);
            }
         }

         while (!queue.empty()) {
            auto* currRoot = queue.front();
            queue.pop();
            for (auto* otherRoot : inverseDependencies[currRoot]) {
               if (dependCount[otherRoot] > 0 && otherRoot != currRoot) {
                  dependCount[otherRoot]--;
                  if (dependCount[otherRoot] == 0) {
                     queue.push(otherRoot);
                  }
               }
            }
            std::vector<mlir::Type> returnTypes;
            for (auto produced : producedState[currRoot]) {
               returnTypes.push_back(produced.getType());
            }
            mlir::OpBuilder outerBuilder(&getContext());
            outerBuilder.setInsertionPoint(executionGroup.getSubOps().front().getTerminator());
            std::vector<mlir::Value> inputs;
            std::vector<mlir::Value> blockArgs;
            llvm::SmallVector<bool> threadLocal;
            auto* block = new mlir::Block;
            for (auto [required, local] : requiredState[currRoot]) {
               assert(stateMapping.count(required));
               inputs.push_back(stateMapping[required]);
               blockArgs.push_back(block->addArgument(local.getType(), local.getLoc()));
               threadLocal.push_back(false);
            }
            mlir::OpBuilder builder(&getContext());
            builder.setInsertionPointToStart(block);
            for (auto* op : steps[currRoot]) {
               op->remove();
               for (auto [o, n] : llvm::zip(requiredState[currRoot], blockArgs)) {
                  auto [required, local] = o;
                  local.replaceUsesWithIf(n, [&](mlir::OpOperand& operand) {
                     return op->isAncestor(operand.getOwner());
                  });
               }
               builder.insert(op);
            }
            builder.create<subop::ExecutionStepReturnOp>(currRoot->getLoc(), producedState[currRoot]);
            auto executionStepOp = outerBuilder.create<subop::ExecutionStepOp>(currRoot->getLoc(), returnTypes, inputs, outerBuilder.getBoolArrayAttr(threadLocal));

            executionStepOp.getSubOps().getBlocks().push_back(block);
            for (auto [s1, s2] : llvm::zip(producedState[currRoot], executionStepOp.getResults())) {
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
