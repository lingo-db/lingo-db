#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "llvm/Support/Debug.h"

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

class PrepareLoweringPass : public mlir::PassWrapper<PrepareLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareLoweringPass)
   virtual llvm::StringRef getArgument() const override { return "subop-prepare-lowering"; }
   void splitNested(subop::ExecutionGroupOp executionGroupOp) { // we take the query
      std::vector<subop::ContainsNestedSubOps> opsWithNesting;
      executionGroupOp->walk<mlir::WalkOrder::PreOrder>([&](subop::ContainsNestedSubOps containsNestedSubOps) { // walk through query ops
         opsWithNesting.push_back(containsNestedSubOps); // gather ops that follow ContainsNestedSubOps interface (e.g., nested map or loop)
      });
      for (auto containsNestedSubOps : opsWithNesting) {
         llvm::DenseMap<mlir::Value, size_t> valueToNestedExecutionGroupOperand;

         std::unordered_map<mlir::Operation*, std::vector<mlir::Operation*>> steps; // {startPipelineOp, pipelineOps}
         std::unordered_map<mlir::Operation*, mlir::Operation*> opToStep; // {pipelineOp, startPipelineOp}
         for (mlir::Operation& op : *containsNestedSubOps.getBody()) {
            if (&op == containsNestedSubOps.getBody()->getTerminator()) {
               continue;
            }
            mlir::Operation* beforeInStream = nullptr;
            for (auto operand : op.getOperands()) {
               if (mlir::isa<tuples::TupleStreamType>(operand.getType())) {
                  if (auto* producer = operand.getDefiningOp()) {
                     assert(!beforeInStream);
                     beforeInStream = producer; // we expect only one tuple-stream per op in *containsNestedSubOps.getBody()
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
         std::unordered_map<mlir::Operation*, std::vector<std::tuple<mlir::Value, mlir::Value, bool>>> requiredState;
         std::unordered_map<mlir::Operation*, std::vector<mlir::Value>> producedState;
         std::unordered_set<mlir::Operation*> getLocals;

         enum Kind {
            READ,
            WRITE
         };

         std::unordered_map<std::string, std::vector<std::tuple<subop::SubOperator, mlir::Operation*, Kind>>> memberUsage;

         for (auto& step : steps) {
            for (auto* op : step.second) {
               // collect produced state of nested operations that are currently investigated
               for (auto result : op->getResults()) {
                  if (!mlir::isa<tuples::TupleStreamType>(result.getType())) {
                     producedState[step.first].push_back(result);
                  }
               }
               //check how states are used
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
                     auto* opProducingOperand = operand.getDefiningOp() ? operand.getDefiningOp() : mlir::cast<mlir::BlockArgument>(operand).getOwner()->getParentOp();
                     if (opProducingOperand->isProperAncestor(op) || (operand.getDefiningOp() && operand.getDefiningOp()->getBlock()->getParentOp()->isAncestor(op->getBlock()->getParentOp()))) {
                        if (mlir::isa<tuples::TupleStreamType>(operand.getType())) {
                           continue;
                        }
                        requiredState[step.first].push_back({operand, operand, false});
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

         auto* nestedExecutionBlock = new mlir::Block;
         std::vector<mlir::Value> nestedExecutionOperands;
         mlir::OpBuilder b(&getContext());

         b.setInsertionPointToStart(nestedExecutionBlock);

         auto returnOp = b.create<subop::NestedExecutionGroupReturnOp>(containsNestedSubOps.getLoc(), mlir::ValueRange{});
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
            outerBuilder.setInsertionPoint(returnOp);
            std::vector<mlir::Value> inputs;
            std::vector<mlir::Value> blockArgs;
            llvm::SmallVector<bool> threadLocal;
            auto* block = new mlir::Block;
            llvm::DenseMap<mlir::Value, size_t> availableStates;

            for (auto [required, local, isThreadLocal] : requiredState[currRoot]) {
               if (availableStates.contains(required)) {
                  blockArgs.push_back(block->getArgument(availableStates[required]));
                  continue;
               }
               if (stateMapping.count(required) == 0) {
                  if (valueToNestedExecutionGroupOperand.contains(required)) {
                     inputs.push_back(nestedExecutionBlock->getArgument(valueToNestedExecutionGroupOperand[required]));
                  } else {
                     nestedExecutionOperands.push_back(required);
                     auto nestedExecutionBlockArg = nestedExecutionBlock->addArgument(required.getType(), required.getLoc());
                     inputs.push_back(nestedExecutionBlockArg);
                     valueToNestedExecutionGroupOperand[required] = nestedExecutionBlockArg.getArgNumber();
                  }
               } else {
                  assert(stateMapping.count(required));
                  inputs.push_back(stateMapping[required]);
               }
               blockArgs.push_back(block->addArgument(local.getType(), local.getLoc()));
               threadLocal.push_back(isThreadLocal);
               availableStates[required] = block->getNumArguments() - 1;
            }
            mlir::OpBuilder builder(&getContext());
            builder.setInsertionPointToStart(block);
            for (auto* op : steps[currRoot]) {
               op->remove();
               for (auto [o, n] : llvm::zip(requiredState[currRoot], blockArgs)) {
                  auto [required, local, isThreadLocal] = o;

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
         auto* terminator = containsNestedSubOps.getBody()->getTerminator();
         b.setInsertionPoint(terminator);
         std::vector<mlir::Value> toReturn;
         std::vector<mlir::Value> toMap;
         std::vector<mlir::Type> toReturnTypes;
         for (auto operand : terminator->getOperands()) {
            if (stateMapping.count(operand)) {
               toReturn.push_back(stateMapping[operand]);
               toReturnTypes.push_back(operand.getType());
               toMap.push_back(operand);
            }
         }
         returnOp->setOperands(toReturn);
         auto nestedExecutionGroup = b.create<subop::NestedExecutionGroupOp>(containsNestedSubOps.getLoc(), toReturnTypes, nestedExecutionOperands);
         nestedExecutionGroup.getSubOps().getBlocks().clear();
         nestedExecutionGroup.getSubOps().push_back(nestedExecutionBlock);
         b.setInsertionPointToStart(nestedExecutionBlock);
         for (auto [from, to] : llvm::zip(toMap, nestedExecutionGroup.getResults())) {
            from.replaceUsesWithIf(to, [&](mlir::OpOperand& operand) {
               return terminator == operand.getOwner();
            });
         }

         for (auto* getLocal : getLocals) {
            getLocal->erase();
         }
      }
   }
   void runOnOperation() override {
      subop::ColumnUsageAnalysis usedColumns(getOperation());
      subop::ColumnCreationAnalysis createdColumns(getOperation());
      std::vector<mlir::Operation*> opsToErase;
      getOperation()->walk([&](subop::NestedMapOp nestedMapOp) { // for each nested map
         std::unordered_map<tuples::Column*, size_t> colArgs;
         std::vector<mlir::Attribute> newParams;
         size_t argId = 1;
         for (auto colRefAttr : nestedMapOp.getParameters()) {
            colArgs.insert({&mlir::cast<tuples::ColumnRefAttr>(colRefAttr).getColumn(), argId++});
            newParams.push_back(colRefAttr);
         }
         auto tuple = nestedMapOp.getBody()->getArgument(0);
         for (auto* user : tuple.getUsers()) {
            if (auto combineOp = mlir::dyn_cast<subop::CombineTupleOp>(user)) { // if this tuple gets combined
               if (combineOp->getBlock() != nestedMapOp.getBody()) {
                  nestedMapOp.emitError("NestedMapOp: tuple parameter must only be used in the body block");
                  return signalPassFailure();
               }
               std::unordered_set<tuples::Column*> availableColumns;
               std::function<void(mlir::Value)> computeAvailableColumns = [&](mlir::Value v) { // fills availableColumns
                  if (auto* defOp = v.getDefiningOp()) {
                     auto createdOps = createdColumns.getCreatedColumns(defOp); // gather columns that combineOp's defining op created

                     availableColumns.insert(createdOps.begin(), createdOps.end());
                     for (auto operand : defOp->getOperands()) {
                        if (mlir::isa<tuples::TupleStreamType>(operand.getType())) {
                           computeAvailableColumns(operand);
                        }
                     }
                  }
               };
               std::unordered_set<tuples::Column*> requiredColumns;
               std::function<void(mlir::Operation*, std::unordered_set<tuples::Column*>)> addRequiredColumns = [&](mlir::Operation* op, std::unordered_set<tuples::Column*> availableColumns) {
                  auto created = createdColumns.getCreatedColumns(op); // for the combine op, get the columns it creates
                  availableColumns.insert(created.begin(), created.end()); // aggregate them in availableColumns, it contains all columns that are in the tuple-stream up to the current  combineOp
                  for (auto* usedColumn : usedColumns.getUsedColumns(op)) {
                     if (!availableColumns.contains(usedColumn)) {
                        requiredColumns.insert(usedColumn); // if the combineOp uses a column that it doesn't create or that doesn't come from its definitions, the combineOp requires it
                     }
                  }
                  for (auto* user : op->getUsers()) {
                     addRequiredColumns(user, availableColumns); // update the required cols for all users of this combineOp
                  }
               };
               computeAvailableColumns(combineOp);
               addRequiredColumns(combineOp, availableColumns);
               if(requiredColumns.empty()) {
                  combineOp->replaceAllUsesWith(mlir::ValueRange{combineOp.getStream()}); //useless combineop -> just remove
                  opsToErase.push_back(combineOp);
                  continue;
               }else {
                  std::vector<mlir::Attribute> createdByMap;
                  mlir::OpBuilder b(combineOp);
                  mlir::Block* mapBlock = new mlir::Block;
                  {
                     mlir::OpBuilder::InsertionGuard guard(b);
                     b.setInsertionPointToStart(mapBlock);
                     auto& colManager = getContext().getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
                     std::vector<mlir::Value> mapOpReturnVals;
                     for (auto* r : requiredColumns) { // we need to introduce every required column to the MapOp (MapOp will replace CombineTupleOp)
                        auto colRefAttr = colManager.createRef(r);
                        auto colDefAttr = colManager.createDef(r);

                        if (!colArgs.contains(r)) { // if the required column is not a param of nested map
                           nestedMapOp.getBody()->addArgument(r->type, b.getUnknownLoc()); // we need to add it to nested map op arguments
                           colArgs.insert({r, argId++});
                           newParams.push_back(colRefAttr); // and include in the parameter
                        }
                        createdByMap.push_back(colDefAttr); // every required column is created/returned by the MapOp
                        mapOpReturnVals.push_back(nestedMapOp.getBody()->getArgument(colArgs[r]));
                     }
                     b.create<tuples::ReturnOp>(b.getUnknownLoc(), mapOpReturnVals);
                  }
                  auto mapOp = b.create<subop::MapOp>(b.getUnknownLoc(), tuples::TupleStreamType::get(b.getContext()), combineOp.getStream(), b.getArrayAttr(createdByMap), b.getArrayAttr({}));
                  mapOp.getFn().push_back(mapBlock);
                  combineOp->replaceAllUsesWith(mlir::ValueRange{mapOp.getResult()}); // replace combineOp with mapOp
                  opsToErase.push_back(combineOp);
                  createdColumns.update(mapOp);
               }

            } else {
               nestedMapOp.emitError("NestedMapOp: tuple parameter must only be used with CombineTupleOp");
               return signalPassFailure();
            }
         }
         nestedMapOp.setParametersAttr(mlir::ArrayAttr::get(&getContext(), newParams));
      });
      for (auto* op : opsToErase) {
         op->erase();
      }
      getOperation()->walk([&](subop::ExecutionGroupOp executionGroupOp) {
         splitNested(executionGroupOp);
      });
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
subop::createPrepareLoweringPass() { return std::make_unique<PrepareLoweringPass>(); }