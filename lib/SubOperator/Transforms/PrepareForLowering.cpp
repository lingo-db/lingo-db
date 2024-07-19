#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SubOperator/SubOperatorDialect.h"
#include "mlir/Dialect/SubOperator/SubOperatorOps.h"

#include "mlir/Dialect/SubOperator/Transforms/ColumnCreationAnalysis.h"
#include "mlir/Dialect/SubOperator/Transforms/ColumnUsageAnalysis.h"

#include "mlir/Dialect/SubOperator/Transforms/Passes.h"
#include "mlir/Dialect/SubOperator/Transforms/SubOpDependencyAnalysis.h"
#include "mlir/Dialect/TupleStream/TupleStreamDialect.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <queue>
namespace {

class PrepareLoweringPass : public mlir::PassWrapper<PrepareLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareLoweringPass)
   virtual llvm::StringRef getArgument() const override { return "subop-prepare-lowering"; }
   void splitNested(mlir::subop::ExecutionGroupOp executionGroupOp) {
      std::vector<mlir::subop::ContainsNestedSubOps> opsWithNesting;
      executionGroupOp->walk<mlir::WalkOrder::PreOrder>([&](mlir::subop::ContainsNestedSubOps containsNestedSubOps) {
         opsWithNesting.push_back(containsNestedSubOps);
      });
      for (auto containsNestedSubOps : opsWithNesting) {
         std::unordered_map<mlir::Operation*, std::vector<mlir::Operation*>> steps;
         std::unordered_map<mlir::Operation*, mlir::Operation*> opToStep;
         for (mlir::Operation& op : *containsNestedSubOps.getBody()) {
            if (&op == containsNestedSubOps.getBody()->getTerminator()) {
               continue;
            }
            mlir::Operation* beforeInStream = nullptr;
            for (auto operand : op.getOperands()) {
               if (operand.getType().isa<mlir::tuples::TupleStreamType>()) {
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
         std::unordered_map<mlir::Operation*, std::vector<std::tuple<mlir::Value, mlir::Value, bool>>> requiredState;
         std::unordered_map<mlir::Operation*, std::vector<mlir::Value>> producedState;
         std::unordered_set<mlir::Operation*> getLocals;

         enum Kind {
            READ,
            WRITE
         };

         std::unordered_map<std::string, std::vector<std::tuple<mlir::subop::SubOperator, mlir::Operation*, Kind>>> memberUsage;

         for (auto& step : steps) {
            for (auto* op : step.second) {
               for (auto result : op->getResults()) {
                  if (!result.getType().isa<mlir::tuples::TupleStreamType>()) {
                     producedState[step.first].push_back(result);
                  }
               }
               op->walk([&](mlir::Operation* nestedOp) {
                  if (mlir::subop::SubOperator potentialSubOp = mlir::dyn_cast_or_null<mlir::subop::SubOperator>(nestedOp)) {
                     for (auto member : potentialSubOp.getReadMembers()) {
                        memberUsage[member].push_back({potentialSubOp, op, READ});
                     }
                     for (auto member : potentialSubOp.getWrittenMembers()) {
                        memberUsage[member].push_back({potentialSubOp, op, WRITE});
                     }
                  }
                  for (auto operand : nestedOp->getOperands()) {
                     auto* parentOp = operand.getDefiningOp() ? operand.getDefiningOp() : mlir::cast<mlir::BlockArgument>(operand).getOwner()->getParentOp();

                     if (parentOp->isProperAncestor(op) || (operand.getDefiningOp() && operand.getDefiningOp()->getBlock() == op->getBlock())) {
                        if (operand.getType().isa<mlir::tuples::TupleStreamType>()) {
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

         auto returnOp = b.create<mlir::subop::NestedExecutionGroupReturnOp>(containsNestedSubOps.getLoc(), mlir::ValueRange{});
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

            for (auto [required, local, isThreadLocal] : requiredState[currRoot]) {
               if (stateMapping.count(required) == 0) {
                  nestedExecutionOperands.push_back(required);
                  auto nestedExecutionBlockArg = nestedExecutionBlock->addArgument(required.getType(), required.getLoc());
                  inputs.push_back(nestedExecutionBlockArg);
               } else {
                  assert(stateMapping.count(required));
                  inputs.push_back(stateMapping[required]);
               }
               blockArgs.push_back(block->addArgument(local.getType(), local.getLoc()));
               threadLocal.push_back(isThreadLocal);
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
            builder.create<mlir::subop::ExecutionStepReturnOp>(currRoot->getLoc(), producedState[currRoot]);
            auto executionStepOp = outerBuilder.create<mlir::subop::ExecutionStepOp>(currRoot->getLoc(), returnTypes, inputs, outerBuilder.getBoolArrayAttr(threadLocal));

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
         auto nestedExecutionGroup = b.create<mlir::subop::NestedExecutionGroupOp>(containsNestedSubOps.getLoc(), toReturnTypes, nestedExecutionOperands);
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
      mlir::subop::ColumnUsageAnalysis usedColumns(getOperation());
      mlir::subop::ColumnCreationAnalysis createdColumns(getOperation());
      std::vector<mlir::Operation*> opsToErase;
      getOperation()->walk([&](mlir::subop::NestedMapOp nestedMapOp) {
         std::unordered_map<mlir::tuples::Column*, size_t> colArgs;
         std::vector<mlir::Attribute> newParams;
         size_t argId = 1;
         for (auto colRefAttr : nestedMapOp.getParameters()) {
            colArgs.insert({&colRefAttr.cast<mlir::tuples::ColumnRefAttr>().getColumn(), argId++});
            newParams.push_back(colRefAttr);
         }
         auto tuple = nestedMapOp.getBody()->getArgument(0);
         for (auto* user : tuple.getUsers()) {
            if (auto combineOp = mlir::dyn_cast<mlir::subop::CombineTupleOp>(user)) {
               if (combineOp->getBlock() != nestedMapOp.getBody()) {
                  nestedMapOp.emitError("NestedMapOp: tuple parameter must only be used in the body block");
                  return signalPassFailure();
               }
               std::unordered_set<mlir::tuples::Column*> availableColumns;
               std::function<void(mlir::Value)> computeAvailableColumns = [&](mlir::Value v) {
                  if (auto* defOp = v.getDefiningOp()) {
                     auto createdOps = createdColumns.getCreatedColumns(defOp);

                     availableColumns.insert(createdOps.begin(), createdOps.end());
                     for (auto operand : defOp->getOperands()) {
                        if (operand.getType().isa<mlir::tuples::TupleStreamType>()) {
                           computeAvailableColumns(operand);
                        }
                     }
                  }
               };
               std::unordered_set<mlir::tuples::Column*> requiredColumns;
               std::function<void(mlir::Operation*, std::unordered_set<mlir::tuples::Column*>)> addRequiredColumns = [&](mlir::Operation* op, std::unordered_set<mlir::tuples::Column*> availableColumns) {
                  auto created = createdColumns.getCreatedColumns(op);
                  availableColumns.insert(created.begin(), created.end());
                  for (auto* usedColumn : usedColumns.getUsedColumns(op)) {
                     if (!availableColumns.contains(usedColumn)) {
                        requiredColumns.insert(usedColumn);
                     }
                  }
                  for (auto* user : op->getUsers()) {
                     addRequiredColumns(user, availableColumns);
                  }
               };
               computeAvailableColumns(combineOp);
               addRequiredColumns(combineOp, availableColumns);
               std::vector<mlir::Attribute> createdByMap;
               mlir::OpBuilder b(combineOp);
               mlir::Block* mapBlock = new mlir::Block;
               {
                  mlir::OpBuilder::InsertionGuard guard(b);
                  b.setInsertionPointToStart(mapBlock);
                  auto& colManager = getContext().getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
                  std::vector<mlir::Value> mapOpReturnVals;
                  for (auto* r : requiredColumns) {
                     auto colRefAttr = colManager.createRef(r);
                     auto colDefAttr = colManager.createDef(r);

                     if (!colArgs.contains(r)) {
                        nestedMapOp.getBody()->addArgument(r->type, b.getUnknownLoc());
                        colArgs.insert({r, argId++});
                        newParams.push_back(colRefAttr);
                     }
                     createdByMap.push_back(colDefAttr);
                     mapOpReturnVals.push_back(nestedMapOp.getBody()->getArgument(colArgs[r]));
                  }
                  b.create<mlir::tuples::ReturnOp>(b.getUnknownLoc(), mapOpReturnVals);
               }
               auto mapOp = b.create<mlir::subop::MapOp>(b.getUnknownLoc(), mlir::tuples::TupleStreamType::get(b.getContext()), combineOp.getStream(), b.getArrayAttr(createdByMap),b.getArrayAttr({}));
               mapOp.getFn().push_back(mapBlock);
               combineOp->replaceAllUsesWith(mlir::ValueRange{mapOp.getResult()});
               opsToErase.push_back(combineOp);

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
      getOperation()->walk([&](mlir::subop::ExecutionGroupOp executionGroupOp) {
         splitNested(executionGroupOp);
      });
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
mlir::subop::createPrepareLoweringPass() { return std::make_unique<PrepareLoweringPass>(); }