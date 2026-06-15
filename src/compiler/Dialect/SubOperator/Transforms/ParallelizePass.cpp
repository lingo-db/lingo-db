#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/Support/Debug.h"

#include "lingodb/compiler/Dialect/SubOperator/SubOperatorInterfaces.h"
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

#include "mlir/Transforms/DialectConversion.h"

namespace {
using namespace lingodb::compiler::dialect;

struct ProblematicOp {
   subop::SubOperator op;
   subop::SubOperator stateAccessing;
};

struct CollisionGroup {
   std::vector<ProblematicOp> ops;
};
struct ExecutionStepAnalyzed {
   subop::SubOperator pipelineStart;
   std::vector<CollisionGroup> collisionGroups;
   bool notParallel = false;
};

// Resolve a state value through execution_step / nested_execution_group arg->input
// edges back to one of the analyzed step's external states, so a state threaded in
// via nested region args is recognized as shared rather than thread-private.
static mlir::Value resolveThroughNesting(mlir::Value v, const llvm::DenseMap<mlir::Value, mlir::Value>& extStates) {
   while (!extStates.contains(v)) {
      auto barg = mlir::dyn_cast<mlir::BlockArgument>(v);
      if (!barg) break;
      mlir::Operation* parent = barg.getOwner()->getParentOp();
      unsigned idx = barg.getArgNumber();
      if (auto step = mlir::dyn_cast_or_null<subop::ExecutionStepOp>(parent)) {
         if (barg.getOwner() == &step.getSubOps().front() && idx < step.getInputs().size()) {
            v = step.getInputs()[idx];
            continue;
         }
      } else if (auto nestedGroup = mlir::dyn_cast_or_null<subop::NestedExecutionGroupOp>(parent)) {
         if (idx < nestedGroup.getInputs().size()) {
            v = nestedGroup.getInputs()[idx];
            continue;
         }
      }
      break;
   }
   return v;
}

// Like resolveThroughNesting but ignores the per-step extStates guard: walk
// arg->input edges all the way out through execution_step / nested_execution_group
// nesting until reaching a value that is not such a nested block argument. For a
// state created outside an enclosing loop and threaded into the loop body, this
// returns the original (external) create result, whose defining op is reachable.
static mlir::Value resolveToRoot(mlir::Value v) {
   while (auto barg = mlir::dyn_cast<mlir::BlockArgument>(v)) {
      mlir::Operation* parent = barg.getOwner()->getParentOp();
      unsigned idx = barg.getArgNumber();
      if (auto step = mlir::dyn_cast_or_null<subop::ExecutionStepOp>(parent)) {
         if (barg.getOwner() == &step.getSubOps().front() && idx < step.getInputs().size()) {
            v = step.getInputs()[idx];
            continue;
         }
      } else if (auto nestedGroup = mlir::dyn_cast_or_null<subop::NestedExecutionGroupOp>(parent)) {
         if (idx < nestedGroup.getInputs().size()) {
            v = nestedGroup.getInputs()[idx];
            continue;
         }
      }
      break;
   }
   return v;
}

// A reduce combine merges two partial aggregates (the block args are the two
// partials, each `numMembers` wide). It is only a sound parallel merge if its
// result depends on the second partial; a combine like (a,b)->a discards a
// partial and cannot be parallelized.
static bool combineMergesBothPartials(mlir::Region& combine, unsigned numMembers) {
   if (combine.empty() || numMembers == 0) return false;
   mlir::Block& block = combine.front();
   if (block.getNumArguments() != 2 * numMembers) return false;
   llvm::DenseSet<mlir::Value> secondArgs;
   for (unsigned i = numMembers; i < 2 * numMembers; i++)
      secondArgs.insert(block.getArgument(i));
   llvm::SmallVector<mlir::Value> work(block.getTerminator()->getOperands().begin(), block.getTerminator()->getOperands().end());
   llvm::DenseSet<mlir::Value> seen;
   while (!work.empty()) {
      mlir::Value v = work.pop_back_val();
      if (!seen.insert(v).second) continue;
      if (secondArgs.contains(v)) return true;
      if (auto* def = v.getDefiningOp())
         work.append(def->getOperands().begin(), def->getOperands().end());
   }
   return false;
}

ExecutionStepAnalyzed analyze(subop::ExecutionStepOp executionStepOp) {
   subop::ColumnCreationAnalysis columnCreationAnalysis(executionStepOp);
   ExecutionStepAnalyzed result;
   llvm::DenseMap<mlir::Value, mlir::Value> extStates;
   llvm::EquivalenceClasses<mlir::Operation*> equivalenceClasses;
   llvm::DenseMap<mlir::Operation*, size_t> opToIndex;
   std::vector<ProblematicOp> problematicOps;

   std::function<void(subop::SubOperator op, const std::unordered_set<mlir::Operation*>& collisions, subop::SubOperator stateAccessing)> addProblematicOp = [&](subop::SubOperator op, const std::unordered_set<mlir::Operation*>& collisions, subop::SubOperator stateAccessing) {
      ProblematicOp p;
      p.op = op;
      p.stateAccessing = stateAccessing;
      problematicOps.push_back(p);
      opToIndex[op.getOperation()] = problematicOps.size() - 1;
      equivalenceClasses.insert(op.getOperation());
      for (auto* c : collisions) {
         equivalenceClasses.unionSets(op.getOperation(), c);
      }
      if (stateAccessing) {
         equivalenceClasses.unionSets(op.getOperation(), stateAccessing.getOperation());
      }
   };
   llvm::SmallDenseSet<subop::Member> extMembers;
   for (auto [i, a] : llvm::zip(executionStepOp.getInputs(), executionStepOp.getSubOps().getArguments())) {
      auto argType = a.getType();
      extStates.insert({a, i});
      if (auto stateType = mlir::dyn_cast_or_null<subop::State>(argType)) {
         auto stateMembers = stateType.getMembers().getMembers();
         extMembers.insert(stateMembers.begin(), stateMembers.end());
      }
   }
   auto* firstOp = &*executionStepOp.getOps().begin();
   result.pipelineStart = mlir::cast<subop::SubOperator>(firstOp);

   llvm::DenseMap<subop::Member, std::vector<mlir::Operation*>> readMembers;
   llvm::DenseMap<subop::Member, std::vector<mlir::Operation*>> writtenMembers;

   std::vector<subop::SubOperator> pipelineOps;
   executionStepOp.getSubOps().walk([&](subop::SubOperator subOp) {
      if (firstOp != subOp.getOperation() && !mlir::isa<subop::ExecutionStepOp>(subOp.getOperation())) {
         pipelineOps.push_back(subOp);
      }
   });
   for (auto subOp : pipelineOps) {
      auto* pipelineOp = subOp.getOperation();
      auto getCollisions = [&]() {
         std::unordered_set<mlir::Operation*> collisions;

         auto currentReadMembers = subOp.getReadMembers();
         auto currentWrittenMembers = subOp.getWrittenMembers();

         for (auto r : currentReadMembers) {
            if (writtenMembers.contains(r)) {
               collisions.insert(writtenMembers[r].begin(), writtenMembers[r].end());
            }
         }
         for (auto w : currentWrittenMembers) {
            if (readMembers.contains(w)) {
               collisions.insert(readMembers[w].begin(), readMembers[w].end());
            }
         }
         for (auto r : currentReadMembers) {
            if (extMembers.contains(r)) {
               readMembers[r].push_back(subOp);
            }
         }
         for (auto r : currentWrittenMembers) {
            if (extMembers.contains(r)) {
               writtenMembers[r].push_back(subOp);
            }
         }

         return collisions;
      };
      auto isNested = [&](mlir::Value v) { return !extStates.contains(resolveThroughNesting(v, extStates)); };
      auto getCreationOp = [&](tuples::Column& column) {
         auto creationOp = mlir::cast<subop::SubOperator>(columnCreationAnalysis.getColumnCreator(&column));
         if (auto unwrapOp = mlir::dyn_cast_or_null<subop::UnwrapOptionalRefOp>(creationOp.getOperation())) {
            creationOp = mlir::cast<subop::SubOperator>(columnCreationAnalysis.getColumnCreator(&unwrapOp.getOptionalRef().getColumn()));
         }
         return creationOp;
      };
      //
      if (auto stateUsingSubOp = mlir::dyn_cast_or_null<subop::StateUsingSubOperator>(pipelineOp)) {
         if (mlir::isa<subop::ScanListOp, subop::NestedMapOp>(pipelineOp)) {
            // ignore
         } else if (auto materializeOp = mlir::dyn_cast_or_null<subop::MaterializeOp>(pipelineOp)) {
            if (!isNested(materializeOp.getState())) {
               addProblematicOp(materializeOp, getCollisions(), {});
            }
         } else if (auto lookupOp = mlir::dyn_cast_or_null<subop::LookupOp>(pipelineOp)) {
            auto collisions = getCollisions();
            if (!isNested(lookupOp.getState()) && !collisions.empty()) {
               addProblematicOp(lookupOp, collisions, {});
            }
         } else if (auto lookupOrInsertOp = mlir::dyn_cast_or_null<subop::LookupOrInsertOp>(pipelineOp)) {
            if (!isNested(lookupOrInsertOp.getState())) {
               addProblematicOp(lookupOrInsertOp, getCollisions(), {});
            }
         } else if (auto reduceOp = mlir::dyn_cast_or_null<subop::ReduceOp>(pipelineOp)) {
            auto creationOp = getCreationOp(reduceOp.getRef().getColumn());
            auto accessesNestedState = llvm::all_of(creationOp->getOperands(), [&](mlir::Value v) { return mlir::isa<subop::State>(v.getType()) ? isNested(v) : true; });
            if (!accessesNestedState) {
               addProblematicOp(reduceOp, getCollisions(), creationOp);
            }
         } else if (auto scatterOp = mlir::dyn_cast_or_null<subop::ScatterOp>(pipelineOp)) {
            auto creationOp = getCreationOp(scatterOp.getRef().getColumn());
            auto accessesNestedState = llvm::all_of(creationOp->getOperands(), [&](mlir::Value v) { return mlir::isa<subop::State>(v.getType()) ? isNested(v) : true; });
            if (!accessesNestedState) {
               addProblematicOp(scatterOp, getCollisions(), creationOp);
            }
         } else if (auto gatherOp = mlir::dyn_cast_or_null<subop::ScatterOp>(pipelineOp)) {
            auto creationOp = getCreationOp(gatherOp.getRef().getColumn());
            auto accessesNestedState = llvm::all_of(creationOp->getOperands(), [&](mlir::Value v) { return mlir::isa<subop::State>(v.getType()) ? isNested(v) : true; });
            if (!accessesNestedState) {
               addProblematicOp(gatherOp, getCollisions(), creationOp);
            }
         } else {
            //problem: don't know how to handle
            result.notParallel = true;
         }
         // i
      } else {
         //every other operation is not really important
      }
   }
   for (auto it = equivalenceClasses.begin(), itEnd = equivalenceClasses.end(); it != itEnd; ++it)
      if (it->isLeader()) {
         CollisionGroup collisionGroup;

         for (auto member = equivalenceClasses.member_begin(it); member != equivalenceClasses.member_end(); ++member) {
            auto* memberOp = *member;
            if (opToIndex.count(memberOp))
               collisionGroup.ops.push_back(problematicOps[opToIndex[memberOp]]);
         }
         result.collisionGroups.push_back(collisionGroup);
      }

   return result;
}

class ParallelizePass : public mlir::PassWrapper<ParallelizePass, mlir::OperationPass<>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ParallelizePass)
   virtual llvm::StringRef getArgument() const override { return "subop-parallelize"; }
   struct ToThreadLocalInfo {
      mlir::Region* combineRegion = nullptr;
      mlir::Region* compareRegion = nullptr;
      bool requiresCombine = false;
   };
   struct GlobalThreadLocalInfo {
      mlir::Region* combineRegion = nullptr;
      mlir::Region* compareRegion = nullptr;
      llvm::DenseSet<subop::ExecutionStepOp> shouldUse;
      bool requiresCombine = false;
      bool isClosed = false;
   };
   void parallelizeBlock(mlir::Block& block, subop::ColumnUsageAnalysis& columnUsageAnalysis, tuples::ColumnManager& colManager) {
      llvm::DenseMap<mlir::Value, GlobalThreadLocalInfo> toThreadLocalsGlobal;
      std::map<tuples::Column*, std::vector<std::pair<mlir::Value, mlir::Operation*>>> toLockGlobal;
      llvm::DenseSet<mlir::Value> threadLocalNotPossibleAnymore;
      for (auto& op : block) {
            if (auto executionStepOp = mlir::dyn_cast_or_null<subop::ExecutionStepOp>(&op)) {
               llvm::DenseMap<mlir::Value, mlir::Value> extStates;
               for (auto [i, a] : llvm::zip(executionStepOp.getInputs(), executionStepOp.getSubOps().getArguments())) {
                  extStates.insert({a, i});
               }
               // Map a (locally resolved) state operand to the value that identifies
               // the state for thread-localization. Normally this is the execution
               // step input; but for a state created outside an enclosing loop and
               // threaded into the loop body, that input is itself a nested block
               // argument with no defining op. Resolve it the rest of the way out to
               // the external create so the in-loop writers can still be parallelized
               // (see the cross-loop handling in the conversion below).
               auto resolveExt = [&](mlir::Value localResolved) -> mlir::Value {
                  mlir::Value e = extStates[localResolved];
                  if (!e.getDefiningOp()) {
                     auto root = resolveToRoot(e);
                     if (root.getDefiningOp()) return root;
                  }
                  return e;
               };

               if (auto scanRefsOp = mlir::dyn_cast_or_null<subop::ScanRefsOp>(*executionStepOp.getSubOps().getOps().begin())) {
                  if (!scanRefsOp->hasAttr("sequential") && !scanRefsOp->hasAttr("parallel")) {
                     ExecutionStepAnalyzed analyzed = analyze(executionStepOp);
                     std::unordered_set<mlir::Operation*> markAsAtomic;
                     std::map<tuples::Column*, std::vector<std::pair<mlir::Value, mlir::Operation*>>> toLock;
                     llvm::DenseMap<mlir::Value, ToThreadLocalInfo> toThreadLocals;
                     bool canBeParallel = true;
                     for (auto& collisionGroup : analyzed.collisionGroups) {
                        //llvm::dbgs() << "Collision Group:\n";
                        for (auto& problematicOp : collisionGroup.ops) {
                           //llvm::dbgs() << "  Problematic Op: " << problematicOp.op << "\n";
                           if (auto materializeOp = mlir::dyn_cast_or_null<subop::MaterializeOp>(problematicOp.op.getOperation())) {
                              auto ext = resolveExt(resolveThroughNesting(materializeOp.getState(), extStates));
                              if (ext.getDefiningOp()) {
                                 toThreadLocals.insert({ext, {}});
                              } else {
                                 canBeParallel = false;
                              }
                           } else if (auto lookupOrInsert = mlir::dyn_cast_or_null<subop::LookupOrInsertOp>(problematicOp.op.getOperation())) {
                              auto ext = resolveExt(resolveThroughNesting(lookupOrInsert.getState(), extStates));
                              if (ext.getDefiningOp()) {
                                 toThreadLocals[ext].requiresCombine = true;
                                 toThreadLocals[ext].compareRegion = &lookupOrInsert.getEqFn();
                              } else {
                                 canBeParallel = false;
                              }
                           } else if (auto reduceOp = mlir::dyn_cast_or_null<subop::ReduceOp>(problematicOp.op.getOperation())) {
                              auto stateAccessing = problematicOp.stateAccessing;
                              if (auto lookupOp = mlir::dyn_cast_or_null<subop::LookupOp>(stateAccessing.getOperation())) {
                                 auto ext = resolveExt(resolveThroughNesting(lookupOp.getState(), extStates));
                                 if (ext.getDefiningOp() && mlir::isa<subop::SimpleStateType>(ext.getType()) && mlir::dyn_cast_or_null<subop::SimpleStateType>(ext.getType()).getMembers().getMembers().size() == reduceOp.getMembers().size() && !reduceOp.getCombine().empty() && combineMergesBothPartials(reduceOp.getCombine(), reduceOp.getMembers().size())) {
                                    toThreadLocals[ext].requiresCombine = true;
                                    toThreadLocals[ext].combineRegion = &reduceOp.getCombine();
                                 } else if (ext.getDefiningOp() && mlir::isa<subop::HashMapType>(ext.getType())) {
                                    toLock[&reduceOp.getRef().getColumn()].push_back(std::pair<mlir::Value, mlir::Operation*>{ext, reduceOp.getOperation()});
                                 } else {
                                    canBeParallel = false;
                                 }
                              } else if (auto lookupOrInsertOp = mlir::dyn_cast_or_null<subop::LookupOrInsertOp>(stateAccessing.getOperation())) {
                                 // lookupOrInsertOp will be handled either way
                                 auto ext = resolveExt(resolveThroughNesting(lookupOrInsertOp.getState(), extStates));
                                 if (ext.getDefiningOp()) {
                                    toThreadLocals[ext].requiresCombine = true;
                                    toThreadLocals[ext].combineRegion = &reduceOp.getCombine();
                                 } else {
                                    canBeParallel = false;
                                 }
                              } else {
                                 canBeParallel = false;
                              }
                           } else if (auto scatterOp = mlir::dyn_cast_or_null<subop::ScatterOp>(problematicOp.op.getOperation())) {
                              auto stateAccessing = problematicOp.stateAccessing;
                              if (collisionGroup.ops.size() == 1) {
                                 // TODO: check if this is actually a sane optimization on x86 and especially other architectures
                                 markAsAtomic.insert(scatterOp);
                              } else if (auto lookupOp = mlir::dyn_cast_or_null<subop::LookupOp>(stateAccessing.getOperation())) {
                                 auto ext = resolveExt(resolveThroughNesting(lookupOp.getState(), extStates));
                                 if (ext.getDefiningOp() && mlir::isa<subop::HashMapType>(ext.getType())) {
                                    toLock[&scatterOp.getRef().getColumn()].push_back({ext, scatterOp.getOperation()});
                                 } else {
                                    canBeParallel = false;
                                 }
                              } else {
                                 canBeParallel = false;
                              }
                           } else {
                              canBeParallel = false;
                              llvm::dbgs() << "unknown op to handle:\n"
                                           << problematicOp.op << "\n";
                           }
                        }
                     }

                     if (canBeParallel) {
                        //llvm::dbgs() << "parallel: ";
                        //scanRefsOp.dump();
                        for (auto l : toThreadLocals) {
                           if (threadLocalNotPossibleAnymore.contains(l.first)) {
                              canBeParallel = false;
                              continue;
                           }
                           if (l.second.requiresCombine && !l.second.combineRegion) {
                              canBeParallel = false;
                              continue;
                           }
                        }
                        if (canBeParallel) {
                           scanRefsOp->setAttr("parallel", mlir::UnitAttr::get(&getContext()));
                           for (auto l : toThreadLocals) {
                              if (mlir::isa<subop::SimpleStateType>(l.first.getType())) {
                                 assert(l.second.combineRegion);
                              }
                              if (toThreadLocalsGlobal.contains(l.first)) {
                                 auto& globalInfo = toThreadLocalsGlobal[l.first];
                                 //todo: make sure that everything fits together
                                 if (l.second.combineRegion) {
                                    globalInfo.combineRegion = l.second.combineRegion;
                                 }
                                 if (l.second.compareRegion) {
                                    globalInfo.compareRegion = l.second.compareRegion;
                                 }
                                 globalInfo.shouldUse.insert(executionStepOp);
                              } else {
                                 toThreadLocalsGlobal.insert({l.first, {l.second.combineRegion, l.second.compareRegion, {executionStepOp}, l.second.requiresCombine, false}});
                              }
                           }
                           for (auto* mA : markAsAtomic) {
                              mA->setAttr("atomic", mlir::UnitAttr::get(&getContext()));
                           }
                           for (auto l : toLock) {
                              toLockGlobal[l.first].insert(toLockGlobal[l.first].end(), l.second.begin(), l.second.end());
                           }
                        }
                     }
                     if (!canBeParallel) {
                        for (auto x : executionStepOp.getOperands()) {
                           threadLocalNotPossibleAnymore.insert(x);
                        }
                        llvm::dbgs() << "not parallel: ";
                        scanRefsOp.dump();
                     }
                  }
               }
            }
         }

         for (auto toLock : toLockGlobal) {
            mlir::OpBuilder builder(&getContext());
            builder.setInsertionPoint(toLock.second.front().second);

            auto tupleStream = toLock.second.front().second->getOperand(0);
            auto lockOp = builder.create<subop::LockOp>(builder.getUnknownLoc(), tupleStream, colManager.createRef(toLock.first));

            auto stateVal = toLock.second.front().first;
            auto hashMapType = mlir::cast<subop::HashMapType>(stateVal.getType());

            auto newType = subop::HashMapType::get(builder.getContext(), hashMapType.getKeyMembers(), hashMapType.getValueMembers(), true);
            auto executionStep = mlir::cast<subop::ExecutionStepOp>(stateVal.getDefiningOp());
            auto returnOp = mlir::cast<subop::ExecutionStepReturnOp>(executionStep.getSubOps().front().getTerminator());
            auto createOp = mlir::cast<subop::GenericCreateOp>(returnOp.getOperand(0).getDefiningOp());
            createOp->getResult(0).setType(newType);
            executionStep->getResult(0).setType(newType);

            mlir::TypeConverter htTypeConverter;
            htTypeConverter.addConversion([&](subop::HashMapType mapType) {
               return subop::HashMapType::get(builder.getContext(), mapType.getKeyMembers(), mapType.getValueMembers(), true);
            });
            htTypeConverter.addConversion([&](subop::HashMapEntryRefType refType) {
               return subop::HashMapEntryRefType::get(refType.getContext(), subop::HashMapType::get(refType.getContext(), refType.getHashMap().getKeyMembers(), refType.getHashMap().getValueMembers(), true));
            });
            htTypeConverter.addConversion([&](subop::LookupEntryRefType lookupRefType) {
               return subop::LookupEntryRefType::get(lookupRefType.getContext(), mlir::cast<subop::LookupAbleState>(htTypeConverter.convertType(lookupRefType.getState())));
            });

            htTypeConverter.addConversion([&](subop::ListType listType) {
               return subop::ListType::get(listType.getContext(), mlir::cast<subop::StateEntryReference>(htTypeConverter.convertType(listType.getT())));
            });
            htTypeConverter.addConversion([&](subop::OptionalType optionalType) {
               return subop::OptionalType::get(optionalType.getContext(), mlir::cast<subop::StateEntryReference>(htTypeConverter.convertType(optionalType.getT())));
            });
            subop::SubOpStateUsageTransformer htTransformer(columnUsageAnalysis, &getContext(), [&](mlir::Operation* op, mlir::Type type) -> mlir::Type {
               return htTypeConverter.convertType(type);
            });
            for (auto& u : stateVal.getUses()) {
               auto step = mlir::cast<subop::ExecutionStepOp>(u.getOwner());
               htTransformer.updateValue(step.getSubOps().getArgument(u.getOperandNumber()), newType);
               step.getSubOps().getArgument(u.getOperandNumber()).setType(newType);
            }
            lockOp.setRefAttr(colManager.createRef(htTransformer.getNewColumn(toLock.first)));

            auto* block = new mlir::Block;
            auto tupleStream2 = block->addArgument(tuples::TupleStreamType::get(builder.getContext()), builder.getUnknownLoc());
            for (auto [val, op] : toLock.second) {
               op->remove();
               block->push_back(op);
            }
            for (auto lOp : toLock.second) {
               if (lOp.second->getOperand(0) == tupleStream) {
                  lOp.second->setOperand(0, tupleStream2);
               }
            }
            auto* lastOp = toLock.second.back().second;

            if (lastOp->getNumResults() > 0) {
               mlir::Value lastResult = lastOp->getResult(0);
               lastResult.replaceAllUsesWith(lockOp.getRes());
               builder.setInsertionPointToEnd(block);
               builder.create<tuples::ReturnOp>(builder.getUnknownLoc(), lastResult);
            } else {
               builder.setInsertionPointToEnd(block);
               builder.create<tuples::ReturnOp>(builder.getUnknownLoc());
            }
            lockOp.getNested().push_back(block);
         }

         for (auto toThreadLocal : toThreadLocalsGlobal) {
            assert(toThreadLocal.first.getDefiningOp());
            auto producingExecutionStep = mlir::cast<subop::ExecutionStepOp>(toThreadLocal.first.getDefiningOp());
            size_t resultIdx = 0;
            for (auto result : producingExecutionStep.getResults()) {
               if (result == toThreadLocal.first) {
                  break;
               }
               resultIdx++;
            }
            auto* createOp = mlir::cast<subop::ExecutionStepReturnOp>(producingExecutionStep.getSubOps().front().getTerminator()).getOperand(resultIdx).getDefiningOp();
            if (!mlir::isa<subop::State>(createOp->getResultTypes()[0])) continue;
            if (mlir::isa<subop::ThreadLocalType>(createOp->getResultTypes()[0])) continue;

            mlir::OpBuilder builder(&getContext());
            builder.setInsertionPoint(createOp);
            auto mergedType = createOp->getResultTypes()[0];
            auto threadLocalType = subop::ThreadLocalType::get(builder.getContext(), mlir::cast<subop::State>(createOp->getResultTypes()[0]));
            if (mlir::isa<subop::BufferType>(createOp->getResultTypes()[0])) {
               auto createThreadLocal = builder.create<subop::GenericCreateOp>(createOp->getLoc(), threadLocalType);
               createOp->getResult(0).replaceAllUsesWith(createThreadLocal.getRes());
            } else {
               auto createThreadLocal = builder.create<subop::CreateThreadLocalOp>(createOp->getLoc(), threadLocalType);
               auto* block = new mlir::Block;
               createThreadLocal.getInitFn().push_back(block);
               builder.setInsertionPointToStart(block);
               auto* clonedCreate = builder.clone(*createOp);
               clonedCreate->setAttr("allocateOnHeap", builder.getUnitAttr());
               builder.create<tuples::ReturnOp>(createOp->getLoc(), clonedCreate->getResult(0));
               createOp->getResult(0).replaceAllUsesWith(createThreadLocal.getRes());
            }

            auto loc = createOp->getLoc();
            createOp->erase();

            // A clear step (inserted by ReuseLoopScratchPass at the loop body top)
            // must reset all worker-local instances, so it consumes the state
            // thread-locally (isThreadLocal=false -> wrapped arg stays thread_local)
            // and is neither a parallel writer nor a merged consumer.
            auto isClearStep = [](subop::ExecutionStepOp step) {
               bool found = false;
               step.getSubOps().walk([&](subop::ClearOp) { found = true; });
               return found;
            };
            // Build a merge execution step that reduces the thread_local `tlValue`
            // into a single merged state, placed before `mergedUsers.front()`, and
            // redirect each (state-operand) use in `mergedUsers` to the merged result.
            auto buildMerge = [&](mlir::Value tlValue, std::vector<mlir::Operation*>& mergedUsers, llvm::function_ref<bool(mlir::OpOperand&)> isMergedOperand) {
               if (mergedUsers.empty()) return;
               std::sort(mergedUsers.begin(), mergedUsers.end(), [&](mlir::Operation* l, mlir::Operation* r) { return l->isBeforeInBlock(r); });
               builder.setInsertionPoint(mergedUsers.front());
               auto mergeStep = builder.create<subop::ExecutionStepOp>(loc, mergedType, tlValue, builder.getBoolArrayAttr({false}));
               {
                  mlir::OpBuilder::InsertionGuard guard(builder);
                  auto* block = new mlir::Block;
                  auto threadLocalVal = block->addArgument(threadLocalType, builder.getUnknownLoc());
                  mergeStep.getSubOps().push_back(block);
                  builder.setInsertionPointToStart(block);
                  auto mergeOp = builder.create<subop::MergeOp>(loc, mergedType, threadLocalVal);
                  if (toThreadLocal.second.combineRegion) {
                     mlir::IRMapping mapping;
                     toThreadLocal.second.combineRegion->cloneInto(&mergeOp.getCombineFn(), mapping);
                  }
                  if (toThreadLocal.second.compareRegion) {
                     mlir::IRMapping mapping;
                     toThreadLocal.second.compareRegion->cloneInto(&mergeOp.getEqFn(), mapping);
                  }
                  builder.create<subop::ExecutionStepReturnOp>(loc, mergeOp.getResult());
               }
               // Redirect only the matching operands of mergedUsers to the merged
               // result (the new merge step itself keeps consuming tlValue).
               for (mlir::Operation* u : mergedUsers)
                  for (auto& op : u->getOpOperands())
                     if (isMergedOperand(op)) op.set(mergeStep.getResult(0));
            };

            bool crossBlock = false;
            for (auto s : toThreadLocal.second.shouldUse) {
               if (s->getBlock() != producingExecutionStep->getBlock()) { crossBlock = true; break; }
            }

            if (!crossBlock) {
               // ===== In-block create: writers/consumers directly use the state. =====
               mlir::Operation* lastShouldUse = nullptr;
               for (auto step : toThreadLocal.second.shouldUse) {
                  if (!lastShouldUse || lastShouldUse->isBeforeInBlock(step.getOperation())) {
                     lastShouldUse = step.getOperation();
                  }
               }
               std::vector<mlir::Operation*> mergedUsers;
               for (auto& use : toThreadLocal.first.getUses()) {
                  auto usingExecutionStep = mlir::dyn_cast_or_null<subop::ExecutionStepOp>(use.getOwner());
                  if (usingExecutionStep && toThreadLocal.second.shouldUse.contains(usingExecutionStep)) {
                     std::vector<mlir::Attribute> attrs(usingExecutionStep.getIsThreadLocal().begin(), usingExecutionStep.getIsThreadLocal().end());
                     attrs[use.getOperandNumber()] = mlir::BoolAttr::get(&getContext(), true);
                     usingExecutionStep.setIsThreadLocalAttr(mlir::ArrayAttr::get(&getContext(), attrs));
                  } else if (lastShouldUse && usingExecutionStep &&
                             usingExecutionStep.getOperation()->isBeforeInBlock(lastShouldUse)) {
                     std::vector<mlir::Attribute> attrs(usingExecutionStep.getIsThreadLocal().begin(), usingExecutionStep.getIsThreadLocal().end());
                     attrs[use.getOperandNumber()] = mlir::BoolAttr::get(&getContext(), true);
                     usingExecutionStep.setIsThreadLocalAttr(mlir::ArrayAttr::get(&getContext(), attrs));
                  } else {
                     mergedUsers.push_back(use.getOwner());
                  }
               }
               buildMerge(toThreadLocal.first, mergedUsers, [&](mlir::OpOperand& op) {
                  return op.get() == toThreadLocal.first && std::find(mergedUsers.begin(), mergedUsers.end(), op.getOwner()) != mergedUsers.end();
               });
               producingExecutionStep->getResult(resultIdx).setType(threadLocalType);
            } else {
               // ===== Cross-loop create (hoisted by ReuseLoopScratchPass): writers,
               // clears and consumers live inside the loop body's nested execution
               // group, with the state threaded in (possibly at several arg slots).
               // We set the parallel-writer flags + insert one per-iteration merge,
               // then thread the thread_local type through every carrier with the
               // state-usage transformer (which respects the isThreadLocal flags and
               // naturally handles the duplicated threading edges). =====
               auto nestedGroup = (*toThreadLocal.second.shouldUse.begin()).getOperation()->getParentOfType<subop::NestedExecutionGroupOp>();
               if (!nestedGroup) { producingExecutionStep->getResult(resultIdx).setType(threadLocalType); continue; }
               llvm::SmallVector<mlir::Value> reprs;
               auto ngArgs = nestedGroup.getSubOps().getArguments();
               auto ngInputs = nestedGroup.getInputs();
               for (size_t i = 0; i < ngArgs.size() && i < ngInputs.size(); i++) {
                  if (resolveToRoot(ngInputs[i]) == toThreadLocal.first) reprs.push_back(ngArgs[i]);
               }
               if (reprs.empty()) { producingExecutionStep->getResult(resultIdx).setType(threadLocalType); continue; }

               std::vector<mlir::Operation*> mergedUsers;
               for (auto R : reprs) {
                  for (auto& use : R.getUses()) {
                     auto step = mlir::dyn_cast_or_null<subop::ExecutionStepOp>(use.getOwner());
                     if (step && isClearStep(step)) {
                        // keep isThreadLocal=false; transformer retypes arg to thread_local
                     } else if (step && toThreadLocal.second.shouldUse.contains(step)) {
                        std::vector<mlir::Attribute> attrs(step.getIsThreadLocal().begin(), step.getIsThreadLocal().end());
                        attrs[use.getOperandNumber()] = mlir::BoolAttr::get(&getContext(), true);
                        step.setIsThreadLocalAttr(mlir::ArrayAttr::get(&getContext(), attrs));
                     } else {
                        mergedUsers.push_back(use.getOwner());
                     }
                  }
               }
               llvm::DenseSet<mlir::Value> reprSet(reprs.begin(), reprs.end());
               buildMerge(reprs[0], mergedUsers, [&](mlir::OpOperand& op) {
                  return reprSet.contains(op.get()) && std::find(mergedUsers.begin(), mergedUsers.end(), op.getOwner()) != mergedUsers.end();
               });

               producingExecutionStep->getResult(resultIdx).setType(threadLocalType);
               subop::SubOpStateUsageTransformer transformer(columnUsageAnalysis, &getContext(), [](mlir::Operation*, mlir::Type t) -> mlir::Type { return t; });
               transformer.updateValue(toThreadLocal.first, threadLocalType);
            }
         }
      }
   void runOnOperation() override {
      auto columnUsageAnalysis = getAnalysis<subop::ColumnUsageAnalysis>();
      auto& colManager = getContext().getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

      std::vector<subop::ExecutionGroupOp> executionGroupOps;
      getOperation()->walk([&](subop::ExecutionGroupOp executionGroupOp) {
         executionGroupOps.push_back(executionGroupOp);
         return mlir::WalkResult::skip();
      });
      for (auto executionGroup : executionGroupOps) {
         parallelizeBlock(executionGroup.getSubOps().front(), columnUsageAnalysis, colManager);
         // Also parallelize steps inside loop bodies (split into nested groups).
         executionGroup->walk([&](subop::NestedExecutionGroupOp nestedGroup) {
            parallelizeBlock(nestedGroup.getRegion().front(), columnUsageAnalysis, colManager);
         });
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
subop::createParallelizePass() { return std::make_unique<ParallelizePass>(); }
