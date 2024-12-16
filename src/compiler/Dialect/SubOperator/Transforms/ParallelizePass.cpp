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

ExecutionStepAnalyzed analyze(subop::ExecutionStepOp executionStepOp, subop::ColumnCreationAnalysis& columnCreationAnalysis) {
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
   std::unordered_set<std::string> extMembers;
   for (auto [i, a] : llvm::zip(executionStepOp.getInputs(), executionStepOp.getSubOps().getArguments())) {
      auto argType = a.getType();
      extStates.insert({a, i});
      if (auto stateType = mlir::dyn_cast_or_null<subop::State>(argType)) {
         for (auto m : stateType.getMembers().getNames()) {
            extMembers.insert(mlir::cast<mlir::StringAttr>(m).str());
         }
      }
   }
   auto* firstOp = &*executionStepOp.getOps().begin();
   result.pipelineStart = mlir::cast<subop::SubOperator>(firstOp);

   std::unordered_map<std::string, std::vector<mlir::Operation*>> readMembers;
   std::unordered_map<std::string, std::vector<mlir::Operation*>> writtenMembers;

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
      auto isNested = [&](mlir::Value v) { return !extStates.contains(v); };
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
            auto creationOp = mlir::cast<subop::SubOperator>(columnCreationAnalysis.getColumnCreator(&reduceOp.getRef().getColumn()));
            auto accessesNestedState = llvm::all_of(creationOp->getOperands(), [&](mlir::Value v) { return mlir::isa<subop::State>(v.getType()) ? isNested(v) : true; });
            if (!accessesNestedState) {
               addProblematicOp(reduceOp, getCollisions(), creationOp);
            }
         } else if (auto scatterOp = mlir::dyn_cast_or_null<subop::ScatterOp>(pipelineOp)) {
            auto creationOp = mlir::cast<subop::SubOperator>(columnCreationAnalysis.getColumnCreator(&scatterOp.getRef().getColumn()));
            auto accessesNestedState = llvm::all_of(creationOp->getOperands(), [&](mlir::Value v) { return mlir::isa<subop::State>(v.getType()) ? isNested(v) : true; });
            if (!accessesNestedState) {
               addProblematicOp(scatterOp, getCollisions(), creationOp);
            }
         } else if (auto gatherOp = mlir::dyn_cast_or_null<subop::ScatterOp>(pipelineOp)) {
            auto creationOp = mlir::cast<subop::SubOperator>(columnCreationAnalysis.getColumnCreator(&gatherOp.getRef().getColumn()));
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
   void runOnOperation() override {
      std::vector<subop::ExecutionGroupOp> executionGroupOps;
      getOperation()->walk([&](subop::ExecutionGroupOp executionGroupOp) {
         executionGroupOps.push_back(executionGroupOp);
      });
      for (auto executionGroup : executionGroupOps) {
         auto columnCreationAnalysis = getAnalysis<subop::ColumnCreationAnalysis>();
         llvm::DenseMap<mlir::Value, GlobalThreadLocalInfo> toThreadLocalsGlobal;
         std::vector<std::pair<tuples::ColumnRefAttr, std::vector<mlir::Operation*>>> toLockGlobal;
         llvm::DenseSet<mlir::Value> threadLocalNotPossibleAnymore;
         for (auto& op : executionGroup.getSubOps().front()) {
            if (auto executionStepOp = mlir::dyn_cast_or_null<subop::ExecutionStepOp>(&op)) {
               llvm::DenseMap<mlir::Value, mlir::Value> extStates;
               for (auto [i, a] : llvm::zip(executionStepOp.getInputs(), executionStepOp.getSubOps().getArguments())) {
                  extStates.insert({a, i});
               }

               if (auto scanRefsOp = mlir::dyn_cast_or_null<subop::ScanRefsOp>(*executionStepOp.getSubOps().getOps().begin())) {
                  if (!scanRefsOp->hasAttr("sequential")) {
                     ExecutionStepAnalyzed analyzed = analyze(executionStepOp, columnCreationAnalysis);
                     std::unordered_set<mlir::Operation*> markAsAtomic;

                     llvm::DenseMap<mlir::Value, ToThreadLocalInfo> toThreadLocals;
                     bool canBeParallel = true;
                     for (auto& collisionGroup : analyzed.collisionGroups) {
                        //llvm::dbgs() << "Collision Group:\n";
                        for (auto& problematicOp : collisionGroup.ops) {
                           //llvm::dbgs() << "  Problematic Op: " << problematicOp.op << "\n";
                           if (auto materializeOp = mlir::dyn_cast_or_null<subop::MaterializeOp>(problematicOp.op.getOperation())) {
                              auto ext = extStates[materializeOp.getState()];
                              if (ext.getDefiningOp()) {
                                 toThreadLocals.insert({ext, {}});
                              } else {
                                 canBeParallel = false;
                              }
                           } else if (auto lookupOrInsert = mlir::dyn_cast_or_null<subop::LookupOrInsertOp>(problematicOp.op.getOperation())) {
                              auto ext = extStates[lookupOrInsert.getState()];
                              if (ext.getDefiningOp()) {
                                 toThreadLocals[ext].requiresCombine = true;
                                 toThreadLocals[ext].compareRegion = &lookupOrInsert.getEqFn();
                              } else {
                                 canBeParallel = false;
                              }
                           } else if (auto reduceOp = mlir::dyn_cast_or_null<subop::ReduceOp>(problematicOp.op.getOperation())) {
                              auto stateAccessing = problematicOp.stateAccessing;
                              if (auto lookupOp = mlir::dyn_cast_or_null<subop::LookupOp>(stateAccessing.getOperation())) {
                                 auto ext = extStates[lookupOp.getState()];
                                 if (ext.getDefiningOp() && mlir::isa<subop::SimpleStateType>(ext.getType()) && mlir::dyn_cast_or_null<subop::SimpleStateType>(ext.getType()).getMembers().getTypes().size() == reduceOp.getMembers().size() && !reduceOp.getCombine().empty()) {
                                    toThreadLocals[ext].requiresCombine = true;
                                    toThreadLocals[ext].combineRegion = &reduceOp.getCombine();
                                 } else {
                                    canBeParallel = false;
                                 }
                              } else if (auto lookupOrInsertOp = mlir::dyn_cast_or_null<subop::LookupOrInsertOp>(stateAccessing.getOperation())) {
                                 // lookupOrInsertOp will be handled either way
                                 auto ext = extStates[lookupOrInsertOp.getState()];
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
                              if (collisionGroup.ops.size() == 1) {
                                 markAsAtomic.insert(scatterOp);
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
                           //for (auto l : toLock) {
                           //   toLockGlobal.push_back(l);
                           //}
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
         /*
         for (auto toLock : toLockGlobal) {
            mlir::OpBuilder builder(&getContext());
            builder.setInsertionPoint(toLock.second.front());
            auto tupleStream = toLock.second.front()->getOperand(0);
            auto lockOp = builder.create<subop::LockOp>(builder.getUnknownLoc(), tupleStream, toLock.first);
            auto* block = new mlir::Block;
            auto tupleStream2 = block->addArgument(mlir::tuples::TupleStreamType::get(builder.getContext()), builder.getUnknownLoc());
            for (auto* op : toLock.second) {
               op->remove();
               block->push_back(op);
            }
            toLock.second.front()->setOperand(0, tupleStream2);
            auto* lastOp = toLock.second.back();

            if (lastOp->getNumResults() > 0) {
               mlir::Value lastResult = lastOp->getResult(0);
               lastResult.replaceAllUsesWith(lockOp.getRes());
               builder.setInsertionPointToEnd(block);
               builder.create<mlir::tuples::ReturnOp>(builder.getUnknownLoc(), lastResult);
            } else {
               builder.setInsertionPointToEnd(block);
               builder.create<mlir::tuples::ReturnOp>(builder.getUnknownLoc());
            }
            lockOp.getNested().push_back(block);
         }*/
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

            mlir::OpBuilder builder(&getContext());
            builder.setInsertionPoint(createOp);
            auto mergedType = createOp->getResultTypes()[0];
            auto threadLocalType = subop::ThreadLocalType::get(builder.getContext(), mlir::cast<subop::State>(createOp->getResultTypes()[0]));
            auto createThreadLocal = builder.create<subop::CreateThreadLocalOp>(createOp->getLoc(), threadLocalType);
            auto* block = new mlir::Block;
            createThreadLocal.getInitFn().push_back(block);
            builder.setInsertionPointToStart(block);
            auto* clonedCreate = builder.clone(*createOp);
            clonedCreate->setAttr("allocateOnHeap", builder.getUnitAttr());
            builder.create<tuples::ReturnOp>(createOp->getLoc(), clonedCreate->getResult(0));
            createOp->getResult(0).replaceAllUsesWith(createThreadLocal.getRes());

            auto loc = createOp->getLoc();
            createOp->erase();

            std::vector<mlir::Operation*> mergedUsers;
            for (auto& use : toThreadLocal.first.getUses()) {
               auto usingExecutionStep = mlir::dyn_cast_or_null<subop::ExecutionStepOp>(use.getOwner());
               if (usingExecutionStep && toThreadLocal.second.shouldUse.contains(usingExecutionStep)) {
                  std::vector<mlir::Attribute> attrs(usingExecutionStep.getIsThreadLocal().begin(), usingExecutionStep.getIsThreadLocal().end());
                  attrs[use.getOperandNumber()] = mlir::BoolAttr::get(&getContext(), true);
                  usingExecutionStep.setIsThreadLocalAttr(mlir::ArrayAttr::get(&getContext(), attrs));
                  //use.getOwner()->setOperand(use.getOperandNumber(), createThreadLocal.getResult());
               } else {
                  mergedUsers.push_back(use.getOwner());
               }
            }

            std::sort(mergedUsers.begin(), mergedUsers.end(), [&](mlir::Operation* left, mlir::Operation* right) {
               return left->isBeforeInBlock(right);
            });

            assert(mergedUsers.size() > 0);
            builder.setInsertionPoint(mergedUsers.front());
            auto mergeStep = builder.create<subop::ExecutionStepOp>(loc, mergedType, toThreadLocal.first, builder.getBoolArrayAttr({false}));
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
            toThreadLocal.first.replaceUsesWithIf(mergeStep.getResult(0), [&](mlir::OpOperand& op) {
               return std::find(mergedUsers.begin(), mergedUsers.end(), op.getOwner()) != mergedUsers.end();
            });
            producingExecutionStep->getResult(resultIdx).setType(threadLocalType);
         }
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
subop::createParallelizePass() { return std::make_unique<ParallelizePass>(); }