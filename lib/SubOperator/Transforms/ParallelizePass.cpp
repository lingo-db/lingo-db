#include "mlir/Dialect/SubOperator/SubOperatorInterfaces.h"
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
namespace {

class ParallelizePass : public mlir::PassWrapper<ParallelizePass, mlir::OperationPass<>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ParallelizePass)
   virtual llvm::StringRef getArgument() const override { return "subop-parallelize"; }
   struct ToThreadLocalInfo {
      mlir::Region* combineRegion = nullptr;
      mlir::Region* compareRegion = nullptr;
      bool requiresCombine = false;
      size_t numUses = 0;
   };
   struct GlobalThreadLocalInfo {
      mlir::Region* combineRegion = nullptr;
      mlir::Region* compareRegion = nullptr;
      llvm::DenseSet<mlir::subop::ExecutionStepOp> shouldUse;
      bool requiresCombine = false;
      bool isClosed = false;
   };
   void runOnOperation() override {
      std::vector<mlir::subop::ExecutionGroupOp> executionGroupOps;
      getOperation()->walk([&](mlir::subop::ExecutionGroupOp executionGroupOp) {
         executionGroupOps.push_back(executionGroupOp);
      });
      for (auto executionGroup : executionGroupOps) {
         auto columnCreationAnalysis = getAnalysis<mlir::subop::ColumnCreationAnalysis>();
         llvm::DenseMap<mlir::Value, GlobalThreadLocalInfo> toThreadLocalsGlobal;
         std::vector<std::pair<mlir::tuples::ColumnRefAttr, std::vector<mlir::Operation*>>> toLockGlobal;
         executionGroup.getSubOps().walk([&](mlir::subop::ExecutionStepOp executionStepOp) {
            llvm::DenseMap<mlir::Value, mlir::Value> extStates;

            std::unordered_set<std::string> extMembers;
            for (auto [i, a] : llvm::zip(executionStepOp.getInputs(), executionStepOp.getSubOps().getArguments())) {
               auto argType = a.getType();
               extStates.insert({a, i});
               if (auto stateType = argType.dyn_cast_or_null<mlir::subop::State>()) {
                  for (auto m : stateType.getMembers().getNames()) {
                     extMembers.insert(m.cast<mlir::StringAttr>().str());
                  }
               }
            }
            auto* firstOp = &*executionStepOp.getOps().begin();
            if (auto scanRefsOp = mlir::dyn_cast_or_null<mlir::subop::ScanRefsOp>(firstOp)) {
               if (!scanRefsOp->hasAttr("sequential")) {
                  std::unordered_map<std::string, std::vector<mlir::Operation*>> readMembers;
                  std::unordered_map<std::string, std::vector<mlir::Operation*>> writtenMembers;

                  std::vector<mlir::subop::SubOperator> pipelineOps;
                  executionStepOp.getSubOps().walk([&](mlir::subop::SubOperator subOp) {
                     if (scanRefsOp.getOperation() != subOp.getOperation() && !mlir::isa<mlir::subop::ExecutionStepOp>(subOp.getOperation())) {
                        pipelineOps.push_back(subOp);
                     }
                  });
                  llvm::DenseMap<mlir::Value, ToThreadLocalInfo> toThreadLocals;
                  std::vector<std::pair<mlir::tuples::ColumnRefAttr, std::vector<mlir::Operation*>>> toLock;
                  std::unordered_set<mlir::Operation*> markAsAtomic;
                  bool canBeParallel = true;

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
                     auto isStreamTypeOrNested = [&](mlir::Value v) {
                        if (v.getType().isa<mlir::tuples::TupleStreamType>()) return true;
                        if (isNested(v)) return true;
                        return false;
                     };
                     if (auto materializeOp = mlir::dyn_cast_or_null<mlir::subop::MaterializeOp>(pipelineOp)) {
                        if (getCollisions().empty()) {
                           if (!isNested(materializeOp.getState())) {
                              auto ext = extStates[materializeOp.getState()];
                              if (ext.getDefiningOp()) {
                                 toThreadLocals[ext].numUses++;
                                 toThreadLocals.insert({ext, {}});
                                 continue;
                              }
                           } else {
                              continue;
                           }
                        }
                     } else if (mlir::isa<mlir::subop::LookupOp>(pipelineOp)) {
                        if (getCollisions().empty()) {
                           //ignore lookupOps
                           continue;
                        }
                     } else if (auto lookupOrInsertOp = mlir::dyn_cast_or_null<mlir::subop::LookupOrInsertOp>(pipelineOp)) {
                        if (!isNested(lookupOrInsertOp.getState())) {
                           if (getCollisions().empty()) {
                              auto ext = extStates[lookupOrInsertOp.getState()];
                              if (ext.getDefiningOp()) {
                                 toThreadLocals[ext].compareRegion = &lookupOrInsertOp.getEqFn();
                                 toThreadLocals[ext].numUses++;
                                 toThreadLocals[ext].requiresCombine = true;
                                 continue;
                              }
                           }
                        } else {
                           continue;
                        }
                     } else if (mlir::isa<mlir::subop::GatherOp, mlir::subop::ScanRefsOp>(pipelineOp)) {
                        if (getCollisions().empty()) {
                           //ignore gathers
                           continue;
                        }
                     } else if (mlir::isa<mlir::subop::ScanListOp, mlir::subop::CombineTupleOp, mlir::subop::GetBeginReferenceOp, mlir::subop::GetEndReferenceOp, mlir::subop::EntriesBetweenOp, mlir::subop::OffsetReferenceBy>(pipelineOp)) {
                        continue;
                     } else if (auto reduceOp = mlir::dyn_cast<mlir::subop::ReduceOp>(pipelineOp)) {
                        //todo: order of members in reduce fn must match member order in state
                        auto collisions = getCollisions();
                        if (collisions.size() == 0) {
                           //todo: needs more detailed handling
                           if (reduceOp.getRef().getColumn().type.isa<mlir::subop::ContinuousEntryRefType>()) {
                              if (reduceOp.getMembers().size() == 1 && reduceOp.getRegion().front().getArguments().back().getType().isIntOrFloat()) {
                                 //we can perform generic atomic
                                 markAsAtomic.insert(pipelineOp);
                                 continue;
                              }
                           } else {
                              continue;
                           }
                        }
                        if (collisions.size() == 1) {
                           if (auto lookupOp = mlir::dyn_cast<mlir::subop::LookupOp>(*collisions.begin())) {
                              if (!isNested(lookupOp.getState())) {
                                 auto ext = extStates[lookupOp.getState()];
                                 if (ext.getDefiningOp()) {
                                    if (auto simpleStateType = ext.getType().dyn_cast_or_null<mlir::subop::SimpleStateType>()) {
                                       if (simpleStateType.getMembers().getTypes().size() == reduceOp.getMembers().size() && !reduceOp.getCombine().empty()) {
                                          toThreadLocals[ext].combineRegion = &reduceOp.getCombine();
                                          toThreadLocals[ext].numUses++;
                                          continue;
                                       }
                                    }
                                    if (reduceOp.getMembers().size() == 1 && reduceOp.getRegion().front().getArguments().back().getType().isIntOrFloat()) {
                                       //we can perform generic atomic
                                       markAsAtomic.insert(pipelineOp);
                                       continue;
                                    }
                                    if (auto hashMapType = ext.getType().dyn_cast_or_null<mlir::subop::HashMapType>()) {
                                       //todo: reactivate explicit synchronization
                                       //toLock.push_back({reduceOp.getRef(), {reduceOp}});
                                       //continue;
                                    }
                                 }

                              } else {
                                 continue;
                              }
                           }
                           if (auto lookupOrInsertOp = mlir::dyn_cast<mlir::subop::LookupOrInsertOp>(*collisions.begin())) {
                              if (!isNested(lookupOrInsertOp.getState())) {
                                 auto ext = extStates[lookupOrInsertOp.getState()];
                                 if (ext.getDefiningOp()) {
                                    if (auto hashMapType = ext.getType().dyn_cast_or_null<mlir::subop::HashMapType>()) {
                                       if (hashMapType.getValueMembers().getTypes().size() == reduceOp.getMembers().size() && !reduceOp.getCombine().empty()) {
                                          toThreadLocals[ext].combineRegion = &reduceOp.getCombine();
                                          toThreadLocals[ext].numUses++;
                                          continue;
                                       }
                                    }
                                 }
                              } else {
                                 continue;
                              }
                           }
                        }

                     } else if (auto scatterOp = mlir::dyn_cast_or_null<mlir::subop::ScatterOp>(pipelineOp)) {
                        auto collisions = getCollisions();
                        if (collisions.size() == 0) {
                           continue;
                        }
                        if (collisions.size() == 1) {
                           if (auto lookupOp = mlir::dyn_cast<mlir::subop::LookupOp>(*collisions.begin())) {
                              markAsAtomic.insert(pipelineOp);
                              continue;
                           }
                        }
                     } else if (llvm::all_of(pipelineOp->getOperands(), isStreamTypeOrNested)) {
                        //ignore: do not interact with states
                        continue;
                     }
                     //llvm::dbgs() << "problematic operation:";
                     //pipelineOp->dump();
                     canBeParallel = false;
                  }
                  //finally: mark as parallel
                  if (canBeParallel) {
                     scanRefsOp->setAttr("parallel", mlir::UnitAttr::get(&getContext()));
                     //llvm::dbgs() << "parallel: ";
                     //scanRefsOp.dump();

                     for (auto l : toThreadLocals) {
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
                        toLockGlobal.push_back(l);
                     }
                  } else {
                     //llvm::dbgs()<<"not parallel: ";
                     //scanRefsOp.dump();
                  }
               }
            }
         });
         /*
         for (auto toLock : toLockGlobal) {
            mlir::OpBuilder builder(&getContext());
            builder.setInsertionPoint(toLock.second.front());
            auto tupleStream = toLock.second.front()->getOperand(0);
            auto lockOp = builder.create<mlir::subop::LockOp>(builder.getUnknownLoc(), tupleStream, toLock.first);
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
            auto producingExecutionStep = mlir::cast<mlir::subop::ExecutionStepOp>(toThreadLocal.first.getDefiningOp());
            size_t resultIdx = 0;
            for (auto result : producingExecutionStep.getResults()) {
               if (result == toThreadLocal.first) {
                  break;
               }
               resultIdx++;
            }
            auto* createOp = mlir::cast<mlir::subop::ExecutionStepReturnOp>(producingExecutionStep.getSubOps().front().getTerminator()).getOperand(resultIdx).getDefiningOp();

            mlir::OpBuilder builder(&getContext());
            builder.setInsertionPoint(createOp);
            auto mergedType = createOp->getResultTypes()[0];
            auto threadLocalType = mlir::subop::ThreadLocalType::get(builder.getContext(), createOp->getResultTypes()[0].cast<mlir::subop::State>());
            auto createThreadLocal = builder.create<mlir::subop::CreateThreadLocalOp>(createOp->getLoc(), threadLocalType);
            auto* block = new mlir::Block;
            createThreadLocal.getInitFn().push_back(block);
            builder.setInsertionPointToStart(block);
            auto* clonedCreate = builder.clone(*createOp);
            clonedCreate->setAttr("allocateOnHeap", builder.getUnitAttr());
            builder.create<mlir::tuples::ReturnOp>(createOp->getLoc(), clonedCreate->getResult(0));
            createOp->getResult(0).replaceAllUsesWith(createThreadLocal.getRes());

            auto loc = createOp->getLoc();
            createOp->erase();

            std::vector<mlir::Operation*> mergedUsers;
            for (auto& use : toThreadLocal.first.getUses()) {
               auto usingExecutionStep = mlir::dyn_cast_or_null<mlir::subop::ExecutionStepOp>(use.getOwner());
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
            auto mergeStep = builder.create<mlir::subop::ExecutionStepOp>(loc, mergedType, toThreadLocal.first, builder.getBoolArrayAttr({false}));
            {
               mlir::OpBuilder::InsertionGuard guard(builder);
               auto* block = new mlir::Block;
               auto threadLocalVal = block->addArgument(threadLocalType, builder.getUnknownLoc());
               mergeStep.getSubOps().push_back(block);
               builder.setInsertionPointToStart(block);
               auto mergeOp = builder.create<mlir::subop::MergeOp>(loc, mergedType, threadLocalVal);
               if (toThreadLocal.second.combineRegion) {
                  mlir::IRMapping mapping;
                  toThreadLocal.second.combineRegion->cloneInto(&mergeOp.getCombineFn(), mapping);
               }
               if (toThreadLocal.second.compareRegion) {
                  mlir::IRMapping mapping;
                  toThreadLocal.second.compareRegion->cloneInto(&mergeOp.getEqFn(), mapping);
               }
               builder.create<mlir::subop::ExecutionStepReturnOp>(loc, mergeOp.getResult());
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
mlir::subop::createParallelizePass() { return std::make_unique<ParallelizePass>(); }