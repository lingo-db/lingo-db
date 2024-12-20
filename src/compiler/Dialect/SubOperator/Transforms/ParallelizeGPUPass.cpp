

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

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/IR/Use.h"
#include "llvm/Support/Debug.h"
namespace {
using namespace lingodb::compiler::dialect;
struct ProblematicOp {
   subop::SubOperator op;
   subop::SubOperator stateAccessing; // transitive dependency (e.g., creates a col from external state that op uses)
};

struct CollisionGroup {
   std::vector<ProblematicOp> ops;
};
struct ExecutionStepAnalyzed {
   subop::SubOperator pipelineStart;
   std::vector<CollisionGroup> collisionGroups; // many problematicOps can take part in each collision
   bool notParallel = false;
};

ExecutionStepAnalyzed analyze(subop::ExecutionStepOp executionStepOp, subop::ColumnCreationAnalysis& columnCreationAnalysis) {
   ExecutionStepAnalyzed result;
   llvm::EquivalenceClasses<mlir::Operation*> equivalenceClasses;
   std::vector<ProblematicOp> problematicOps;
   llvm::DenseMap<mlir::Operation*, size_t> opToProblematicOpsIndex; // ProblematicOp is too big to be the value for an efficient llvm::DenseMap

   std::function<void(subop::SubOperator subop, const std::unordered_set<mlir::Operation*>& collisions, subop::SubOperator stateAccessing)> addProblematicOp = [&](subop::SubOperator subop, const std::unordered_set<mlir::Operation*>& collisions, subop::SubOperator stateAccessing) {
      // Aggregate ops into an equivalence class that denotes a set of subops that have RW conflicts on a member of a step-external state.
      // Example of adding a new conflict:
      //  We have recorded three lookups on some member A, then comes scatter that writes to A ->
      //   scatter is inserted and unionized with lookups (that are implicitly inserted).
      //   Then comes another lookup, for which the scatter is reported as a collision ->
      //    the new lookup is inserted and unionized through scatter with the previous three lookups -> an equivalence class now has 5 operations.
      mlir::Operation* subopAsOpPtr = subop.getOperation();
      problematicOps.push_back({subop, stateAccessing});
      opToProblematicOpsIndex[subopAsOpPtr] = problematicOps.size() - 1;
      equivalenceClasses.insert(subopAsOpPtr);
      for (auto* c : collisions) {
         equivalenceClasses.unionSets(subopAsOpPtr, c);
      }
      if (stateAccessing) {
         equivalenceClasses.unionSets(subopAsOpPtr, stateAccessing.getOperation());
      }
   };
   // Stage 1: Gather step inputs and, if an input is a state, its members as well.
   std::unordered_set<subop::Member> stepExternalStateMembers;
   llvm::DenseMap<mlir::Value, mlir::Value> stepSubopRegionArgsToStepInputs;
   for (auto [i, a] : llvm::zip(executionStepOp.getInputs(), executionStepOp.getSubOps().getArguments())) {
      auto argType = a.getType();
      stepSubopRegionArgsToStepInputs.insert({a, i});
      if (auto stateType = mlir::dyn_cast_or_null<subop::State>(argType)) {
         for (auto m : stateType.getMembers().getMembers()) {
            stepExternalStateMembers.insert(m);
         }
      }
   }
   // Stage 2: Gather all suboperators of a step region, excluding nested steps and the first suboperator.
   auto* firstOp = &*executionStepOp.getOps().begin();
   result.pipelineStart = mlir::cast<subop::SubOperator>(firstOp);
   std::vector<subop::SubOperator> pipelineOps;
   executionStepOp.getSubOps().walk([&](subop::SubOperator subOp) {
      if (firstOp != subOp.getOperation() && !mlir::isa<subop::ExecutionStepOp>(subOp.getOperation())) {
         pipelineOps.push_back(subOp);
      }
   });

   // Stage 3: Walk over suboperators from Stage 2 and, for those known to cause conflicts,
   //  analyze if a subop has RW conflict on any member of a step input state.
   std::unordered_map<subop::Member, std::vector<mlir::Operation*>> externalStateMembersReadBySubopsSoFar;
   std::unordered_map<subop::Member, std::vector<mlir::Operation*>> externalStateMembersWrittenBySubopsSoFar;
   auto getExternalStateMemberReadWriteCollisionOps = [&](subop::SubOperator subOp) {
      std::unordered_set<mlir::Operation*> collisions;
      auto currentReadMembers = subOp.getReadMembers();
      auto currentWrittenMembers = subOp.getWrittenMembers();
      for (auto r : currentReadMembers) {
         if (externalStateMembersWrittenBySubopsSoFar.contains(r)) { // potential READ AFTER WRITE
            collisions.insert(externalStateMembersWrittenBySubopsSoFar[r].begin(), externalStateMembersWrittenBySubopsSoFar[r].end());
         }
      }
      for (auto w : currentWrittenMembers) {
         if (externalStateMembersReadBySubopsSoFar.contains(w)) { // potential WRITE AFTER READ
            collisions.insert(externalStateMembersReadBySubopsSoFar[w].begin(), externalStateMembersReadBySubopsSoFar[w].end());
         }
      }
      for (auto r : currentReadMembers) {
         if (stepExternalStateMembers.contains(r)) {
            externalStateMembersReadBySubopsSoFar[r].push_back(subOp);
         }
      }
      for (auto r : currentWrittenMembers) {
         if (stepExternalStateMembers.contains(r)) {
            externalStateMembersWrittenBySubopsSoFar[r].push_back(subOp);
         }
      }
      return collisions;
   };

   // Nested states (not part of step inputs) cannot cause RW conflicts, as they are not shared among step executors.
   auto isNested = [&](mlir::Value v) { return !stepSubopRegionArgsToStepInputs.contains(v); };

   for (auto subOp : pipelineOps) {
      auto* pipelineOp = subOp.getOperation();
      auto handleColCreators = [&](subop::SubOperator subOp, subop::SubOperator colCreationSubOp) {
         auto accessesOnlyNestedState = llvm::all_of(colCreationSubOp->getOperands(), [&](mlir::Value v) { return mlir::isa<subop::State>(v.getType()) ? isNested(v) : true; });
         if (!accessesOnlyNestedState) { // if a column is created using states that are not step-local
            addProblematicOp(subOp, getExternalStateMemberReadWriteCollisionOps(subOp), colCreationSubOp); // keep track of the column creating op and column consumer op -> transitive dependency
         }
      };

      // A subop can only be problematic if it uses some state.
      if (mlir::isa<subop::StateUsingSubOperator>(pipelineOp)) {
         // Check a set of suboperators that are known to cause RW conflicts.
         if (mlir::isa<subop::ScanListOp, subop::NestedMapOp>(pipelineOp)) {
            // ignore
         } else if (auto materializeOp = mlir::dyn_cast_or_null<subop::MaterializeOp>(pipelineOp)) {
            if (!isNested(materializeOp.getState())) {
               addProblematicOp(materializeOp, getExternalStateMemberReadWriteCollisionOps(subOp), {});
            }
         } else if (auto lookupOp = mlir::dyn_cast_or_null<subop::LookupOp>(pipelineOp)) {
            // Example: we want to lookup step's input state, but some subop in this step before us writes to it -> read after write conflict
            if (!isNested(lookupOp.getState())) { // lookup external is ok, unless someone in a step writes to members we lookup.
               auto collisions = getExternalStateMemberReadWriteCollisionOps(subOp); // only aggregates members accessed by pipelineOps visited so far
               if (!collisions.empty()) {
                  addProblematicOp(lookupOp, collisions, {});
               }
            }
         } else if (auto lookupOrInsertOp = mlir::dyn_cast_or_null<subop::LookupOrInsertOp>(pipelineOp)) {
            if (!isNested(lookupOrInsertOp.getState())) {
               addProblematicOp(lookupOrInsertOp, getExternalStateMemberReadWriteCollisionOps(subOp), {});
            }
         } else if (auto reduceOp = mlir::dyn_cast_or_null<subop::ReduceOp>(pipelineOp)) { // if subops write to a column that was created using external state
            auto creationOp = mlir::cast<subop::SubOperator>(columnCreationAnalysis.getColumnCreator(&reduceOp.getRef().getColumn()));
            handleColCreators(reduceOp, creationOp);
         } else if (auto scatterOp = mlir::dyn_cast_or_null<subop::ScatterOp>(pipelineOp)) {
            auto creationOp = mlir::cast<subop::SubOperator>(columnCreationAnalysis.getColumnCreator(&scatterOp.getRef().getColumn()));
            handleColCreators(scatterOp, creationOp);
         } else if (auto gatherOp = mlir::dyn_cast_or_null<subop::GatherOp>(pipelineOp)) {
            auto creationOp = mlir::cast<subop::SubOperator>(columnCreationAnalysis.getColumnCreator(&gatherOp.getRef().getColumn()));
            handleColCreators(gatherOp, creationOp);
         } else {
            //problem: don't know how to handle
            result.notParallel = true;
         }
      } else {
         //every other operation is not really important
      }
   }

   // Stage 4: Materialize problematic operations that collide on the same member.
   for (auto it = equivalenceClasses.begin(), itEnd = equivalenceClasses.end(); it != itEnd; ++it) {
      if (it->isLeader()) {
         CollisionGroup collisionGroup;

         for (auto member = equivalenceClasses.member_begin(it); member != equivalenceClasses.member_end(); ++member) {
            auto* memberOp = *member;
            if (opToProblematicOpsIndex.contains(memberOp))
               collisionGroup.ops.push_back(problematicOps[opToProblematicOpsIndex[memberOp]]);
         }
         result.collisionGroups.push_back(collisionGroup);
      }
   }
   return result; // Per each member that causes a collision, get a set of colliding ops.
}

/*
   ------------------ CPU ------------------
   In CPU parallelization, for each scanref of a step of a group:
      1. We detect states that have any RW conflicts.
      2. We create their thread local versions and replace the orginial state usages with thread local.
      3. Thread local handling is the task of the subsequent SubopToControlFlow.

   ------------------ GPU ------------------
   Since CPU controls the data management and control flow, it always has a GPU state handle that
      is used as an input to execution steps. A handle represents ONE "global" instance of a state.

   GPU steps can be executed by kernels with variyng launch configs, the meaning of state locality is therefore different
      for each step. Because of that, we do not replace the state creation op, but modify steps that consume it.

   GPU parallelization only encapsulates the way GPU processes a state within a step context.

   A GPU step that receives a "global" state handle needs to "localize" the state in case of RW conflicts:
      1. Before scanRef: add createKernelLocal subop that will expand "global" state to thread/warp/thread-block local states.
      2. For scanRef: replace input step with a kernelLocal one.
      3. After scanRef: add mergeInto subop that will merge kernelLocal states into a "global" state.

   Control flow goes back to CPU with a global handle to a consistent state.

   Q: The same state can be used in two (subsequent) steps that both modify it.
      How shall the "localization" work for the second step?
   A: Atomic/Lock, non-parallel.
   Q: Why not lock-free? If we do not need an intermediate merge, can't we just reuse kernel_local?
   A: No (or complex yes), the kernel_local size (i.e., how many "local" instances we have) depends on kernel launch
      config that can differ and stems from runtime. Hence, the second step would need to "map" between old kernel_local config and current kernel_local config.
   Q: Alright, but how bad is locking in subsequent steps for performance?
   A: Depends on the granularity of updates, if each thread wants to independently modify arbitrary state entries, it might indeed be problematic.
      Need a concrete workload to actually see the impact.
*/

class ParallelizeGPUPass : public mlir::PassWrapper<ParallelizeGPUPass, mlir::OperationPass<>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ParallelizeGPUPass)
   virtual llvm::StringRef getArgument() const override { return "subop-parallelize-gpu"; }

   struct ToLocalInfo {
      mlir::Region* combineRegion = nullptr;
      mlir::Region* compareRegion = nullptr;
      bool requiresCombine = false;
   };

   struct AggregatedLocalStateInfo {
      mlir::Region* combineRegion{nullptr};
      mlir::Region* compareRegion{nullptr};
      subop::ExecutionStepOp firstUse{nullptr};
      llvm::DenseSet<subop::ExecutionStepOp> laterUses;
      bool requiresCombine{false};
      bool isClosed{false};
   };

   void runOnOperation() override {
      std::vector<subop::ExecutionGroupOp> executionGroupOps;
      getOperation()->walk([&](subop::ExecutionGroupOp executionGroupOp) {
         executionGroupOps.push_back(executionGroupOp);
      });
      for (auto executionGroupOp : executionGroupOps) {
         auto columnCreationAnalysis = getAnalysis<subop::ColumnCreationAnalysis>(); // used for conflict analysis
         llvm::DenseMap<subop::ExecutionStepOp, llvm::DenseSet<mlir::Value>> stepStatesToBecomeKernelLocal;
         llvm::DenseMap<mlir::Value, AggregatedLocalStateInfo> statesToBecomeKernelLocalInfo;
         executionGroupOp->walk<mlir::WalkOrder::PreOrder>([&](subop::ExecutionStepOp executionStepOp) {
            // We still need to map arguments to inputs, because through arguments we want to get to the defining op of the input.
            // However, unlike in CPU case, we will not rewrite the defining op, but will clone its init fn.
            llvm::DenseMap<mlir::Value, mlir::Value> extStates;
            for (auto [i, a] : llvm::zip(executionStepOp.getInputs(), executionStepOp.getSubOps().getArguments())) {
               extStates.insert({a, i});
            }
            auto scanRefsOp = mlir::dyn_cast_or_null<subop::ScanRefsOp>(*executionStepOp.getSubOps().getOps().begin());
            llvm::DenseMap<mlir::Value, ToLocalInfo> statesToBecomeKernelLocals;
            std::unordered_set<mlir::Operation*> markAsAtomic;
            if (executionStepOp.isOnGPU() && scanRefsOp && !scanRefsOp->hasAttr("sequential")) {
               bool canBeParallel = true;
               ExecutionStepAnalyzed statesRWConflictAnalysis = analyze(executionStepOp, columnCreationAnalysis);
               // llvm::dbgs() << "Step " << executionStepOp << "\n";
               // If a step with scan refs has any collisions, resolve them.
               for (auto& collisionGroup : statesRWConflictAnalysis.collisionGroups) {
                  // llvm::dbgs() << "  Collision Group:\n";
                  for (auto& problematicOp : collisionGroup.ops) {
                     // llvm::dbgs() << "    Problematic Op: " << problematicOp.op << "\n";
                     // A set of operators known to cause collisions. We know that their states are external (otherwise no collision).
                     if (auto lookupOrInsertOp = mlir::dyn_cast_or_null<subop::LookupOrInsertOp>(problematicOp.op.getOperation())) {
                        auto state = lookupOrInsertOp.getState(); // state here is block argument
                        statesToBecomeKernelLocals[state].requiresCombine = true;
                        statesToBecomeKernelLocals[state].compareRegion = &lookupOrInsertOp.getEqFn();
                     } else if (auto materializeOp = mlir::dyn_cast_or_null<subop::MaterializeOp>(problematicOp.op.getOperation())) {
                        auto state = materializeOp.getState(); // state here is block argument
                        statesToBecomeKernelLocals.insert({state, {}});
                     } else if (auto reduceOp = mlir::dyn_cast_or_null<subop::ReduceOp>(problematicOp.op.getOperation())) {
                        auto stateAccessing = problematicOp.stateAccessing; // we address column producer of a reduce op argument (e.g., lookup produces a reference to a column of simple state)
                        if (auto lookupOp = mlir::dyn_cast_or_null<subop::LookupOp>(stateAccessing.getOperation())) {
                           auto state = lookupOp.getState();
                           // If we reduce all columns of a simple state
                           if (mlir::isa<subop::SimpleStateType>(state.getType()) && mlir::dyn_cast_or_null<subop::SimpleStateType>(state.getType()).getMembers().getMembers().size() == reduceOp.getMembers().size() && !reduceOp.getCombine().empty()) {
                              statesToBecomeKernelLocals[state].requiresCombine = true;
                              statesToBecomeKernelLocals[state].combineRegion = &reduceOp.getCombine();
                           }
                        } else if (auto lookupOrInsertOp = mlir::dyn_cast_or_null<subop::LookupOrInsertOp>(stateAccessing.getOperation())) {
                           auto state = lookupOrInsertOp.getState();
                           statesToBecomeKernelLocals[state].requiresCombine = true;
                           statesToBecomeKernelLocals[state].combineRegion = &reduceOp.getCombine();
                        }
                     } else if (auto scatterOp = mlir::dyn_cast_or_null<subop::ScatterOp>(problematicOp.op.getOperation())) {
                        if (collisionGroup.ops.size() == 1) {
                           markAsAtomic.insert(scatterOp);
                        } else {
                           canBeParallel = false;
                        }
                     } else if (auto gatherOp = mlir::dyn_cast_or_null<subop::GatherOp>(problematicOp.op.getOperation())) {
                     } else {
                        canBeParallel = false;
                        llvm::dbgs() << "unknown op to handle:\n"
                                     << problematicOp.op << "\n";
                     }
                  }
               }
               if (canBeParallel) { // If there were no unknown ops in collisions, we should have gathered all states that need to become kernel-local for this step
                  scanRefsOp->setAttr("parallel", mlir::UnitAttr::get(&getContext()));
                  for (auto [state, info] : statesToBecomeKernelLocals) {
                     if (statesToBecomeKernelLocalInfo.contains(state)) {
                        auto& globalInfo = statesToBecomeKernelLocalInfo[state];
                        if (info.combineRegion) {
                           globalInfo.combineRegion = info.combineRegion;
                        }
                        if (info.compareRegion) {
                           globalInfo.compareRegion = info.compareRegion;
                        }
                        globalInfo.laterUses.insert(executionStepOp);
                     } else {
                        statesToBecomeKernelLocalInfo.insert({state, {info.combineRegion, info.compareRegion, {executionStepOp}, {}, info.requiresCombine, false}});
                     }
                     stepStatesToBecomeKernelLocal[executionStepOp].insert(state);
                  }
               }
               for (auto* mA : markAsAtomic) {
                  mA->setAttr("atomic", mlir::UnitAttr::get(&getContext()));
               }
               if (!canBeParallel) {
                  // llvm::dbgs() << "not parallel: ";
                  scanRefsOp.dump();
               }
            }
            // By now we know per step what states need to be localized. For each such state we also have relevant info.
            // llvm::dbgs() << "[Localizing] Step " << executionStepOp << "\n";
            for (auto [stateToLocalize, info] : statesToBecomeKernelLocals) {
               // llvm::dbgs() << "[Localizing] stateToLocalize = " << stateToLocalize << "\n";
               auto& globalInfo = statesToBecomeKernelLocalInfo[stateToLocalize];
               subop::ExecutionStepOp stateFirstWriterStep = globalInfo.firstUse;
               assert(executionStepOp == globalInfo.firstUse && "[GPU parallelization] For now, only uninitialized states will be localized (i.e., only one wiriting allowed).");
               auto scanRefsOp = mlir::dyn_cast_or_null<subop::ScanRefsOp>(*stateFirstWriterStep.getSubOps().getOps().begin());
               assert(scanRefsOp);
               mlir::OpBuilder builder(&getContext());
               if (executionStepOp == stateFirstWriterStep) {
                  {
                     mlir::OpBuilder::InsertionGuard guard(builder);
                     // Within each GPU step that has localizable untouched states
                     // Step 1: create kernel local state handle
                     builder.setInsertionPointToStart(scanRefsOp->getBlock());
                     // llvm::dbgs() << "[Localizing] Insertion point is in block ";
                     // scanRefsOp->getBlock()->dump();
                     // llvm::dbgs() << "\n";
                     auto stateType = mlir::cast<subop::State>(stateToLocalize.getType());
                     auto kernelLocalType = subop::KernelLocalType::get(builder.getContext(), stateType);
                     subop::KernelLocalityLvl stateLocality = subop::KernelLocalityLvl::thread;
                     if (mlir::isa<subop::SimpleStateType>(stateType)) {
                        stateLocality = subop::KernelLocalityLvl::thread;
                     } else if (mlir::isa<subop::BufferType>(stateType)) {
                        stateLocality = subop::KernelLocalityLvl::warp;
                     } else if (mlir::isa<subop::HashMapType>(stateType)) { // can be further specialized to PreAggregationHTFrag
                        stateLocality = subop::KernelLocalityLvl::thread_block;
                     }
                     mlir::Value createKernelLocal = builder.create<subop::CreateKernelLocalOp>(executionStepOp->getLoc(), kernelLocalType, stateLocality);
                     // llvm::dbgs() << "[Localizing] createKernelLocal = " << *createKernelLocal.getDefiningOp() << "\n";
                     auto createKernelLocalOp = mlir::cast<subop::CreateKernelLocalOp>(createKernelLocal.getDefiningOp());
                     // Step 2: let each locality level initialize the local state
                     auto producingStep = mlir::cast<subop::ExecutionStepOp>(extStates[stateToLocalize].getDefiningOp()); // stateToLocalize should not be a block argument (!)
                     // llvm::dbgs() << "[Localizing] producingStep = " << producingStep << "\n";
                     auto* block = new mlir::Block;
                     createKernelLocalOp.getInitFn().push_back(block);
                     builder.setInsertionPointToStart(block);
                     auto producingStepTerminator = mlir::cast<subop::ExecutionStepReturnOp>(producingStep.getSubOps().front().getTerminator());
                     auto* producingStepTerminatorInputDefOp = producingStepTerminator.getInputs().front().getDefiningOp();
                     assert(mlir::isa<subop::StateContextSwitchOp>(producingStepTerminatorInputDefOp) && "Input state of a GPU step must come from a context switch!");
                     auto* producingStepStateCreationOp = producingStepTerminatorInputDefOp->getOperand(0).getDefiningOp();
                     auto* clonedCreate = builder.clone(*producingStepStateCreationOp);
                     builder.create<tuples::ReturnOp>(executionStepOp->getLoc(), clonedCreate->getResult(0));
                     // llvm::dbgs() << "[Localizing] clonedCreate = " << *clonedCreate << "\n";
                     // Step 3: replace usages of the global state within the step (except for CreateKernelLocalOp), now ops (e.g., reduce) should refer to kernel_local
                     stateToLocalize.replaceUsesWithIf(createKernelLocal, [&](mlir::OpOperand& use) -> bool {
                        return use.getOwner() != createKernelLocal.getDefiningOp();
                     });
                     // Step 4: insert a merge to global value at the end to ensure that CPU has a handle to a consistent (reduced) state
                     auto stepTerminator = mlir::cast<subop::ExecutionStepReturnOp>(executionStepOp.getSubOps().front().getTerminator());
                     builder.setInsertionPoint(stepTerminator);
                     subop::MergeIntoOp mergeintoop = builder.create<subop::MergeIntoOp>(executionStepOp->getLoc(), stateToLocalize, createKernelLocal);
                     if (info.combineRegion) {
                        mlir::IRMapping mapping;
                        info.combineRegion->cloneInto(&mergeintoop.getCombineFn(), mapping);
                     }
                     if (info.compareRegion) {
                        mlir::IRMapping mapping;
                        info.compareRegion->cloneInto(&mergeintoop.getEqFn(), mapping);
                     }
                     // llvm::dbgs() << "[Localizing] mergeintoop = " << mergeintoop << "\n";
                  }
               }
            }
         });
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> lingodb::compiler::dialect::subop::createParallelizeGPUPass() { return std::make_unique<ParallelizeGPUPass>(); }