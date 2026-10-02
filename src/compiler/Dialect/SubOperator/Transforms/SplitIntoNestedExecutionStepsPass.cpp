// SplitIntoNestedExecutionStepsPass — analyze-first rewrite.
//
// Wraps each `ContainsNestedSubOps` body (NestedMapOp / LoopOp) in a
// `subop.nested_execution_group` containing per-pipeline `subop.execution_step`s.
//
// Counterpart of `OrganizeExecutionStepsPass` for inner bodies. Difference:
// bodies contain no `subop.union` (eliminated by InlineNestedMapPass), so each
// op has at most one stream producer and lives in exactly one pipeline (no
// clone-on-demand needed).
//
// Inter-pipeline ordering is decided by:
//   1) SSA edges between pipelines (always correct: producer-before-consumer
//      is enforced by MLIR for non-stream values).
//   2) An SSA-respecting topo sort with ties broken by the FIRST walk-order
//      position of each pipeline's ops. This produces a deterministic base
//      order that respects SSA and matches the lowering's intended sequence.
//   3) Member-conflict edges (RW/WR/WW) added in the direction of the base
//      order — never against it. This avoids the cycle that walk-order-per-op
//      direction causes when post-`InlineNestedMapPass` cloning interleaves
//      pipelines (e.g. two `subop.materialize` ops to the same buffer from
//      different pipelines: walk-order direction would pick one direction
//      for the WW conflict and the opposite for the genuine RW conflict —
//      cycle. Base-order direction is consistent.)
//
// Phases per ContainsNestedSubOps body:
//   A: assign each op to its single pipeline root via stream-chain.
//   B: collect required and produced state per pipeline.
//   C: build inter-pipeline SSA dependencies.
//   D: priority-queue Kahn topo sort (priority = first walk-position).
//   E: add member-conflict edges in topo direction.
//   F: create NestedExecutionGroupOp + ExecutionStepOps, move ops into them,
//      rewire body terminator through state mapping.

#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/StepGraphUtils.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace {
using namespace lingodb::compiler::dialect;

class SplitIntoNestedExecutionStepsPass : public mlir::PassWrapper<SplitIntoNestedExecutionStepsPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitIntoNestedExecutionStepsPass)
   virtual llvm::StringRef getArgument() const override { return "subop-split-into-nested-steps"; }

   enum Kind { READ,
               WRITE };

   struct Analysis {
      std::vector<mlir::Operation*> roots;
      llvm::DenseMap<mlir::Operation*, std::vector<mlir::Operation*>> pipelines;
      llvm::DenseMap<mlir::Operation*, mlir::Operation*> opToRoot;
      llvm::DenseMap<mlir::Operation*, std::vector<mlir::Value>> requiredState;
      llvm::DenseMap<mlir::Operation*, std::vector<mlir::Value>> producedState;
      llvm::DenseMap<mlir::Operation*, llvm::DenseSet<mlir::Operation*>> dependencies;
      std::vector<mlir::Operation*> topoOrder;
   };

   static bool isStream(mlir::Value v) {
      return mlir::isa<tuples::TupleStreamType>(v.getType());
   }

   // Pipeline assignment by stream-chain. No unions present so each op has
   // at most one stream producer.
   void buildPipelines(mlir::Block* body, Analysis& a) {
      auto* terminator = body->getTerminator();
      for (mlir::Operation& op : *body) {
         if (&op == terminator) continue;
         mlir::Operation* prev = nullptr;
         for (auto operand : op.getOperands()) {
            if (!isStream(operand)) continue;
            if (auto* p = operand.getDefiningOp()) {
               prev = p;
               break;
            }
         }
         if (prev && a.opToRoot.count(prev)) {
            auto* root = a.opToRoot[prev];
            a.opToRoot[&op] = root;
            a.pipelines[root].push_back(&op);
         } else {
            a.opToRoot[&op] = &op;
            a.pipelines[&op].push_back(&op);
            a.roots.push_back(&op);
         }
      }
   }

   // Is `block` `body` itself or a block nested (at any depth) in an op of
   // `body`? Works for detached bodies (islands, see below) too.
   static bool isInsideBody(mlir::Block* block, mlir::Block* body) {
      while (block) {
         if (block == body) return true;
         auto* parentOp = block->getParentOp();
         block = parentOp ? parentOp->getBlock() : nullptr;
      }
      return false;
   }

   // Required state of pipeline P: non-stream values used by ops in P that are
   // defined outside P. Sources:
   //   - block args of `body`
   //   - values defined elsewhere in `body` but in a different pipeline
   //   - values defined strictly outside `body`
   void computeStates(mlir::Block* body, Analysis& a) {
      llvm::DenseMap<mlir::Operation*, llvm::DenseSet<mlir::Operation*>> pipelineOpSet;
      for (auto& [root, ops] : a.pipelines) {
         for (auto* op : ops) pipelineOpSet[root].insert(op);
      }

      // SetVector preserves insertion order so that downstream IR
      // construction (step inputs / results) is deterministic.
      for (auto& [root, ops] : a.pipelines) {
         llvm::SetVector<mlir::Value> requiredSet;
         llvm::SetVector<mlir::Value> producedSet;
         for (auto* op : ops) {
            for (auto result : op->getResults()) {
               if (!isStream(result)) producedSet.insert(result);
            }
            op->walk([&](mlir::Operation* nestedOp) {
               for (auto operand : nestedOp->getOperands()) {
                  if (isStream(operand)) continue;
                  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
                     mlir::Block* ownerBlock = blockArg.getOwner();
                     // Per-tuple arg of the body, or an arg of a block above
                     // it: required. Args of a block strictly nested inside an
                     // op of the body are internal (the enclosing op gets
                     // moved into its step as a unit).
                     if (ownerBlock == body || !isInsideBody(ownerBlock, body)) {
                        requiredSet.insert(operand);
                     }
                     continue;
                  }
                  auto* prod = operand.getDefiningOp();
                  if (!prod) continue;
                  if (prod->getBlock() == body) {
                     // Defined in body but possibly different pipeline.
                     if (!pipelineOpSet[root].contains(prod)) {
                        requiredSet.insert(operand);
                     }
                     continue;
                  }
                  // Defined inside a nested region of some op in the body
                  // (e.g., the constant-init region of a simple_state):
                  // internal. Otherwise: required input from above.
                  if (isInsideBody(prod->getBlock(), body)) continue;
                  requiredSet.insert(operand);
               }
            });
         }
         a.requiredState[root].assign(requiredSet.begin(), requiredSet.end());
         a.producedState[root].assign(producedSet.begin(), producedSet.end());
      }
   }

   // SSA inter-pipeline edges only. Member-conflict edges are added later
   // (after the base topo sort) in `addMemberConflictDeps`.
   void buildSSADeps(Analysis& a) {
      llvm::DenseMap<mlir::Value, mlir::Operation*> producer;
      for (auto& [root, vals] : a.producedState) {
         for (auto v : vals) producer[v] = root;
      }
      for (auto& [root, vals] : a.requiredState) {
         for (auto v : vals) {
            auto it = producer.find(v);
            if (it != producer.end() && it->second != root) {
               a.dependencies[root].insert(it->second);
            }
         }
      }
   }

   // First walk-order position of any op belonging to each pipeline.
   llvm::DenseMap<mlir::Operation*, size_t>
   computeFirstPos(mlir::Block* body, Analysis& a) {
      llvm::DenseMap<mlir::Operation*, size_t> firstPos;
      auto* terminator = body->getTerminator();
      size_t pos = 0;
      for (mlir::Operation& op : *body) {
         if (&op == terminator) continue;
         pos++;
         auto* root = a.opToRoot.lookup(&op);
         if (root && !firstPos.count(root)) firstPos[root] = pos;
      }
      return firstPos;
   }

   // Add member-conflict edges in the order of `topoOrder` (later depends on
   // earlier). Since SSA edges are already in topo direction and member
   // edges are now too, the augmented graph is a forward DAG (no cycles).
   void addMemberConflictDeps(mlir::Block* body, Analysis& a) {
      llvm::DenseMap<mlir::Operation*, size_t> orderIdx;
      for (size_t i = 0; i < a.topoOrder.size(); ++i) orderIdx[a.topoOrder[i]] = i;

      llvm::DenseMap<subop::Member, std::vector<std::pair<mlir::Operation*, Kind>>> memberUsage;
      auto* terminator = body->getTerminator();
      for (mlir::Operation& op : *body) {
         if (&op == terminator) continue;
         auto* root = a.opToRoot.lookup(&op);
         if (!root) continue;
         op.walk([&](mlir::Operation* nestedOp) {
            auto subOp = mlir::dyn_cast_or_null<subop::SubOperator>(nestedOp);
            if (!subOp) return;
            for (auto m : subOp.getReadMembers()) memberUsage[m].push_back({root, READ});
            for (auto m : subOp.getWrittenMembers()) memberUsage[m].push_back({root, WRITE});
         });
      }
      for (auto& [member, entries] : memberUsage) {
         for (size_t i = 0; i < entries.size(); ++i) {
            for (size_t j = i + 1; j < entries.size(); ++j) {
               auto [pi, ki] = entries[i];
               auto [pj, kj] = entries[j];
               if (pi == pj) continue;
               bool conflict = (ki == WRITE && kj == WRITE) ||
                  (ki == WRITE && kj == READ) ||
                  (ki == READ && kj == WRITE);
               if (!conflict) continue;
               if (orderIdx[pi] < orderIdx[pj])
                  a.dependencies[pj].insert(pi);
               else
                  a.dependencies[pi].insert(pj);
            }
         }
      }
   }

   // Materialize the new IR: NestedExecutionGroupOp containing ExecutionStepOps
   // in topo order, inserted before `insertBefore`. Every use of an `escaping`
   // value outside the new group is rewired to the corresponding group result.
   void materialize(mlir::Location loc, mlir::Operation* insertBefore, llvm::ArrayRef<mlir::Value> escaping, Analysis& a) {
      llvm::DenseMap<mlir::Value, mlir::Value> stateMapping;
      llvm::DenseMap<mlir::Value, size_t> valueToNestedGroupArg;

      auto* nestedExecutionBlock = new mlir::Block;
      std::vector<mlir::Value> nestedExecutionOperands;

      mlir::OpBuilder builder(&getContext());
      builder.setInsertionPointToStart(nestedExecutionBlock);
      auto returnOp = builder.create<subop::NestedExecutionGroupReturnOp>(loc, mlir::ValueRange{});

      for (auto* root : a.topoOrder) {
         std::vector<mlir::Type> resultTypes;
         for (auto v : a.producedState[root]) resultTypes.push_back(v.getType());

         std::vector<mlir::Value> inputs;
         std::vector<mlir::Value> blockArgs;
         llvm::SmallVector<bool> threadLocal;
         auto* stepBlock = new mlir::Block;
         llvm::DenseMap<mlir::Value, size_t> availableStates;

         for (auto required : a.requiredState[root]) {
            if (availableStates.contains(required)) {
               blockArgs.push_back(stepBlock->getArgument(availableStates[required]));
               continue;
            }
            if (stateMapping.count(required)) {
               // Produced by an earlier-processed step in this body.
               inputs.push_back(stateMapping[required]);
            } else if (valueToNestedGroupArg.contains(required)) {
               // Already plumbed into the NestedExecutionGroupOp.
               inputs.push_back(nestedExecutionBlock->getArgument(valueToNestedGroupArg[required]));
            } else {
               // First step needing this externally-defined value: add a
               // NestedExecutionGroup operand + corresponding block arg.
               nestedExecutionOperands.push_back(required);
               auto nestedArg = nestedExecutionBlock->addArgument(required.getType(), required.getLoc());
               valueToNestedGroupArg[required] = nestedArg.getArgNumber();
               inputs.push_back(nestedArg);
            }
            blockArgs.push_back(stepBlock->addArgument(required.getType(), required.getLoc()));
            threadLocal.push_back(false);
            availableStates[required] = stepBlock->getNumArguments() - 1;
         }

         mlir::OpBuilder outerBuilder(&getContext());
         outerBuilder.setInsertionPoint(returnOp);
         auto stepOp = outerBuilder.create<subop::ExecutionStepOp>(
            root->getLoc(), resultTypes, inputs,
            outerBuilder.getBoolArrayAttr(threadLocal));
         stepOp.getSubOps().getBlocks().push_back(stepBlock);

         // Move ops into the step block, remapping required-state operands
         // to the corresponding block arg.
         mlir::OpBuilder stepBuilder(&getContext());
         stepBuilder.setInsertionPointToStart(stepBlock);
         for (auto* op : a.pipelines[root]) {
            op->remove();
            for (auto [origReq, blockArg] : llvm::zip(a.requiredState[root], blockArgs)) {
               origReq.replaceUsesWithIf(blockArg, [&](mlir::OpOperand& operand) {
                  return op->isAncestor(operand.getOwner());
               });
            }
            stepBuilder.insert(op);
         }
         stepBuilder.create<subop::ExecutionStepReturnOp>(root->getLoc(), a.producedState[root]);

         for (auto [orig, res] : llvm::zip(a.producedState[root], stepOp.getResults())) {
            stateMapping[orig] = res;
         }
      }

      // Create the NestedExecutionGroupOp and plumb out its results.
      builder.setInsertionPoint(insertBefore);
      std::vector<mlir::Value> toReturn;
      std::vector<mlir::Value> toMap;
      std::vector<mlir::Type> toReturnTypes;
      for (auto value : escaping) {
         if (stateMapping.count(value)) {
            toReturn.push_back(stateMapping[value]);
            toReturnTypes.push_back(value.getType());
            toMap.push_back(value);
         }
      }
      returnOp->setOperands(toReturn);
      auto nestedExecutionGroup = builder.create<subop::NestedExecutionGroupOp>(
         loc, toReturnTypes, nestedExecutionOperands);
      nestedExecutionGroup.getSubOps().getBlocks().clear();
      nestedExecutionGroup.getSubOps().push_back(nestedExecutionBlock);
      for (auto [from, to] : llvm::zip(toMap, nestedExecutionGroup.getResults())) {
         from.replaceUsesWithIf(to, [&](mlir::OpOperand& operand) {
            return !nestedExecutionGroup->isAncestor(operand.getOwner());
         });
      }
   }

   // Split `body` (terminated; the terminator itself is not moved) into
   // steps of a new NestedExecutionGroupOp placed before `insertBefore`.
   bool splitBody(mlir::Operation* errorOp, mlir::Block* body, mlir::Operation* insertBefore, llvm::ArrayRef<mlir::Value> escaping) {
      Analysis a;
      buildPipelines(body, a);
      computeStates(body, a);
      buildSSADeps(a);
      auto firstPos = computeFirstPos(body, a);
      a.topoOrder = subop::kahnTopoSort(a.roots, a.dependencies, firstPos);
      if (a.topoOrder.size() != a.roots.size()) {
         errorOp->emitError("SplitIntoNestedExecutionStepsPass: cycle in SSA dependencies of nested body");
         return false;
      }
      addMemberConflictDeps(body, a);
      // Member-conflict edges are added in topo direction, so the augmented
      // graph stays acyclic and the existing topoOrder is still valid.
      materialize(errorOp->getLoc(), insertBefore, escaping, a);
      return true;
   }

   void splitContainsNestedSubOps(subop::ContainsNestedSubOps cn) {
      mlir::Block* body = cn.getBody();
      if (!body) return;
      auto* terminator = body->getTerminator();
      llvm::SmallVector<mlir::Value> escaping(terminator->getOperands().begin(), terminator->getOperands().end());
      if (!splitBody(cn.getOperation(), body, terminator, escaping)) return signalPassFailure();
   }

   // A block of imperative code nested inside an execution step (e.g. the
   // body of an scf.for in a subop.map lambda — nested SQL issued per loop
   // iteration by a hipy UDF — or a subop.map lambda itself) that directly
   // contains subop ops. Those ops form an "island": a nested query that is
   // executed in place, each time control reaches it.
   static bool isIslandBlock(mlir::Block& block) {
      auto* parentOp = block.getParentOp();
      if (!parentOp) return false;
      // subop ops directly in a map lambda: a nested query evaluated in place
      // per tuple (e.g. relalg.getfirstrow outside of a loop)
      if (mlir::isa<subop::SubOperatorDialect>(parentOp->getDialect()) && !mlir::isa<subop::MapOp>(parentOp)) return false;
      if (mlir::isa<mlir::func::FuncOp, mlir::ModuleOp>(parentOp)) return false;
      if (!parentOp->getParentOfType<subop::ExecutionStepOp>()) return false;
      // Only real query work forms an island (e.g. not the generate_emit ops
      // of a subop.generate region's imperative body).
      for (auto& op : block) {
         if (mlir::isa<subop::SubOperator>(&op)) return true;
      }
      return false;
   }
   static bool isIslandOp(mlir::Operation* op) {
      return mlir::isa<subop::SubOperatorDialect>(op->getDialect()) && !mlir::isa<subop::NestedExecutionGroupOp>(op);
   }

   // Does `op` (or an op nested in it) use a result of an op of `island` (ops of `block`)?
   static bool usesIsland(mlir::Operation* op, mlir::Block& block, const llvm::SetVector<mlir::Operation*>& island) {
      bool res = false;
      op->walk([&](mlir::Operation* nested) {
         for (auto operand : nested->getOperands()) {
            if (auto* def = operand.getDefiningOp(); def && def->getBlock() == &block && island.contains(def)) res = true;
         }
      });
      return res;
   }

   // Partition the subop ops of `block` into islands, one per nested query
   // (in block order): an island ends at the first imperative op that consumes
   // one of its results (e.g. the util.unpack after a state_to_native), so the
   // imperative glue code between two nested queries (e.g. the second one
   // using the first one's result as a parameter) runs between them instead of
   // being moved into one group. Islands that share subop values (e.g. a
   // common scan) stay together.
   static std::vector<llvm::SetVector<mlir::Operation*>> partitionIslands(mlir::Block& block) {
      std::vector<llvm::SetVector<mlir::Operation*>> islands;
      llvm::SetVector<mlir::Operation*> current;
      for (auto& op : block) {
         if (isIslandOp(&op)) {
            if (!islands.empty() && current.empty() && usesIsland(&op, block, islands.back())) {
               // uses a subop value of the previous island: continue that one
               current = islands.back();
               islands.pop_back();
            }
            current.insert(&op);
         } else if (!current.empty() && usesIsland(&op, block, current)) {
            islands.push_back(current);
            current.clear();
         }
      }
      if (!current.empty()) islands.push_back(current);
      return islands;
   }

   // Wrap `island` (ops of `block`) into a NestedExecutionGroupOp at the
   // position of its last subop op. Imperative ops in between that consume
   // island results (transitively) join the island, so that every value the
   // group needs is defined before it and every result is only used after it.
   bool splitIsland(mlir::Block& block, llvm::SetVector<mlir::Operation*> island) {
      mlir::Operation* last = island.back();
      for (auto& op : block) {
         if (&op == last) break;
         if (!island.contains(&op) && usesIsland(&op, block, island)) island.insert(&op);
      }
      llvm::SmallVector<mlir::Operation*> ordered;
      for (auto& op : block) {
         if (island.contains(&op)) ordered.push_back(&op);
      }
      llvm::SetVector<mlir::Value> escaping;
      for (auto* op : ordered) {
         for (auto result : op->getResults()) {
            for (auto* user : result.getUsers()) {
               if (!island.contains(block.findAncestorOpInBlock(*user))) escaping.insert(result);
            }
         }
      }
      auto* insertBefore = last->getNextNode();
      auto loc = ordered.front()->getLoc();
      auto* tmpBody = new mlir::Block;
      for (auto* op : ordered) {
         op->remove();
         tmpBody->push_back(op);
      }
      mlir::OpBuilder builder(&getContext());
      builder.setInsertionPointToEnd(tmpBody);
      builder.create<subop::NestedExecutionGroupReturnOp>(loc, mlir::ValueRange{});
      bool success = splitBody(insertBefore, tmpBody, insertBefore, escaping.getArrayRef());
      delete tmpBody;
      return success;
   }

   void runOnOperation() override {
      // Islands first, innermost first (post-order), so that an island nested
      // in an op of an enclosing island/body is moved along as a unit.
      std::vector<mlir::Block*> islands;
      getOperation()->walk([&](mlir::Block* block) {
         if (isIslandBlock(*block)) islands.push_back(block);
      });
      for (auto* block : islands) {
         for (auto& island : partitionIslands(*block)) {
            if (!splitIsland(*block, island)) return signalPassFailure();
         }
      }
      // Collect first; mutating during walk would invalidate the iteration.
      std::vector<subop::ContainsNestedSubOps> targets;
      getOperation()->walk<mlir::WalkOrder::PreOrder>([&](subop::ContainsNestedSubOps cn) {
         targets.push_back(cn);
      });
      for (auto cn : targets) {
         splitContainsNestedSubOps(cn);
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
subop::createSplitIntoNestedExecutionStepsPass() { return std::make_unique<SplitIntoNestedExecutionStepsPass>(); }
