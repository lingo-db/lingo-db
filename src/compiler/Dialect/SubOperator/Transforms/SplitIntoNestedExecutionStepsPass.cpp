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
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

#include <queue>

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

   // Required state of pipeline P: non-stream values used by ops in P that are
   // defined outside P. Sources:
   //   - block args of `body`
   //   - values defined elsewhere in `body` but in a different pipeline
   //   - values defined strictly outside the ContainsNestedSubOps op
   void computeStates(subop::ContainsNestedSubOps cn, mlir::Block* body, Analysis& a) {
      llvm::DenseMap<mlir::Operation*, llvm::DenseSet<mlir::Operation*>> pipelineOpSet;
      for (auto& [root, ops] : a.pipelines) {
         for (auto* op : ops) pipelineOpSet[root].insert(op);
      }

      for (auto& [root, ops] : a.pipelines) {
         llvm::DenseSet<mlir::Value> requiredSet;
         llvm::DenseSet<mlir::Value> producedSet;
         for (auto* op : ops) {
            for (auto result : op->getResults()) {
               if (!isStream(result)) producedSet.insert(result);
            }
            op->walk([&](mlir::Operation* nestedOp) {
               for (auto operand : nestedOp->getOperands()) {
                  if (isStream(operand)) continue;
                  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
                     mlir::Block* ownerBlock = blockArg.getOwner();
                     if (ownerBlock == body) {
                        // Per-tuple arg of cn — required.
                        requiredSet.insert(operand);
                        continue;
                     }
                     // Args of a block strictly nested inside an op of the
                     // body are internal (the enclosing op gets moved into
                     // its step as a unit). Args of any block at or above
                     // cn's level are external.
                     bool insideCn = false;
                     for (mlir::Operation* p = ownerBlock->getParentOp(); p; p = p->getParentOp()) {
                        if (p == cn.getOperation()) {
                           insideCn = true;
                           break;
                        }
                     }
                     if (!insideCn) {
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
                  // Defined outside body. If it's still inside `cn`, the
                  // producer must be inside a nested region of some op in
                  // the body (e.g., the constant-init region of a
                  // simple_state). Internal — skip.
                  mlir::Operation* p = prod;
                  while (p && p != cn.getOperation()) p = p->getParentOp();
                  if (p == cn.getOperation()) continue;
                  // Strictly outside cn: required input from above.
                  requiredSet.insert(operand);
               }
            });
         }
         for (auto v : requiredSet) a.requiredState[root].push_back(v);
         for (auto v : producedSet) a.producedState[root].push_back(v);
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

   // Kahn's topo sort with ties broken by `priority` (smaller value = earlier
   // in the result). Returns the ordered list; if a cycle prevents full
   // ordering, the returned vector has size < roots.size().
   std::vector<mlir::Operation*>
   kahn(const std::vector<mlir::Operation*>& roots,
        const llvm::DenseMap<mlir::Operation*, llvm::DenseSet<mlir::Operation*>>& deps,
        const llvm::DenseMap<mlir::Operation*, size_t>& priority) {
      llvm::DenseMap<mlir::Operation*, size_t> indegree;
      llvm::DenseMap<mlir::Operation*, llvm::DenseSet<mlir::Operation*>> reverse;
      for (auto* r : roots) indegree[r] = 0;
      for (auto& [r, ds] : deps) {
         for (auto* d : ds) {
            reverse[d].insert(r);
            indegree[r]++;
         }
      }
      auto cmp = [&](mlir::Operation* a, mlir::Operation* b) {
         size_t pa = priority.count(a) ? priority.lookup(a) : SIZE_MAX;
         size_t pb = priority.count(b) ? priority.lookup(b) : SIZE_MAX;
         return pa > pb; // min-heap
      };
      std::priority_queue<mlir::Operation*, std::vector<mlir::Operation*>, decltype(cmp)> pq(cmp);
      for (auto& [r, n] : indegree) {
         if (n == 0) pq.push(r);
      }
      std::vector<mlir::Operation*> out;
      while (!pq.empty()) {
         auto* r = pq.top();
         pq.pop();
         out.push_back(r);
         for (auto* s : reverse[r]) {
            if (--indegree[s] == 0) pq.push(s);
         }
      }
      return out;
   }

   // Materialize the new IR: NestedExecutionGroupOp containing ExecutionStepOps
   // in topo order, body terminator rewired through state mapping.
   void materialize(subop::ContainsNestedSubOps cn, mlir::Block* body, Analysis& a) {
      llvm::DenseMap<mlir::Value, mlir::Value> stateMapping;
      llvm::DenseMap<mlir::Value, size_t> valueToNestedGroupArg;

      auto* nestedExecutionBlock = new mlir::Block;
      std::vector<mlir::Value> nestedExecutionOperands;

      mlir::OpBuilder builder(&getContext());
      builder.setInsertionPointToStart(nestedExecutionBlock);
      auto returnOp = builder.create<subop::NestedExecutionGroupReturnOp>(cn.getLoc(), mlir::ValueRange{});

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

      // Rewire body terminator through stateMapping; create the
      // NestedExecutionGroupOp and plumb out its results.
      auto* terminator = body->getTerminator();
      builder.setInsertionPoint(terminator);
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
      auto nestedExecutionGroup = builder.create<subop::NestedExecutionGroupOp>(
         cn.getLoc(), toReturnTypes, nestedExecutionOperands);
      nestedExecutionGroup.getSubOps().getBlocks().clear();
      nestedExecutionGroup.getSubOps().push_back(nestedExecutionBlock);
      for (auto [from, to] : llvm::zip(toMap, nestedExecutionGroup.getResults())) {
         from.replaceUsesWithIf(to, [&](mlir::OpOperand& operand) {
            return terminator == operand.getOwner();
         });
      }
   }

   void splitContainsNestedSubOps(subop::ContainsNestedSubOps cn) {
      mlir::Block* body = cn.getBody();
      if (!body) return;
      Analysis a;
      buildPipelines(body, a);
      computeStates(cn, body, a);
      buildSSADeps(a);
      auto firstPos = computeFirstPos(body, a);
      a.topoOrder = kahn(a.roots, a.dependencies, firstPos);
      if (a.topoOrder.size() != a.roots.size()) {
         cn.emitError("SplitIntoNestedExecutionStepsPass: cycle in SSA dependencies of nested body");
         return signalPassFailure();
      }
      addMemberConflictDeps(body, a);
      // Member-conflict edges are added in topo direction, so the augmented
      // graph stays acyclic and the existing topoOrder is still valid.
      materialize(cn, body, a);
   }

   void runOnOperation() override {
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
