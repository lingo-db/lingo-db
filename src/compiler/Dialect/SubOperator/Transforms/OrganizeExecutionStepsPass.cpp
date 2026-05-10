// OrganizeExecutionStepsPass — splits an ExecutionGroupOp body into per-
// pipeline ExecutionStepOps. (Replaces the historical EnforceOrder +
// Finalize + SplitIntoExecutionSteps trio.)
//
// Approach:
//   Phase 0: reify multi-emit subop.generate into a temp buffer + per-stream
//            materialize + scan_refs + gather. (Single mutation up front, only
//            because multi-emit generates have non-cloneable semantics — the
//            imperative body must run once.)
//
//   Per ExecutionGroupOp:
//   Phase A: analyze the original IR to compute pipeline membership (with
//            multi-root semantics for ops downstream of unions), required /
//            produced states per pipeline, inter-pipeline dependencies, and a
//            topological order.
//
//   Phase B: create empty ExecutionStepOps in topological order, plumbing
//            their state inputs/outputs based on Phase A's analysis.
//
//   Phase C: populate each step. For an op O that belongs to N pipelines:
//            in topo order, the LAST pipeline gets the moved original; all
//            earlier pipelines get clones via cloneSubOp. Tuple-stream
//            operands that go through a subop.union are rewired to skip the
//            union and refer directly to the operand whose root corresponds
//            to the current pipeline.
//
//   Phase D: erase now-unused unions; rewire the execution_group_return.

#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <queue>
#include <unordered_set>

namespace {
using namespace lingodb::compiler::dialect;

// True if `op` is "imperative" — i.e., not a SubOperator-implementing op. Such
// ops (arith, scf, util, memref, …) are grouped into pipelines via SSA closure
// rather than tuple-stream chains.
static bool isImperative(mlir::Operation* op) {
   return !mlir::isa<subop::SubOperator>(op);
}

// Tiny union-find over `mlir::Operation*` for grouping imperative ops via
// shared SSA values. Nodes are added on first reference.
struct OpUnionFind {
   llvm::DenseMap<mlir::Operation*, mlir::Operation*> parent;
   mlir::Operation* find(mlir::Operation* x) {
      auto it = parent.find(x);
      if (it == parent.end()) {
         parent[x] = x;
         return x;
      }
      if (it->second == x) return x;
      auto* root = find(it->second);
      parent[x] = root;
      return root;
   }
   void unite(mlir::Operation* a, mlir::Operation* b) {
      a = find(a);
      b = find(b);
      if (a != b) parent[a] = b;
   }
};

class OrganizeExecutionStepsPass : public mlir::PassWrapper<OrganizeExecutionStepsPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OrganizeExecutionStepsPass)
   virtual llvm::StringRef getArgument() const override { return "subop-organize-execution-steps"; }

   //===----------------------------------------------------------------===//
   // Phase 0 — reify multi-emit generate; flatten single-emit
   //===----------------------------------------------------------------===//
   void reifyGenerates() {
      std::vector<subop::GenerateOp> generateOps;
      getOperation()->walk([&](subop::GenerateOp generateOp) {
         generateOps.push_back(generateOp);
      });
      auto& memberManager = getContext().getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
      auto& colManager = getContext().getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      for (auto generateOp : generateOps) {
         size_t emitOps = 0;
         generateOp.getRegion().walk([&](subop::GenerateEmitOp) { emitOps++; });
         if (emitOps == 0) {
            generateOp.emitError("GenerateOp must contain at least one GenerateEmitOp");
            return signalPassFailure();
         }
         if (emitOps == 1) {
            generateOp.getRes().replaceAllUsesWith(generateOp.getStreams()[0]);
            continue;
         }
         // Multi-emit: reify via temp buffer.
         mlir::OpBuilder builder(generateOp);
         auto loc = generateOp.getLoc();
         llvm::SmallVector<subop::Member> members;
         llvm::SmallVector<subop::DefMappingPairT> defMapping;
         llvm::SmallVector<subop::RefMappingPairT> refMapping;
         for (auto m : generateOp.getGeneratedColumns()) {
            auto* column = &mlir::cast<tuples::ColumnDefAttr>(m).getColumn();
            auto member = memberManager.createMember("tmp_union", column->type);
            members.push_back(member);
            auto colDef = mlir::cast<tuples::ColumnDefAttr>(m);
            defMapping.push_back({member, colDef});
            refMapping.push_back({member, colManager.createRef(&colDef.getColumn())});
         }
         auto bufferType = subop::BufferType::get(builder.getContext(), subop::StateMembersAttr::get(builder.getContext(), members));
         mlir::Value tmpBuffer = builder.create<subop::GenericCreateOp>(loc, bufferType);
         builder.setInsertionPointAfter(generateOp);
         for (auto stream : generateOp.getStreams()) {
            builder.create<subop::MaterializeOp>(loc, stream, tmpBuffer, subop::ColumnRefMemberMappingAttr::get(builder.getContext(), refMapping));
         }
         auto scanRefDef = colManager.createDef(colManager.getUniqueScope("tmp_union"), "scan_ref");
         scanRefDef.getColumn().type = subop::EntryRefType::get(builder.getContext(), mlir::cast<subop::State>(tmpBuffer.getType()));
         auto scan = builder.create<subop::ScanRefsOp>(loc, tmpBuffer, scanRefDef);
         mlir::Value loaded = builder.create<subop::GatherOp>(loc, scan, colManager.createRef(&scanRefDef.getColumn()),
                                                              subop::ColumnDefMemberMappingAttr::get(builder.getContext(), defMapping));
         generateOp.getRes().replaceAllUsesWith(loaded);
      }
   }

   //===----------------------------------------------------------------===//
   // Phase A — analyze (per ExecutionGroupOp body, no mutations)
   //===----------------------------------------------------------------===//

   enum Kind { READ,
               WRITE };

   struct Analysis {
      // Pipeline root → ordered list of ops in that pipeline (in original IR order).
      llvm::DenseMap<mlir::Operation*, std::vector<mlir::Operation*>> pipelines;
      // Op → set of roots (multi-rooted iff downstream of a union).
      llvm::DenseMap<mlir::Operation*, llvm::SmallVector<mlir::Operation*, 2>> opToRoots;
      // Pipeline → state inputs it requires (values defined outside the pipeline's ops).
      llvm::DenseMap<mlir::Operation*, std::vector<mlir::Value>> requiredState;
      // Pipeline → state values it produces (non-tuple-stream results referenced outside).
      llvm::DenseMap<mlir::Operation*, std::vector<mlir::Value>> producedState;
      // Pipeline dependency graph.
      llvm::DenseMap<mlir::Operation*, llvm::DenseSet<mlir::Operation*>> dependencies;
      // Topologically ordered list of pipeline roots.
      std::vector<mlir::Operation*> topoOrder;
      // For each op with multiple roots: which root is "last" in topo order.
      // The op gets MOVED into that pipeline's step; all earlier pipelines get clones.
      llvm::DenseMap<mlir::Operation*, mlir::Operation*> lastPipelineFor;
      // For each tuple-stream value that's a subop.union result: per pipeline-root,
      // the operand to use instead of the union.
      llvm::DenseMap<std::pair<mlir::Value, mlir::Operation*>, mlir::Value> unionRewire;
   };

   // True if `v` is a tuple-stream value.
   static bool isStream(mlir::Value v) {
      return mlir::isa<tuples::TupleStreamType>(v.getType());
   }

   // Compute roots[op] for every op in the execution group body.
   //
   // SubOperator ops:
   //   - With tuple-stream operand(s): inherit roots from each producer's set
   //     (deduplicated). UnionOps accumulate roots from all their operands.
   //   - With no tuple-stream operand: their own (singleton) root.
   //
   // Imperative ops (arith / scf / util / …):
   //   - Grouped via SSA closure: any two imperative ops sharing an SSA value
   //     end up with the same root. Implemented as a separate union-find pass
   //     after the stream-chain assignment.
   void computeRoots(subop::ExecutionGroupOp eg, Analysis& a) {
      // Pass 1: stream-chain-based root assignment.
      for (mlir::Operation& op : eg.getSubOps().front()) {
         if (mlir::isa<subop::ExecutionGroupReturnOp>(op)) continue;
         llvm::SmallVector<mlir::Operation*, 2> roots;
         llvm::DenseSet<mlir::Operation*> seen;
         for (auto operand : op.getOperands()) {
            if (!isStream(operand)) continue;
            if (auto* prod = operand.getDefiningOp()) {
               for (auto* r : a.opToRoots[prod]) {
                  if (seen.insert(r).second) roots.push_back(r);
               }
            }
         }
         if (roots.empty()) {
            roots.push_back(&op);
         }
         a.opToRoots[&op] = roots;
      }

      // Pass 2: union-find over imperative ops via SSA edges. An imperative
      // op merges with any imperative producer of any of its operands, so
      // diamond and side-effect-via-shared-object patterns end up in one
      // pipeline.
      OpUnionFind uf;
      for (mlir::Operation& op : eg.getSubOps().front()) {
         if (mlir::isa<subop::ExecutionGroupReturnOp>(op)) continue;
         if (!isImperative(&op)) continue;
         for (auto operand : op.getOperands()) {
            auto* prod = operand.getDefiningOp();
            if (!prod) continue;
            if (!isImperative(prod)) continue;
            uf.unite(&op, prod);
         }
      }

      // Pass 3: rewrite imperative roots to UF representatives.
      for (auto& [op, roots] : a.opToRoots) {
         if (!isImperative(op)) continue;
         assert(roots.size() == 1);
         roots[0] = uf.find(roots[0]);
      }
   }

   // Group ops into pipelines (op may be in multiple).
   // Skip UnionOps from pipeline membership — they're virtual fan-in nodes,
   // erased after Phase C.
   void buildPipelines(subop::ExecutionGroupOp eg, Analysis& a) {
      for (mlir::Operation& op : eg.getSubOps().front()) {
         if (mlir::isa<subop::ExecutionGroupReturnOp>(op)) continue;
         if (mlir::isa<subop::UnionOp>(op)) continue;
         for (auto* r : a.opToRoots[&op]) {
            a.pipelines[r].push_back(&op);
         }
      }
   }

   // For a tuple-stream value `v`, find the operand (within `v`'s defining
   // chain through unions) whose roots include `pipelineRoot`, then chain
   // through `mapping` to its current in-pipeline substitution. Handles
   // nested unions (union(map(union(...)), ...)) by recursing through union
   // operands. The mapping chain handles values that have already been
   // moved/cloned during this pipeline's processing.
   mlir::Value resolveStreamForPipeline(mlir::Value v, mlir::Operation* pipelineRoot,
                                        mlir::IRMapping& mapping, Analysis& a) {
      if (auto unionOp = v.getDefiningOp<subop::UnionOp>()) {
         for (auto operand : unionOp.getOperands()) {
            auto* prod = operand.getDefiningOp();
            if (!prod) continue;
            const auto& roots = a.opToRoots[prod];
            if (std::find(roots.begin(), roots.end(), pipelineRoot) != roots.end()) {
               return resolveStreamForPipeline(operand, pipelineRoot, mapping, a);
            }
         }
         return v;
      }
      while (auto m = mapping.lookupOrNull(v)) {
         if (m == v) break;
         v = m;
      }
      return v;
   }

   // For each pipeline, collect required-state inputs (non-tuple-stream
   // operands whose definers aren't in this pipeline) and produced-state
   // outputs. Walks into nested regions (nested_map / loop bodies are opaque
   // at this level but their member access still counts).
   void computeStates(subop::ExecutionGroupOp eg, Analysis& a) {
      // For required-state: a value is "external" to a pipeline if its
      // defining op is NOT in `pipelineOpSet[root]`. Build that lookup.
      llvm::DenseMap<mlir::Operation*, llvm::DenseSet<mlir::Operation*>> pipelineOpSet;
      for (auto& [root, ops] : a.pipelines) {
         for (auto* op : ops) pipelineOpSet[root].insert(op);
      }

      for (auto& [root, ops] : a.pipelines) {
         llvm::DenseSet<mlir::Value> requiredSet;
         llvm::DenseSet<mlir::Value> producedSet;
         for (auto* op : ops) {
            // Produced: non-tuple-stream results that are USED outside the
            // pipeline. (Imperative pipelines can have many internal results;
            // skipping unused ones keeps the step's result list small.)
            for (auto result : op->getResults()) {
               if (isStream(result)) continue;
               for (auto* user : result.getUsers()) {
                  if (!pipelineOpSet[root].contains(user)) {
                     producedSet.insert(result);
                     break;
                  }
               }
            }
            // Required: walk operands (incl. nested regions).
            op->walk([&](mlir::Operation* nestedOp) {
               for (auto operand : nestedOp->getOperands()) {
                  if (isStream(operand)) continue;
                  auto* prod = operand.getDefiningOp();
                  if (!prod) continue;
                  if (prod->getBlock() != op->getBlock()) continue;
                  if (pipelineOpSet[root].contains(prod)) continue;
                  requiredSet.insert(operand);
               }
            });
         }
         for (auto v : requiredSet) a.requiredState[root].push_back(v);
         for (auto v : producedSet) a.producedState[root].push_back(v);
      }
   }

   // Build inter-pipeline dependencies:
   //  - SSA: pipeline R needs a state produced by pipeline R'  →  R depends on R'.
   //  - Member conflicts (RW/WW/WR): direction by walk-order on original IR.
   void buildDependencies(subop::ExecutionGroupOp eg, Analysis& a) {
      // Map state value → pipeline that produces it.
      llvm::DenseMap<mlir::Value, mlir::Operation*> producer;
      for (auto& [root, vals] : a.producedState) {
         for (auto v : vals) producer[v] = root;
      }

      // SSA edges
      for (auto& [root, vals] : a.requiredState) {
         for (auto v : vals) {
            auto it = producer.find(v);
            if (it != producer.end() && it->second != root) {
               a.dependencies[root].insert(it->second);
            }
         }
      }

      // Member conflicts. memberUsage[m] = ordered list (by walk-order) of
      // (pipeline, op, kind) tuples touching member m.
      llvm::DenseMap<subop::Member, std::vector<std::tuple<mlir::Operation*, mlir::Operation*, Kind>>> memberUsage;
      // Walk ops in IR order; for each pipeline root that contains the op,
      // record its member access.
      for (mlir::Operation& op : eg.getSubOps().front()) {
         if (mlir::isa<subop::ExecutionGroupReturnOp>(op)) continue;
         if (mlir::isa<subop::UnionOp>(op)) continue;
         const auto& roots = a.opToRoots[&op];
         op.walk([&](mlir::Operation* nestedOp) {
            auto subOp = mlir::dyn_cast_or_null<subop::SubOperator>(nestedOp);
            if (!subOp) return;
            auto reads = subOp.getReadMembers();
            auto writes = subOp.getWrittenMembers();
            for (auto* r : roots) {
               for (auto m : reads) memberUsage[m].push_back({r, &op, READ});
               for (auto m : writes) memberUsage[m].push_back({r, &op, WRITE});
            }
         });
      }
      for (auto& [member, entries] : memberUsage) {
         for (size_t i = 0; i < entries.size(); ++i) {
            for (size_t j = i + 1; j < entries.size(); ++j) {
               auto [pi, opi, ki] = entries[i];
               auto [pj, opj, kj] = entries[j];
               if (pi == pj) continue;
               // Skip conflicts that come from the *same* multi-rooted op
               // appearing in both pipelines. Both entries refer to one
               // physical op that will be moved/cloned into the right
               // pipelines during Phase C; no inter-pipeline ordering
               // edge is needed (and adding one creates spurious cycles
               // when other genuine conflicts already constrain the order).
               if (opi == opj) continue;
               // Skip if a multi-rooted op is "phantom-present" in the
               // other pipeline: e.g. if opi is in both pi and pj (so
               // physically lives in pj's flow too), the conflict between
               // opi and opj is *also* recorded as the intra-pipeline pair
               // (pj, opi)–(pj, opj) and is correctly skipped there. Adding
               // an edge from this duplicate cross-pair creates spurious
               // cycles.
               const auto& rootsI = a.opToRoots[opi];
               const auto& rootsJ = a.opToRoots[opj];
               if (std::find(rootsI.begin(), rootsI.end(), pj) != rootsI.end()) continue;
               if (std::find(rootsJ.begin(), rootsJ.end(), pi) != rootsJ.end()) continue;
               // i is before j in walk order; conflict ⇒ pj depends on pi.
               bool conflict = (ki == WRITE && kj == WRITE) ||
                  (ki == WRITE && kj == READ) ||
                  (ki == READ && kj == WRITE);
               if (!conflict) continue;
               a.dependencies[pj].insert(pi);
            }
         }
      }
   }

   // Topological sort via Kahn's. On cycle: emit diagnostic, fail.
   bool topoSort(Analysis& a, mlir::Operation* errLoc) {
      llvm::DenseMap<mlir::Operation*, size_t> inDegree;
      llvm::DenseMap<mlir::Operation*, llvm::DenseSet<mlir::Operation*>> reverseDeps;
      for (auto& [root, _] : a.pipelines) inDegree[root] = 0;
      for (auto& [root, deps] : a.dependencies) {
         for (auto* d : deps) {
            reverseDeps[d].insert(root);
            inDegree[root]++;
         }
      }
      std::queue<mlir::Operation*> queue;
      for (auto& [root, n] : inDegree) {
         if (n == 0) queue.push(root);
      }
      while (!queue.empty()) {
         auto* root = queue.front();
         queue.pop();
         a.topoOrder.push_back(root);
         for (auto* succ : reverseDeps[root]) {
            if (--inDegree[succ] == 0) queue.push(succ);
         }
      }
      if (a.topoOrder.size() != a.pipelines.size()) {
         errLoc->emitError("OrganizeExecutionStepsPass: cycle in pipeline dependency graph; "
                           "the inter-pipeline state-conflict graph is not topologically sortable. "
                           "This usually indicates conflicting state operations whose direction "
                           "cannot be resolved by walk order.");
         return false;
      }
      return true;
   }

   // For each multi-rooted op, decide which pipeline gets the moved original
   // (the topologically-LAST pipeline it belongs to).
   void computeLastPipelines(Analysis& a) {
      llvm::DenseMap<mlir::Operation*, size_t> topoIndex;
      for (size_t i = 0; i < a.topoOrder.size(); ++i) {
         topoIndex[a.topoOrder[i]] = i;
      }
      for (auto& [op, roots] : a.opToRoots) {
         if (roots.size() == 1) {
            a.lastPipelineFor[op] = roots[0];
            continue;
         }
         mlir::Operation* best = roots[0];
         size_t bestIdx = topoIndex[best];
         for (size_t i = 1; i < roots.size(); ++i) {
            size_t idx = topoIndex[roots[i]];
            if (idx > bestIdx) {
               best = roots[i];
               bestIdx = idx;
            }
         }
         a.lastPipelineFor[op] = best;
      }
   }

   //===----------------------------------------------------------------===//
   // Phase B — create empty ExecutionStepOps in topo order
   //===----------------------------------------------------------------===//

   struct StepCtx {
      subop::ExecutionStepOp step;
      // For each required-state (original Value), the corresponding block arg.
      llvm::DenseMap<mlir::Value, mlir::Value> stateInputBlockArgs;
   };

   void createSteps(subop::ExecutionGroupOp eg, Analysis& a,
                    llvm::DenseMap<mlir::Operation*, StepCtx>& ctxByRoot,
                    llvm::DenseMap<mlir::Value, mlir::Value>& stateMapping) {
      auto* groupTerm = eg.getSubOps().front().getTerminator();
      mlir::OpBuilder outerBuilder(eg.getContext());
      outerBuilder.setInsertionPoint(groupTerm);
      for (auto* root : a.topoOrder) {
         std::vector<mlir::Type> resultTypes;
         for (auto v : a.producedState[root]) resultTypes.push_back(v.getType());
         std::vector<mlir::Value> inputs;
         std::vector<mlir::Value> blockArgs;
         llvm::SmallVector<bool> threadLocal;
         auto* block = new mlir::Block;
         StepCtx ctx;
         for (auto v : a.requiredState[root]) {
            // Map original required-state value to the producing step's result
            // (or pass through if it's defined in the group's outer block).
            mlir::Value outerInput = v;
            auto it = stateMapping.find(v);
            if (it != stateMapping.end()) outerInput = it->second;
            inputs.push_back(outerInput);
            auto blockArg = block->addArgument(v.getType(), v.getLoc());
            blockArgs.push_back(blockArg);
            threadLocal.push_back(false);
            ctx.stateInputBlockArgs[v] = blockArg;
         }
         auto stepOp = outerBuilder.create<subop::ExecutionStepOp>(
            root->getLoc(), resultTypes, inputs,
            outerBuilder.getBoolArrayAttr(threadLocal));
         stepOp.getSubOps().getBlocks().push_back(block);
         // Provisional terminator — operands filled in Phase C after producing-state values are mapped.
         mlir::OpBuilder innerBuilder(eg.getContext());
         innerBuilder.setInsertionPointToStart(block);
         innerBuilder.create<subop::ExecutionStepReturnOp>(root->getLoc(), mlir::ValueRange{});
         ctx.step = stepOp;
         ctxByRoot[root] = ctx;
         // Map produced-state values to step results (used by later steps).
         for (auto [orig, res] : llvm::zip(a.producedState[root], stepOp.getResults())) {
            stateMapping[orig] = res;
         }
      }
   }

   //===----------------------------------------------------------------===//
   // Phase C — populate steps (move original into last pipeline; clone elsewhere)
   //===----------------------------------------------------------------===//

   void populateSteps(subop::ExecutionGroupOp eg, Analysis& a,
                      llvm::DenseMap<mlir::Operation*, StepCtx>& ctxByRoot) {
      // Per pipeline, an IRMapping that tracks: state-input → block-arg, plus
      // op-result substitutions as we move/clone within the pipeline.
      llvm::DenseMap<mlir::Operation*, mlir::IRMapping> mappingByRoot;
      for (auto* root : a.topoOrder) {
         auto& m = mappingByRoot[root];
         for (auto& [orig, blockArg] : ctxByRoot[root].stateInputBlockArgs) {
            m.map(orig, blockArg);
         }
      }

      // Walk pipelines in topo order. For each pipeline, walk its ops in
      // their original IR-order index (the lists from buildPipelines are
      // already in IR order because we iterated the body in IR order).
      for (auto* root : a.topoOrder) {
         auto& step = ctxByRoot[root].step;
         auto& mapping = mappingByRoot[root];
         auto* block = &step.getSubOps().front();
         auto* term = block->getTerminator();
         mlir::OpBuilder builder(eg.getContext());
         builder.setInsertionPoint(term);
         // ColumnMapping is shared across all clones in this pipeline so
         // that a renamed column produced by one clone (e.g. lookup_or_insert
         // creating a new @lookup_u_5__u_1::@ref) is propagated to subsequent
         // clones (e.g. a reduce that reads that column).
         subop::ColumnMapping columnMapping;

         for (auto* op : a.pipelines[root]) {
            // Just before processing each op, resolve any union-typed stream
            // operands to their current in-pipeline substitution. We do this
            // inline (rather than upfront) because a union's operand may
            // itself have been moved/cloned during processing of an earlier
            // op in this pipeline — the inline mapping picks up that update.
            for (auto operand : op->getOperands()) {
               if (!isStream(operand)) continue;
               if (operand.getDefiningOp<subop::UnionOp>()) {
                  mapping.map(operand, resolveStreamForPipeline(operand, root, mapping, a));
               }
            }
            bool isLast = a.lastPipelineFor[op] == root;
            if (isLast) {
               // MOVE: extract the original op, remap its operands using
               // `mapping`, insert into the step body. Update mapping for
               // results (identity — values are unchanged for moves).
               // The remap must walk *into* nested regions, because a
               // moved op (e.g., a nested_map) may have ops in its body
               // that reference values from outside the new step boundary.
               op->remove();
               op->walk([&](mlir::Operation* sub) {
                  for (auto& opOperand : sub->getOpOperands()) {
                     if (auto m = mapping.lookupOrNull(opOperand.get())) {
                        opOperand.set(m);
                     }
                  }
               });
               builder.insert(op);
               for (auto result : op->getResults()) {
                  mapping.map(result, result);
               }
            } else {
               // CLONE: only multi-rooted SubOp ops reach here (imperative
               // ops are merged via SSA closure into a single root, so they
               // are always last for their pipeline → MOVE path).
               auto subOp = mlir::dyn_cast<subop::SubOperator>(op);
               if (!subOp) {
                  op->emitError("OrganizeExecutionStepsPass: imperative op unexpectedly multi-rooted");
                  signalPassFailure();
                  return;
               }
               // cloneSubOp uses `mapping` for operand lookup AND updates
               // `mapping` with new result mappings. ColumnMapping is shared
               // across this pipeline (see above).
               subOp.cloneSubOp(builder, mapping, columnMapping);
            }
         }

         // Now finalize the step's terminator with produced-state values
         // (post-mapping, since they may be results of moved ops).
         std::vector<mlir::Value> termOperands;
         for (auto v : a.producedState[root]) {
            termOperands.push_back(mapping.lookupOrDefault(v));
         }
         term->setOperands(termOperands);
      }
   }

   //===----------------------------------------------------------------===//
   // Phase D — cleanup
   //===----------------------------------------------------------------===//

   void cleanup(subop::ExecutionGroupOp eg,
                llvm::DenseMap<mlir::Value, mlir::Value>& stateMapping) {
      // Erase unions (now unused).
      std::vector<subop::UnionOp> unions;
      for (mlir::Operation& op : eg.getSubOps().front()) {
         if (auto u = mlir::dyn_cast<subop::UnionOp>(op)) unions.push_back(u);
      }
      for (auto u : unions) {
         if (!u.getResult().use_empty()) {
            u.emitError("OrganizeExecutionStepsPass: union still has uses after step population");
            return signalPassFailure();
         }
         u.erase();
      }
      // Rewire execution_group_return through stateMapping.
      auto retOp = mlir::cast<subop::ExecutionGroupReturnOp>(eg.getSubOps().front().getTerminator());
      std::vector<mlir::Value> remapped;
      for (auto v : retOp.getInputs()) {
         remapped.push_back(stateMapping.count(v) ? stateMapping[v] : v);
      }
      retOp->setOperands(remapped);
   }

   //===----------------------------------------------------------------===//
   // Driver
   //===----------------------------------------------------------------===//

   // Wrap a func body into a single subop.execution_group. Moves the SSA
   // closure of all subop ops at body level — the subop ops themselves (incl.
   // subop.set_result, which lowers via SetResultOpLowering) plus any
   // top-level non-subop op that produces or consumes a value flowing from/to
   // them. The latter handles imperative DAGs threaded through subop steps
   // (util.pack/unpack between simple_state ↔ tuple conversions, scf.for/scf
   // bodies that compute on those values, …). Strictly post-set_result
   // runtime plumbing (util.varlen32_create_const, arith.constant, util.invoke
   // for appendTableFromResult/clearResult) is SSA-disconnected from the
   // subop graph and stays at func body level.
   void ensureWrapped(mlir::func::FuncOp funcOp) {
      if (funcOp.getBody().empty()) return;
      auto& body = funcOp.getBody().front();
      auto* terminator = body.getTerminator();

      llvm::DenseSet<mlir::Operation*> moveSet;
      std::vector<mlir::Operation*> worklist;
      for (auto& op : body) {
         if (&op == terminator) continue;
         if (mlir::isa<subop::SubOperatorDialect>(op.getDialect())) {
            moveSet.insert(&op);
            worklist.push_back(&op);
         }
      }
      if (worklist.empty()) return;

      // Top-level ancestor of `op` in `body`, or null if `op` isn't in this
      // func.
      auto topLevel = [&](mlir::Operation* op) -> mlir::Operation* {
         while (op && op->getBlock() != &body) op = op->getParentOp();
         return op;
      };
      auto enqueue = [&](mlir::Operation* op) {
         if (!op || op == terminator) return;
         if (op->getBlock() != &body) return;
         if (moveSet.insert(op).second) worklist.push_back(op);
      };
      while (!worklist.empty()) {
         auto* op = worklist.back();
         worklist.pop_back();
         op->walk([&](mlir::Operation* nested) {
            for (auto operand : nested->getOperands()) {
               if (auto* defOp = operand.getDefiningOp()) enqueue(topLevel(defOp));
            }
            for (auto result : nested->getResults()) {
               for (auto* user : result.getUsers()) enqueue(topLevel(user));
            }
         });
      }

      std::vector<mlir::Operation*> opsToMove;
      for (auto& op : body) {
         if (moveSet.contains(&op)) opsToMove.push_back(&op);
      }

      // An op is "inside the moved set" if it is a top-level moved op OR is
      // transitively nested under one — needed because moved ops carry their
      // region trees with them.
      auto isInsideMoveSet = [&](mlir::Operation* op) {
         while (op) {
            if (moveSet.contains(op)) return true;
            op = op->getParentOp();
         }
         return false;
      };
      auto isExternal = [&](mlir::Value v) {
         if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(v))
            return !isInsideMoveSet(blockArg.getOwner()->getParentOp());
         if (auto* defOp = v.getDefiningOp()) return !isInsideMoveSet(defOp);
         return true;
      };

      llvm::SetVector<mlir::Value> externalInputs;
      llvm::SetVector<mlir::Value> groupResults;
      for (auto* op : opsToMove) {
         op->walk([&](mlir::Operation* nested) {
            for (auto operand : nested->getOperands()) {
               if (isExternal(operand)) externalInputs.insert(operand);
            }
         });
         for (auto result : op->getResults()) {
            for (auto* user : result.getUsers()) {
               if (!moveSet.contains(user)) {
                  groupResults.insert(result);
                  break;
               }
            }
         }
      }

      mlir::OpBuilder builder(funcOp);
      builder.setInsertionPointToStart(&body);
      llvm::SmallVector<mlir::Type> resultTypes;
      for (auto v : groupResults) resultTypes.push_back(v.getType());
      llvm::SmallVector<mlir::Value> inputs(externalInputs.begin(), externalInputs.end());
      auto execGroup = builder.create<subop::ExecutionGroupOp>(funcOp.getLoc(), resultTypes, inputs);
      auto* newBody = new mlir::Block();
      for (auto v : externalInputs) newBody->addArgument(v.getType(), funcOp.getLoc());
      execGroup.getSubOps().push_back(newBody);

      for (auto* op : opsToMove) op->moveBefore(newBody, newBody->end());

      builder.setInsertionPointToEnd(newBody);
      builder.create<subop::ExecutionGroupReturnOp>(funcOp.getLoc(), groupResults.getArrayRef());

      // Rewire external operand uses inside the group → block args.
      for (size_t i = 0; i < externalInputs.size(); i++) {
         auto inp = externalInputs[i];
         auto blockArg = newBody->getArgument(i);
         inp.replaceUsesWithIf(blockArg, [&](mlir::OpOperand& opOperand) {
            return execGroup->isAncestor(opOperand.getOwner());
         });
      }
      // Rewire group-result uses outside the group → execGroup results.
      for (size_t i = 0; i < groupResults.size(); i++) {
         auto inner = groupResults[i];
         auto outer = execGroup.getResult(i);
         inner.replaceUsesWithIf(outer, [&](mlir::OpOperand& opOperand) {
            return !execGroup->isAncestor(opOperand.getOwner());
         });
      }
   }

   void runOnOperation() override {
      reifyGenerates();
      getOperation()->walk([&](mlir::func::FuncOp funcOp) { ensureWrapped(funcOp); });
      std::vector<subop::ExecutionGroupOp> egs;
      getOperation()->walk([&](subop::ExecutionGroupOp eg) { egs.push_back(eg); });
      for (auto eg : egs) {
         Analysis a;
         computeRoots(eg, a);
         buildPipelines(eg, a);
         computeStates(eg, a);
         buildDependencies(eg, a);
         if (!topoSort(a, eg)) return;
         computeLastPipelines(a);
         llvm::DenseMap<mlir::Operation*, StepCtx> ctxByRoot;
         llvm::DenseMap<mlir::Value, mlir::Value> stateMapping;
         createSteps(eg, a, ctxByRoot, stateMapping);
         populateSteps(eg, a, ctxByRoot);
         cleanup(eg, stateMapping);
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
subop::createOrganizeExecutionStepsPass() {
   return std::make_unique<OrganizeExecutionStepsPass>();
}
