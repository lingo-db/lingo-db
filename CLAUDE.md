# graphalg branch — performance work

## Core finding
Graph algos (WCC, BFS, LCC, …) left CPU cores idle while TPC-H parallelized fine. Root
cause: tiny per-iteration frontiers. Dispatching a morsel task wakes every worker to contend
for a handful of 200-element morsels, then sleep — that wakeup/sync overhead dwarfs the
actual work. Dense graphs (e.g. dota-league) instead bottleneck on data volume. The branch
attacks this on two fronts: avoid dispatching parallelism when it doesn't pay, and cut the
per-iteration allocation/rebuild cost inside fixpoint loops.

## Build & test
```
mold -run cmake --build build/lingodb-debug/ --target mlir-db-opt run-mlir run-sql sql-to-mlir sqlite-tester tester sql run-mlir-csv -- -j24 && clear && make test-no-rebuild-lit
```
- `make test-no-rebuild-lit` runs the lit suite without re-linking.
- e2e graphalg tests: `test/lit/GraphAlg/e2e/` (wcc, sssp, bfs, pagerank, lcc, reachability).
  Transform/structure tests alongside (e.g. `semi-naive.mlir`).
- **`cldp.mlir` / `cldp_simple.mlir` are expected to FAIL — ignore them.** CDLP (community
  detection / label propagation) uses an argmax that isn't yet lowered; unrelated to this
  branch's work. A green run = all pass except those two.

## Execution model essentials
- **A tuplestream can be consumed only once.** Multiple consumers require materialization.
  `RelAlg/Transforms/IntroduceTmp.cpp` inserts a `relalg.tmp` to fan out — but that
  materializes, so it is costly. On this branch IntroduceTmp special-cases a `buffer_scan`
  with multiple consumers: clone the scan per use instead of materializing.
- **`buffer_scan`** (relalg) reads a runtime buffer; cheap to duplicate (vs a `tmp`).
- Pipeline: graphalg → graphalg-core → relalg → subop → DB/arrow → LLVM. `subop.loop` is NOT
  IsolatedFromAbove, so loop-invariant state can be referenced from before the loop.

## WCC perf (two-stage, this branch)
- **GraphAlgFuseMatMul** (`graphalg/GraphAlgFuseMatMul.cpp`, pass
  `--graphalg-fuse-matmul`, in `buildGraphAlgToCorePipeline` before split-aggregate). Semiring
  distributivity: within an `ewise ADD` tree, fuse MatMuls sharing the same rhs into one,
  `A·X ⊕ Aᵀ·X → (A∪Aᵀ)·X`. WCC's undirected propagation runs one matmul (+ one reduce/pick_any)
  instead of two; `A∪Aᵀ` is loop-invariant so its edge index builds once. ~2x on dota-league.
- **argmin accumulating map** (in `GraphAlgToLingoRelAlg.cpp`, the `convertLoop` /
  `GraphAlgYieldOpConversion` accumulating path). WCC carries labels as a bool matrix with the
  label in the column, collapsed by `pick_any` (per-row min col) — a row-keyed tropical-min
  monoid. The carried state is now a persistent `subop.hashmap<[row],[label]>` merged in place
  with the per-iter candidate delta (`lookup_or_insert`+min `reduce`, O(Δ)), eliminating the
  per-iteration full-|V| re-materialize + `union all`+`aggregation min` rebuild. The frontier
  Δ' (improved cells) is a **pre-merge** map lookup (`subop.lookup`/`unwrap`/`gather`, compared
  `cand < current`); execution-step ordering keeps that read before the in-place merge. The
  semi-naive `until` was bound to the (now-erased) delta op, so for argmin loops the folded
  until subtree is erased and termination is rebuilt from the new frontier (continue iff any
  frontier row AND idx<bound). Note: the generic accumulating (`deferred_reduce`) path was
  latent/never-triggered (semi-naive's `delta(M_next,M)` gives `M_next` two uses → fails the
  `hasOneUse` guard); Stage B is the first live user.

## Relevant passes (all in the default pipeline unless noted)
- **HoistInvariantStatePass** (`SubOperator/Transforms/`, `Execution.cpp:174`) — LICM for
  `subop.loop`. graphalg loops rebuild a join's build side (the adjacency/edge index of a
  matmul/vxm) every iteration; only the lookup depends on the per-iteration frontier.
  Hoists the `create buffer` + `materialize edges` + `create_hash_indexed_view` out of the
  loop. Tracks state mutation transitively through ref/list SSA values; a create is hoisted
  only if all its in-loop mutators are too.
- **ReuseLoopScratchPass** (`SubOperator/Transforms/`, `Execution.cpp:175`) — each
  `subop.create` is freed only at end-of-query, so an N-iteration loop leaks N copies of
  transient build/aggregation state → OOM on large graphs. Hoists iteration-local scratch
  `create`s before the loop and emits `subop.clear` at the body top (reset in place, keep
  the allocation). Only states dead at the iteration boundary are eligible; carried state
  (operand of `subop.loop_continue`, e.g. the ping-pong frontier) is left in place. Must run
  before SplitIntoExecutionSteps and Parallelize.
- **IntroduceTmp** (`RelAlg/Transforms/`) — see tuplestream note above.
- **Adaptive parallel scan** — `runtime/Buffer.cpp` `getParallelScanThreshold()` (env
  `LINGODB_PARALLEL_THRESHOLD`, default 400). Below it, scans run inline on the calling
  worker: no wakeup, single thread-local state ⇒ free merge.
- **Cardinality** — `graphalg/Transforms/GraphAlgToLingoRelAlg.cpp`
  `estimateScanRows`/`estimateDimSize` (replaced hardcoded `rows=100`). Edge/captured scans
  (`loopInvariant=true`) get an nnz estimate so the edge table is always the larger join
  side; state scans use |V|. Fixes build/probe placement.
- **GraphAlgSemiNaive** (`graphalg/GraphAlgSemiNaive.cpp`, pass `--graphalg-semi-naive`, in
  `buildGraphAlgToCorePipeline`) — delta iteration. Idempotent semiring + linear body ⇒ loop
  carries (M, Δ); body re-runs on Δ only; `M' = OUTER(M ∪ f(Δ))`, `Δ' = delta(M', M)`;
  `until` stops when all Δ empty. Applies to **WCC + SSSP** only. `graphalg.delta` lowers to
  `relalg.antisemijoin`. Eligibility: empty original `until`, every state idempotent (bool
  OR / tropical min/max), each yield = `M ⊕ g(M)` with g linear + self-accumulating, final
  reducer `pick_any`/`deferred_reduce`.

# Core goals of this branch
We want to test how to integrate graphalg into LingoDB. The core evaluation happens in another benchmark repo, I will do all large dataset testing and benchmarking on my own. Relevant dataset are like dota-league or cit-patents, so size S. Currently most algorithms work, PR and SSSP are already faster than in other db implementations. What we still have to focus on are BFS and WCC. Especially WCC runs currently for most parts just singlethread and is overall extremly slow ,e.g. 30x slower than other implemetnations. 


## vendored/graphalg
This contains a reference implementation, which is imperformant and only works for like 200 nodes. Ignored if not explicit asked to look at it.


## other helpsful things,
snapshotting:
```bash 
export LINGODB_SNAPSHOT_PASSES=true build/lingodb-debug/run-mlir test/lit/GraphAlg/e2e/wcc.mlir
```
this creates several detailed-snapshots-*.mlir. In detailed-snapshot-info.json you will find which pass happens when and which was executed and so on. So please always read this.

