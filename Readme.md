# GraphAlg Integration

## Current State

- [x] Initial integration into LingoDB
- [ ] optimizations (started)
- [ ] final benchmarks
- [ ] Verify sideeffects for SQL queries

```json
16:01 [WARN] [Runner r514306] => 16: 01 [WARN] LingoDB-MLIR: name                         QOpt   lowerGraphAlgHigh   lowerGraphAlgCore         lowerRelAlg          lowerSubOp             lowerDB          lowerArrow         lowerToLLVM    baselineLowering            toLLVMIR        llvmOptimize         llvmCodeGen     baselineCodeGen        baselineEmit       executionTime               total
16: 01 [WARN] [Runner r514306] => 16: 01 [WARN] LingoDB-MLIR: pagerank.mlir               2.728               1.101               0.195               0.512               4.425               0.771               0.252               5.892                                   1.494               3.263              44.393                                                     3521.09              3590.3
16: 02 [WARN] [Runner r922306] => 16:02 [WARN] LingoDB-MLIR: bfs.mlir                    2.972               1.386               0.244               0.481               5.021               0.796               0.267               6.141                                   1.515               3.532               41.37                                                     1607.48             1676.17
16: 02 [WARN] [Runner r663916] => 16: 02 [WARN] LingoDB-MLIR: sssp.mlir                   2.128               0.996                0.18               0.397               2.719               0.599               0.201               4.464                                   1.151                2.34              32.213                                                     5119.02             5171.71
16: 10 [WARN] [Runner r652048] => 16: 10 [WARN] LingoDB-MLIR: wcc.mlir                    2.267               1.009               0.173               0.435                3.25               0.654               0.207               4.828                                   1.246               2.781              36.039                                                     25980.2             26036.9         
```

CLDP: currently broken when activating JoinOptimizationPass

## Setup

No additional setup is required for this branch. Small graph lit tests are in the `test/lit/GraphAlg`

### Benchmarking

Benchmarking happens using the Graphalytic Benchmark Harness, see the dedicated LingoDB Graphalytic repository for more
information.

## Key problems to tackle

CLDP: currently broken when activating JoinOptimizationPass

Currently, results from within the loop body have to be materialized at the end of the loop body, so they can then
scanned again at the beginning of the loop. This is highly inperformant and causes plenty of memory allocations.
Ideally, we want to avoid having to materialize when the given stream is only used once and otherwise use the same
semantics as produced by the IntroduceTmpPass (materialize once and then scan the materialized buffer).

According to the folks at AvantGraph, in-place aggregation / state is the most important optimization that we are
missing in the current state.

## Relevant Files

Conversion to RelAlg: `src/compiler/Dialect/graphalg/Transforms/GraphAlgToLingoRelAlg.cpp`, this the where the magic
happens. We convert the lower-level GraphAlg-Core dialect to our own RelAlg dialect.

tests: `test/lit/GraphAlg`

## References to take a look at

The awesome GraphAlg documentation and reference implementation: https://wildarch.dev/graphalg/
and https://github.com/wildarch/graphalg

AvantGraph (not yet published at the time of writing this): https://github.com/avantlab/avantgraph

Original Readme:

<div align="center">
  <img src=".github/lingodb-black-title.svg" height="50">
</div>
<p>&nbsp;</p>
<p align="center">
<a href="https://github.com/lingo-db/lingo-db/actions/workflows/workflow-ubuntu-latest-x86_64.yml">
  <img src="https://github.com/lingo-db/lingo-db/actions/workflows/workflow-ubuntu-latest-x86_64.yml/badge.svg" alt="build+test (ubuntu 24.04 x86_64)">
</a>
  <a href="https://codecov.io/gh/lingo-db/lingo-db" >
    <img src="https://codecov.io/gh/lingo-db/lingo-db/branch/main/graph/badge.svg?token=7RC3UD5YEA"/>
  </a>
</p>

# LingoDB
LingoDB is a cutting-edge data processing system that leverages compiler technology to achieve unprecedented flexibility and extensibility without sacrificing performance. It supports a wide range of data-processing workflows beyond relational SQL queries, thanks to declarative sub-operators. Furthermore, LingoDB can perform cross-domain optimization by interleaving optimization passes of different domains and its flexibility enables sustainable support for heterogeneous hardware.

# Using LingoDB
You can try out LingoDB through different ways:
1. Use the hosted [SQL Webinterface](https://www.lingo-db.com/interface/)
2. Use the python package: `pip install lingodb`
3. Build it yourself by following the [documentation](https://www.lingo-db.com/docs/gettingstarted/install/#building-from-source)

## Documentation
For LingoDB's documentation, please visit [the documentation website](https://www.lingo-db.com/docs/) based on [this github repo](https://github.com/lingo-db/lingo-db.github.io).

## Contributing
Before contributing, please first read the [contribution guidelines](https://www.lingo-db.com/docs/next/ForDevelopers/Contributing).