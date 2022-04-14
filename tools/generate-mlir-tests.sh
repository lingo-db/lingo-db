#!/bin/bash
for i in {1..22}
do
   python3 tools/generate-mlir-test.py build/lingodb-debug tpch resources/mlir/tpch-queries/${i}.mlir > test/RunRelAlg/tpch/${i}.mlir
done
