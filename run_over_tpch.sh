#!/bin/bash
set -e
FILES=resources/sql/parseable/*
for f in $FILES
do
  echo "Process '$f':"
 python3 ./tools/sql-to-mlir/sql-to-mlir.py $f | build/build-debug-llvm-release/mlir-db-opt $@
done