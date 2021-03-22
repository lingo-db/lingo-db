#!/bin/bash
set -e
FILES=tools/sql-to-mlir/mlir/tpch/*
for f in $FILES
do
  echo "Process '$f':"
  cmake-build-debugllvm-release/bin/mlir-db-opt  $f $@
done