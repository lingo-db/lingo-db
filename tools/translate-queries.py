import requests
import subprocess
import sys
import re



results = []
for qnum in range(1, 23):
    print("processing: tpch query ", qnum)
    file1 = "resources/sql/hyper/" + str(qnum) + ".sql"
    fileout = "resources/mlir/tpch-queries/" + str(qnum) + ".mlir"
    proc1 = subprocess.run(
        "./build/build-debug-llvm-release/sql-to-mlir  " + file1 + " > "+fileout,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE, shell=True)

