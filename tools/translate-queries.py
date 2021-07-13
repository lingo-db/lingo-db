import requests
import subprocess
import sys
import re



results = []
for qnum in range(1, 22):
    print("processing: tpch query ", qnum)
    file1 = "resources/sql/parseable/" + str(qnum) + ".sql"
    fileout = "resources/mlir/tpch-queries/" + str(qnum) + ".mlir"
    proc1 = subprocess.run(
        "python3 ./tools/sql-to-mlir/sql-to-mlir.py " + file1 + " > "+fileout,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE, shell=True)

