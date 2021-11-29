import requests
import subprocess
import sys
import re
filename=sys.argv[3]
objdir = sys.argv[1]
dataset = sys.argv[2]
proc1 = subprocess.run(
    objdir + "/db-run-query " + filename +" resources/data/" + dataset,
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
returncode = proc1.returncode
runtime = 0.0
if returncode == 0:
    output = proc1.stdout.decode('utf8')
    print("//RUN: db-run-query %s %S/../../../resources/data/tpch | FileCheck %s")
    for line in output.splitlines():
        if line.startswith("|") or line.startswith("-"):
            print("//CHECK:",line)
    with open(filename) as qf:
        print("".join(qf.readlines()))
