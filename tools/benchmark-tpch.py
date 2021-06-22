import requests
import subprocess
import sys
import re

objdir = sys.argv[1]


class QueryResult:
    def __init__(self, query, runtime, return_code):
        self.query = query
        self.runtime = runtime
        self.return_code = return_code


results = []
for qnum in range(1, 22):
    print("processing: tpch query ", qnum)
    file1 = "resources/sql/parseable/" + str(qnum) + ".sql"
    proc1 = subprocess.run(
        "python3 ./tools/sql-to-mlir/sql-to-mlir.py " + file1 + " | " + objdir + "/mlir-db-opt --relalg-extract-nested-operators --relalg-decompose-lambdas --relalg-implicit-to-explicit-joins --relalg-pushdown --relalg-unnesting --relalg-optimize-join-order --relalg-combine-predicates --relalg-optimize-implementations | " + objdir + "/mlir-db-opt -relalg-to-db -canonicalize | timeout 10 " + objdir + "/db-run \"-\" resources/data/tpch-1",
        stdout=subprocess.PIPE,stderr=subprocess.PIPE, shell=True)
    returncode=proc1.returncode
    runtime =0.0
    if returncode==0:
        output = proc1.stdout.decode('utf8')
        m = re.search(r'runtime: ([\d,\.,e,\+]+) ms', output)
        assert m is not None, 'Unexpected output:\n' + output
        runtime = float(m.group(1))
        print(output)
    results.append(QueryResult("tpch" + str(qnum), runtime, returncode))

for res in results:
    print('%12s %5i %5i' % (res.query, res.runtime, res.return_code))
