import requests
import subprocess
import sys
import re

objdir = sys.argv[1]
dataset = sys.argv[2]


class QueryResult:
    def __init__(self, query, runtime,optimization_time,lower_to_std_time,lower_to_llvm_time,llvm_time, return_code):
        self.query = query
        self.runtime = runtime
        self.optimization_time=optimization_time
        self.lower_to_std_time = lower_to_std_time
        self.lower_to_llvm_time = lower_to_llvm_time
        self.llvm_time=llvm_time
        self.return_code = return_code


results = []
for qnum in range(1, 23):
    print("processing: tpch query ", qnum)
    file1 = "resources/mlir/tpch-queries/" + str(qnum) + ".mlir"
    proc1 = subprocess.run(
        objdir + "/db-run-query "+file1+" resources/data/"+dataset,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE, shell=True)
    returncode=proc1.returncode
    runtime =0.0
    if returncode==0:
        output = proc1.stdout.decode('utf8')
        m = re.search(r'runtime: ([\d,\.,e,\+]+) ms', output)
        assert m is not None, 'Unexpected output:\n' + output
        runtime = float(m.group(1))
        m = re.search(r'optimization took: ([\d,\.,e,\+]+) ms', output)
        assert m is not None, 'Unexpected output:\n' + output
        optimization_time = float(m.group(1))
        m = re.search(r'lowering to std took: ([\d,\.,e,\+]+) ms', output)
        assert m is not None, 'Unexpected output:\n' + output
        lower_to_std_time = float(m.group(1))
        m = re.search(r'lowering to llvm took: ([\d,\.,e,\+]+) ms', output)
        assert m is not None, 'Unexpected output:\n' + output
        lower_to_llvm_time = float(m.group(1))
        m = re.search(r'jit: ([\d,\.,e,\+]+) ms', output)
        assert m is not None, 'Unexpected output:\n' + output
        jit = float(m.group(1))
        print(output)
    results.append(QueryResult("tpch" + str(qnum), runtime,optimization_time,lower_to_std_time,lower_to_llvm_time,jit, returncode))

for res in results:
    print('%12s %5i %5i %5i %5i %5i %5i' % (res.query, res.runtime,res.optimization_time,res.lower_to_std_time,res.lower_to_llvm_time,res.llvm_time , res.return_code))
