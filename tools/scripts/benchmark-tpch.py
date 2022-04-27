import requests
import subprocess
import sys
import re

objdir = sys.argv[1]
dataset = sys.argv[2]


class QueryResult:
    def __init__(self, query, runtime,optimization_time,lower_to_db_time,lower_to_std_time,lower_to_llvm_time,conversion_time,llvm_time, return_code):
        self.query = query
        self.runtime = runtime
        self.optimization_time=optimization_time
        self.lower_to_db_time = lower_to_db_time
        self.lower_to_std_time = lower_to_std_time
        self.lower_to_llvm_time = lower_to_llvm_time
        self.conversion_time=conversion_time
        self.llvm_time=llvm_time-conversion_time #hack
        self.return_code = return_code


results = []
for qnum in range(1, 23):
    file1 = f"resources/sql/tpch/{qnum}.sql"
    command = [objdir + "/run-sql", file1, "resources/data/"+dataset]
    print("processing: tpch query ", qnum, command)
    proc1 = subprocess.run(command, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
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
        m = re.search(r'lowering to db took: ([\d,\.,e,\+]+) ms', output)
        assert m is not None, 'Unexpected output:\n' + output
        lower_to_db_time = float(m.group(1))
        m = re.search(r'lowering to std took: ([\d,\.,e,\+]+) ms', output)
        assert m is not None, 'Unexpected output:\n' + output
        lower_to_std_time = float(m.group(1))
        m = re.search(r'lowering to llvm took: ([\d,\.,e,\+]+) ms', output)
        assert m is not None, 'Unexpected output:\n' + output
        lower_to_llvm_time = float(m.group(1))
        m = re.search(r'conversion: ([\d,\.,e,\+]+) ms', output)
        assert m is not None, 'Unexpected output:\n' + output
        conversion = float(m.group(1))
        m = re.search(r'jit: ([\d,\.,e,\+]+) ms', output)
        assert m is not None, 'Unexpected output:\n' + output
        jit = float(m.group(1))
        print(output)
    results.append(QueryResult("tpch" + str(qnum), runtime,optimization_time,lower_to_db_time,lower_to_std_time,lower_to_llvm_time,conversion,jit, returncode))
print('%12s %12s %12s %12s %12s %12s %12s %12s %12s' % ("#Query", "Runtime","Query Opt.","-> db","-> std","-> llvm","-> LLVM","LLVM", "Error?"))
for res in results:
    print('%12s %12.2f %12.2f %12.2f %12.2f %12.2f %12.2f %12.2f %12i' % (res.query, res.runtime,res.optimization_time,res.lower_to_db_time,res.lower_to_std_time,res.lower_to_llvm_time,res.conversion_time,res.llvm_time , res.return_code))
