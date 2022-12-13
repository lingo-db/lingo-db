import requests
import subprocess
import sys
import re

objdir = sys.argv[1]
dataset = sys.argv[2]

header=""
results = []
for qnum in range(1, 23):
    file1 = f"resources/sql/tpch/{qnum}.sql"
    command = [objdir + "/run-sql", file1, "resources/data/"+dataset]
    print("processing: tpch query ", qnum, command)
    proc1 = subprocess.run(command, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    returncode=proc1.returncode
    runtime =0.0
    conversion=0.0
    if returncode==0:
        output = proc1.stdout.decode('utf8')
        headerRead=False
        for line in output.splitlines():
            if line.startswith("      name           QOpt"):
                header=line
                headerRead=True
            elif headerRead:
                results.append(line)
                break
            else:
                print(line)
print()
print("""
###############################################################################################################################################################################
#############################################################################    Results    ###################################################################################
###############################################################################################################################################################################
""")
print(header)
for res in results:
    print(res)