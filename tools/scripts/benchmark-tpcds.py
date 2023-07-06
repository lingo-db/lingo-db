import subprocess
import sys
import re

objdir = sys.argv[1]
dataset = sys.argv[2]

header=""
results = []
queries =["1.sql","2.sql","3.sql","4.sql","5.sql","6.sql","7.sql","8.sql","9.sql","10.sql","11.sql","12.sql","13.sql","14a.sql","14b.sql","15.sql","16.sql","17.sql","18.sql","19.sql","20.sql","21.sql","22.sql","23a.sql","23b.sql","24a.sql","24b.sql","25.sql","26.sql","27.sql","28.sql","29.sql","30.sql","31.sql","32.sql","33.sql","34.sql","35.sql","36.sql","37.sql","38.sql","39a.sql","39b.sql","40.sql","41.sql","42.sql","43.sql","44.sql","45.sql","46.sql","47.sql","48.sql","49.sql","50.sql","51.sql","52.sql","53.sql","54.sql","55.sql","56.sql","57.sql","58.sql","59.sql","60.sql","61.sql","62.sql","63.sql","64.sql","65.sql","66.sql","67.sql","68.sql","69.sql","70.sql","71.sql","72.sql","73.sql","74.sql","75.sql","76.sql","77.sql","78.sql","79.sql","80.sql","81.sql","82.sql","83.sql","84.sql","85.sql","86.sql","87.sql","88.sql","89.sql","90.sql","91.sql","92.sql","93.sql","94.sql","95.sql","96.sql","97.sql","98.sql","99.sql"]
for query in queries:
    file1 = f"resources/sql/tpcds/{query}"
    command = [objdir + "/run-sql", file1, "resources/data/"+dataset]
    print("processing: tpcds query ", query, command)
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