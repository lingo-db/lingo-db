import requests
import subprocess
import sys
objdir=sys.argv[1]
for qnum in range(1, 22):
    print("processing:",qnum)
    file1 = "resources/sql/tpch/" + str(qnum) + ".sql"
    file2 = "resources/sql/tpch/" + str(qnum) + ".sql"

    proc1 = subprocess.run(
        objdir+"/sql-to-mlir " + file1 + " resources/data/tpch/metadata.json | "+objdir+"/mlir-db-opt --relalg-extract-nested-operators --relalg-decompose-lambdas --relalg-implicit-to-explicit-joins --relalg-pushdown --relalg-unnesting --relalg-optimize-join-order | "+objdir+"/mlir-to-sql",
        stdout=subprocess.PIPE, shell=True)
    mlir_query = proc1.stdout.decode('utf8')
    r1 = requests.post('https://hyper-db.de/interface/query', data={'query': mlir_query})
    with open(file2) as f:
        query = f.read()
        f.close()
    res1 = r1.json()
    r2 = requests.post('https://hyper-db.de/interface/query', data={'query': query})
    res2 = r2.json()


    def compare(c1: dict, c2: dict):
        if len(c1["columns"]) != len(c2["columns"]):
            return False
        if len(c1["result"]) != len(c2["result"]):
            return False
        for a, b in zip(c1["result"], c2["result"]):
            for x, y in zip(a, b):
                # ignore subtle differences in number precision for now
                if x != y:
                    try:
                        float(x)
                        float(y)
                        if not (x.startswith(y) or y.startswith(x)):
                            return False
                    except ValueError:
                        return False
        return True


    if not compare(res1, res2):
        print("test failed for query", qnum)
        exit(1)
