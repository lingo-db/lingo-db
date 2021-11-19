import requests
import subprocess
import sys
import json
import os
directory=sys.argv[1]
languages={"C/C++ Header":"Headers","C++":"C++","TableGen":"TableGen"}

schema={
    "Dialects": {"inc": ["include/mlir/Dialect", "lib/RelAlg","lib/DB","lib/util"], "exc": ["Transforms"],"nosum":True},
    " $\\rightarrow$ relalg": {"inc": ["include/mlir/Dialect/RelAlg", "lib/RelAlg"], "exc": ["Transforms"]},
    " $\\rightarrow$ db": {"inc": ["include/mlir/Dialect/DB", "lib/DB"], "exc": []},
    " $\\rightarrow$ util":{"inc":["include/mlir/Dialect/util","lib/util"],"exc":[]},
    "Query Opt.": {"inc":["include/mlir/Dialect/RelAlg/Transforms", "lib/RelAlg/Transforms"],"exc":[]},
    "Lowerings":{"inc": ["include/mlir/Conversion", "lib/Conversion"], "exc": [],"nosum":True},
    " $\\rightarrow$ relalg to db": {"inc":["include/mlir/Conversion/RelAlgToDB","lib/Conversion/RelAlgToDB"],"exc":[]},
    " $\\rightarrow$ db to std": {"inc": ["include/mlir/Conversion/DBToArrowStd","lib/Conversion/DBToArrowStd"], "exc": []},
    " $\\rightarrow$ util to llvm": {"inc": ["include/mlir/Conversion/UtilToLLVM", "lib/Conversion/UtilToLLVM"],"exc": []},
    "Runtime": {"inc": ["include/runtime", "lib/runtime"], "exc": []},
    "Runner": {"inc": ["include/runner", "lib/runner"], "exc": []},

}
def runCLOC(inc,exc):
    dirlist=" ".join(list(map(lambda val:directory+"/"+val,inc)))
    excllist=",".join(exc)
    if len(excllist)>0:
        excllist="--exclude-dir="+excllist
    proc1 = subprocess.run(
        "cloc --json  --exclude-lang=CMake --force-lang-def="+os.path.dirname(os.path.realpath(__file__))+"/cloc.defs "+excllist+" "+dirlist,
        stdout=subprocess.PIPE, shell=True)
    res = proc1.stdout.decode('utf8')
    return json.loads(res)
sum=0
sums={"C/C++ Header":0,"C++":0,"TableGen":0}
def format(name,j,nosumup):
    global sum
    global sums
    print(name, end=' ')
    for l in languages:
        print(" & ", end='')
        if l in j:
            print(j[l]['code'],end='')
            if not nosumup:
                sum+=int(j[l]['code'])
                sums[l]+=int(j[l]['code'])
        else:
            print("-",end='')
    print(" \\\\")
print("Components",end='')
for l in languages:
    print(" & "+languages[l],end='');
print(" \\\\\\toprule")
for k in schema:
    format(k,runCLOC(schema[k]["inc"],schema[k]["exc"]),"nosum" in schema[k])
print(" \\midrule")
print("$\Sigma$",end='')
for l in languages:
    print(" & "+str(sums[l]),end='');
print(" \\\\\\bottomrule")



