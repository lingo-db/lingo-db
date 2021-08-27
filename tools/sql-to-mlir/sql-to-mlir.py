from sql2mlir.translator import Translator
from sql2mlir.mlir import DBType
import sys

file=sys.argv[1]
query=""
with open(file) as f:
     query = f.read()
     f.close()
translator=Translator(query)
mlir=translator.translateModule({},{})
print(mlir)