from sql2mlir.translator import Translator
import sys

file=sys.argv[1]
query=""
with open(file) as f:
     query = f.read()
     f.close()
translator=Translator(query)
mlir=translator.translate()
print(mlir)