import sql2mlir.tables
from sql2mlir.translator import Translator
import sys

file = sys.argv[1]
schemapath = sys.argv[2] if len(sys.argv) == 3 else None
sql2mlir.tables.loadSchema(path=schemapath)
query = ""
with open(file) as f:
    query = f.read()
    f.close()
translator = Translator(query)
mlir = translator.translateModule({}, {})
print(mlir)
