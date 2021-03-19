from moz_sql_parser import parse
from tables import getTPCHTable
from mlir import CodeGen, DBType, Attribute
from translator import Translator

for query_number in range(13,23):
    file="./sql/tpch/"+str(query_number)+".sql"
    print(file)
    query=""
    with open(file) as f:
         query = f.read()
         f.close()
    translator=Translator(query)
    mlir=translator.translate()






