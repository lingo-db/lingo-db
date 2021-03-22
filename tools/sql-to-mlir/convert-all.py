from translator import Translator

for query_number in range(1,22):
    file="./sql/tpch/"+str(query_number)+".sql"
    outfile="./mlir/tpch/"+str(query_number)+".mlir"

    query=""
    with open(file) as f:
         query = f.read()
         f.close()
    translator=Translator(query)
    mlir=translator.translate()
    with open(outfile,'w') as of:
        of.write(mlir)
        of.close()
