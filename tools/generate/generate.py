import os

import pyarrow
from pyarrow import csv
import json
import random
import pyarrow.compute


def convertTypeToArrow(type):
    base = type["base"]
    if base == "int":
        width = int(type["props"][0])
        if width == 8:
            return pyarrow.int8()
        elif width == 16:
            return pyarrow.int16()
        elif width ==32:
            return pyarrow.int32()
        elif width==64:
            return pyarrow.int64()
        else:
            raise "problem"
    if base == "date":
        return pyarrow.date32()
    elif base == "string":
        return pyarrow.string()
    elif base == "decimal":
        return pyarrow.decimal128(int(type["props"][0]), int(type["props"][1]))
    elif base == "char":
        return pyarrow.binary(int(type["props"][0]))


def createArrowColumnTypes(cols):
    res = {}
    for col in cols:
        colname = col["name"]
        type = convertTypeToArrow(col["type"])
        res[colname] = type
    return res



def createColumnNames(cols):
    res = []
    for col in cols:
        res.append(col["name"])
    return res


def convertToArrowTable(inpath, outpath, table_schema,meta):
    tablename = table_schema["name"]
    filepath = inpath + "/" + tablename + ".tbl"
    output_filepath = outpath + '/' + tablename + '.arrow'
    if not os.path.exists(filepath):
        return
    print("converting", filepath, "->", output_filepath)
    meta["tables"][tablename]={}
    table_meta_obj=meta["tables"][tablename]
    table = csv.read_csv(filepath,
                         convert_options=pyarrow.csv.ConvertOptions(
                             column_types=createArrowColumnTypes(table_schema["columns"])),
                         parse_options=pyarrow.csv.ParseOptions(delimiter='|'),
                         read_options=pyarrow.csv.ReadOptions(column_names=createColumnNames(table_schema["columns"])))
    table_meta_obj["num_rows"]=table.num_rows
    table_meta_obj["pkey"]=table_schema["pkeys"]
    table_meta_obj["columns"]=[]
    for c in table_schema["columns"]:
        col_meta_obj= {}
        col_meta_obj["name"]=c["name"]
        col_meta_obj["type"]=c["type"]
        col_meta_obj["distinct_values"]=(len(table.select([c["name"]]).to_pandas().drop_duplicates()))
        table_meta_obj["columns"].append(col_meta_obj)
    writer = pyarrow.RecordBatchFileWriter(output_filepath, table.schema)
    writer.write_table(table)
    writer.close()
    sample=table.take(random.sample(range(0,table.num_rows),min(1024,table.num_rows)))
    writer = pyarrow.RecordBatchFileWriter(output_filepath+'.sample', sample.schema)
    writer.write_table(sample)
    writer.close()

import sys
random.seed(0)

input_dir = sys.argv[1]
output_dir = sys.argv[2]
meta={}
meta["tables"]={}

with open(os.path.dirname(os.path.realpath(__file__)) + "/default.schema.json", "r") as schema_file:
    schema = json.load(schema_file)
for table_schema in schema:
    convertToArrowTable(input_dir, output_dir, table_schema,meta)
with open(output_dir+'/metadata.json', 'w') as f:
    json.dump(meta,f)
