import os

import pyarrow
from pyarrow import csv
import json


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


def convertTypeToMLIR(self, descr: str):
    parts = descr.split(" ")
    nullable = True
    if len(parts) == 3 and parts[1] == "not" and parts[2] == "null":
        nullable = False
    print(parts, "nullable", nullable)

    def convertRaw(typedescr):
        if typedescr.startswith("integer"):
            return {"base": "int", "props": ["64"]}
        if typedescr.startswith("date"):
            return {"base": "date", "props": ["day"]}
        elif typedescr.startswith("char") or typedescr.startswith("varchar"):
            return {"base": "string", "props": []}
        elif typedescr.startswith("decimal"):
            props = typedescr.split("(")[1].split(")")[0].split(",")
            precision = int(props[0])
            scale = int(props[1])
            return {"base": "decimal", "props": [str(precision), str(scale)]}

    res = convertRaw(parts[0])
    if nullable:
        res["props"].append("nullable")
    return res


def createMLIRColumns(self, cols):
    res = []
    for col in cols:
        type = self.convertTypeToMLIR(col["type"])
        res.append({"name": col["name"], "type": type})
    return res


def createColumnNames(cols):
    res = []
    for col in cols:
        res.append(col["name"])
    return res


def convertToArrowTable(inpath, outpath, table):
    tablename = table["name"]
    filepath = inpath + "/" + tablename + ".tbl"
    output_filepath = outpath + '/' + tablename + '.arrow'
    if not os.path.exists(filepath):
        return
    print("converting", filepath, "->", output_filepath)
    table = csv.read_csv(filepath,
                         convert_options=pyarrow.csv.ConvertOptions(
                             column_types=createArrowColumnTypes(table["columns"])),
                         parse_options=pyarrow.csv.ParseOptions(delimiter='|'),
                         read_options=pyarrow.csv.ReadOptions(column_names=createColumnNames(table["columns"])))
    writer = pyarrow.RecordBatchFileWriter(output_filepath, table.schema)
    writer.write_table(table)
    writer.close()


import sys

input_dir = sys.argv[1]
output_dir = sys.argv[2]

with open(os.path.dirname(os.path.realpath(__file__)) + "/default.schema.json", "r") as schema_file:
    schema = json.load(schema_file)
for table_schema in schema:
    convertToArrowTable(input_dir, output_dir, table_schema)
