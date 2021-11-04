import os

from sql2mlir.mlir import Attribute, DBType
import json


class Table:
    def __init__(self, table_identifier, scope_name, row_count, pkey, columns):
        self.table_identifier = table_identifier
        self.scope_name = scope_name
        self.columns = columns
        self.row_count = row_count
        self.pkey = pkey


schema = {}


def loadSchema(path):
    global schema
    if not path:
        path = os.path.dirname(os.path.realpath(__file__)) + "/default.schema.json"
    with open(path) as json_file:
        schema = json.load(json_file)

def convertColumns(cols,scope_name):
    res={}
    for col in cols:
        res[col["name"]]=Attribute(scope_name,col["name"],DBType(col["type"]["base"],col["type"]["props"]))

    return res

def getTable(name, scope_name):
    for t in schema:
        if t["name"] == name:
            return Table(t["name"], scope_name, t["rows"], t["pkeys"], convertColumns(t["columns"],scope_name))

    raise Exception("unknown table: " + name)
