import json

import lingodbbridge
import pyarrow as pa


def int_type(width=64, nullable=True):
    return {"base": "int", "nullable": nullable, "props": [width]}


def string_type(nullable=True):
    return {"base": "string", "nullable": nullable}


def float_type(width=64, nullable=True):
    return {"base": "float", "nullable": nullable, "props": [width]}


def column(name, type):
    return {"name": name, "type": type}


def create_meta_data(cols):
    return json.dumps({"columns": cols})


def metadata_from_arrow(table: pa.Table):
    schema = table.schema
    cols = []
    for colname in schema.names:
        field = schema.field(colname)
        t = field.type
        lt = None
        if t == pa.int8() or t == pa.int16() or t == pa.int32() or t == pa.int64():
            lt = int_type(t.bit_width, field.nullable)
        elif t == pa.string() or t == pa.utf8():
            lt = string_type(field.nullable)
        elif t == pa.float16() or t == pa.float32() or t == pa.float64():
            lt = float_type(t.bit_width, field.nullable)
        else:
            raise "Unsupported Type"
        cols.append(column(colname, lt))
    return create_meta_data(cols)


class Connection:
    def __init__(self, native_con):
        self.native_con = native_con

    def sql(self, query):
        return self.native_con.sql_query(query)

    def sql_stmt(self, stmt):
        return self.native_con.sql_stmt(stmt)

    def mlir(self, module):
        return self.native_con.mlir(module)

    def mlir_no_result(self, module):
        return self.native_con.mlir_no_result(module)

    def append_table(self, name, table):
        self.native_con.append(name, table)

    def create_table(self, name, metaData):
        self.native_con.create_table(name, metaData)

    def add_table(self, name, table, metadata=None):
        if not metadata:
            metadata = metadata_from_arrow(table)
        self.create_table(name, metadata)
        self.append_table(name, table)


def connect_to_db(path):
    return Connection(lingodbbridge.ext.connect_to_db(path))


def create_in_memory():
    return Connection(lingodbbridge.ext.in_memory())
