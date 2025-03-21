import json

import lingodbbridge
import pyarrow as pa


def int_type(width=64, nullable=True):
    return f"int{width // 8}{"" if nullable else " not null"}"


def string_type(nullable=True):
    return f"string{"" if nullable else " not null"}"


def float_type(width=64, nullable=True):
    assert width == 64 or width == 32
    return f"{"float" if width == 32 else "double"}{"" if nullable else " not null"}"
def create_create_table_stmt(name, table: pa.Table):
    res = f" create table {name} ("
    schema = table.schema
    first = True
    for colname in schema.names:
        if not first:
            res += ", "
        else:
            first = False
        field = schema.field(colname)
        t = field.type
        lt = None
        if t == pa.int8() or t == pa.int16() or t == pa.int32() or t == pa.int64():
            res += f"{colname} {int_type(t.bit_width, field.nullable)}"
        elif t == pa.string() or t == pa.utf8():
            res += f"{colname} {string_type(field.nullable)}"
        elif t == pa.float16() or t == pa.float32() or t == pa.float64():
            res += f"{colname} {float_type(t.bit_width, field.nullable)}"
        else:
            raise "Unsupported Type"
    res += ");"
    return res



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

    def add_table(self, name, table):
        self.sql_stmt(create_create_table_stmt(name, table))
        self.append_table(name, table)


def connect_to_db(path):
    return Connection(lingodbbridge.ext.connect_to_db(path))


def create_in_memory():
    return Connection(lingodbbridge.ext.in_memory())
