import sys,os
import ctypes
import json

dir_path = os.path.dirname(os.path.realpath(__file__))
ctypes.CDLL(dir_path+'/libs/libpybridge.so',os.RTLD_GLOBAL|os.RTLD_NOW|os.RTLD_DEEPBIND)
import pyarrow as pa
import platform
from pathlib import Path
#pa.create_library_symlinks()
if platform.system().lower() == 'linux':
    import ctypes
    def _set_arrow_symbol_resolution(flag):
        for dir in map(Path, pa.get_library_dirs()):
            arrow_path = dir / 'libarrow.so'
            arrow_python_path = dir / 'libarrow_python.so'
            if arrow_path.exists() and arrow_python_path.exists():
                arrow_python = ctypes.CDLL(arrow_path, flag)
                libarrow_python = ctypes.CDLL(arrow_python_path, flag)
                break
    _set_arrow_symbol_resolution(ctypes.RTLD_GLOBAL)
from . import ext
def connect_to_db(path):
    return ext.connect_to_db(path)
def create_in_memory():
    return ext.in_memory()

def int_type(width=64, nullable=True):
    return {"base": "int", "nullable": nullable, "props": [width]}


def column(name, type):
    return {"name": name, "type": type}


def create_meta_data(cols):
    return json.dumps({"columns": cols})


def meta_data_from_arrow(table: pa.Table):
    schema = table.schema
    cols = []
    for colname in schema.names:
        field = schema.field(colname)
        t = field.type
        lt = None
        if t == pa.int8() or t == pa.int16() or t == pa.int32() or t == pa.int64():
            lt = int_type(t.bit_width, field.nullable)
        else:
            raise "Unsupported Type"
        cols.append(column(colname, lt))
    return create_meta_data(cols)
