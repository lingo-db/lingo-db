import pymlirdb.df
import sys,os
sys.setdlopenflags(os.RTLD_NOW|os.RTLD_GLOBAL)
import pyarrow as pa
import pymlirdbext
from pymlirdb.generator import Generator
generator=Generator()
def read_table(tableName):
    return {} #todo
def load_tables(tables):
    pymlirdbext.load(tables)
def registerFunction(func):
    generator.registerFunction(func)
def query(sql):
    return pymlirdb.df.QueryDF(sql)