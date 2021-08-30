import pymlirdb.df
from pymlirdb.generator import Generator

generator=Generator()
def read_table(tableName):
    return {} #todo

def registerFunction(func):
    generator.registerFunction(func)
def query(sql):
    return pymlirdb.df.QueryDF(sql)