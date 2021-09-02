import pymlirdb
import sys,os
sys.setdlopenflags(os.RTLD_NOW|os.RTLD_GLOBAL)
import pyarrow as pa
import pymlirdbext


class DataFrame(object):
    def __init__(self):
        pass
    def to_pandas(self):
        pymlirdb.generator.generate(self)
        mlirModule=pymlirdb.generator.result()
        res=pymlirdbext.run(mlirModule)
        return res.to_pandas()
class QueryDF(DataFrame):
    def __init__(self,sql):
        self.sql=sql