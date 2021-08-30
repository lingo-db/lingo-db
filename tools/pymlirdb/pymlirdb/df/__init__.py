import pymlirdb


class DataFrame(object):
    def __init__(self):
        pass
    def to_pandas(self):
        pymlirdb.generator.generate(self)
        pymlirdb.generator.result()
class QueryDF(DataFrame):
    def __init__(self,sql):
        self.sql=sql