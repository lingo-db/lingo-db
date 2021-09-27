from pymlirdb.df import QueryDF
from pymlirdb.functions import FunctionRegistry
from sql2mlir.codegen import CodeGen
from sql2mlir.translator import Translator


class Generator:
    def __init__(self):
        self.functions=FunctionRegistry()
        self.codegen=CodeGen(self.functions)
        self.codegen.startModule("testmodule")
        self.codegen.startFunction("main",[],"")
        self.res=None
    def registerFunction(self,func):
        self.functions.register(func)

    def generate(self,df):
        if type(df) is QueryDF:
            translator=Translator(df.sql)
            self.res=translator.translate(self.codegen)
        pass

    def result(self):
        self.codegen.endFunction(self.res)
        self.functions.implement(self.codegen)
        self.codegen.endModule()
        res=self.codegen.getResult()
        print(res)
        return res