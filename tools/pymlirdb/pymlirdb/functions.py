import queue

from pymlirdb.compiler import compileFn


class FunctionRegistry:
    def __init__(self):
        self.registered={}
        self.implemented=set()
        self.to_implement=queue.Queue()

    def implement(self,codegen):

        while not self.to_implement.empty():
            currFn=self.to_implement.get()
            compileFn(currFn.function,codegen)
            self.implemented.add(currFn.name)
    def get(self,name):
        if not name in self.implemented:
            self.implemented.add(name)
            self.to_implement.put(self.registered[name])
        return self.registered[name]
    def register(self,func):
        self.registered[func.name]=func
    def contains(self,name):
        return name in self.registered
