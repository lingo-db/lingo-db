import copy

import pymlirdb
from functools import reduce
import sys
import ast
import types
import ctypes
import inspect
import pprint
from textwrap import dedent

##### test #########
from sql2mlir.codegen import CodeGen
from sql2mlir.mlir import Function, DBType, Attribute, TupleType
from sql2mlir.tables import getTable


def countDelayedOrdersForSupplier(suppkey):
    orders = pymlirdb.read_table("orders")
    orders = orders[orders["o_orderstatus"] == 'F']
    orders["delayed"] = isDelayed(orders["o_orderkey"], suppkey)
    orders = orders[orders["delayed"] == True]
    return orders.count()

def isDelayed(orderkey, suppkey):
    containsDelayed = False
    containsOther = False
    onlyDelayed = True
    items = pymlirdb.read_table("lineitem")
    items = items[items["l_orderkey"] == orderkey]
    for item in items:
        if item["l_suppkey"] == suppkey:
            if item["l_receiptdate"] > item["l_commitdate"]:
                containsDelayed = True
        else:
            containsOther = True
            if item["l_receiptdate"] > item["l_commitdate"]:
                onlyDelayed = False
    return containsDelayed and containsOther and onlyDelayed

##########################################

### == Core Translator ==
class VarResolver:
    def __init__(self):
        self.vars = {}
        pass

    def set(self, var, val):
        self.vars[var] = val

    def resolve(self, var):
        return self.vars[var]
    def canresolve(self,var):
        return var in self.vars

class AttrTupleManager:
    def __init__(self):
        self.cntr=0
        self.ids={}
        self.order=[]
    def register(self,name):
        if not name in self.ids:
            self.ids[name]=self.cntr
            self.cntr+=1
            self.order.append(name)
    def resolve(self,name):
        return self.ids[name]
    def attrs(self):
        return self.order
class BlockAnalyzer(ast.NodeVisitor):

    def __init__(self,dataframes,df_tuples,resolver):
        self.block_vars=set()
        self.dataframes=dataframes
        self.df_tuples=df_tuples
        self.resolver=resolver
    def visit_Module(self, node):
        self.visit(node.body[0])

    def visit_Name(self, node):
        return self.resolver.resolve(node.id)

    def getScopeName(self, base):
        if base in self.basetables:
            self.basetables[base] += 1
            base += str(self.basetables[base])
        else:
            self.basetables[base] = 0
        return base

    def visit_Call(self, node):
        pass


    def visit_Assign(self, node):
        assert len(node.targets) == 1
        t = node.targets[0]
        if type(t) is ast.Name:
            var = t.id
            if self.resolver.canresolve(var):
                self.block_vars.add(var)
            return


    def visit_FunctionDef(self, node):
        for stmt in node.body:
            if type(stmt) is ast.Return:
                self.visit(stmt.value)
            else:
                self.visit(stmt)


    def isDataFrame(self, var):
        return var in self.dataframes
    def isDataFrameTuple(self,var):
        return var in self.df_tuples
    def visit_Subscript(self, node):
        if isinstance(node.ctx, ast.Load):
            if node.slice:
                val = self.visit(node.value)
                slice=node.slice
                if self.isDataFrame(val):
                    if type(slice) is ast.Constant:
                        pass
                    else:
                        sel_res=self.visit(slice)
                if self.isDataFrameTuple(val):
                    if type(slice) is ast.Constant:
                        attr_name=slice.value
                        self.df_tuples[val].register(attr_name)

    def visit_For(self, node):
        for stmt in node.body:
            self.visit(stmt)
    def visit_If(self, node: ast.If):
        self.visit(node.test)
        for stmt in node.body:
            self.visit(stmt)
        if node.orelse:
            for stmt in node.orelse:
                self.visit(stmt)

    def visit_Compare(self,node):
        left=self.visit(node.left)
        right=self.visit(node.comparators[0])
        op=node.ops[0]
    def visit_Constant(self, node:ast.Constant):
        pass
    def visit_BoolOp(self, node: ast.BoolOp):
        values = list(map(self.visit, node.values))


    def generic_visit(self, node):
        raise NotImplementedError(ast.dump(node))
    def visitStatements(self,stmts):
        for stmt in stmts:
            self.visit(stmt)





























class PythonVisitor(ast.NodeVisitor):

    def __init__(self):
        pass

    def __call__(self, source, codegen):
        if isinstance(source, types.ModuleType):
            source = dedent(inspect.getsource(source))
        if isinstance(source, types.FunctionType):
            source = dedent(inspect.getsource(source))
        if isinstance(source, types.LambdaType):
            source = dedent(inspect.getsource(source))
        elif isinstance(source, str):
            source = dedent(source)
        else:
            raise NotImplementedError
        self.dataframes = {}
        self.df_tuples = {}
        self._source = source
        self.codegen = codegen
        self.functions=self.codegen.functions
        self._ast = ast.parse(source)
        return self.visit(self._ast)

    def visit_Module(self, node):
        self.visit(node.body[0])

    def visit_Name(self, node):
        return self.resolver.resolve(node.id)

    def visit_Call(self, node):
        if type(node.func) is ast.Attribute:
            if type(node.func.value) is ast.Name and node.func.value.id == "pymlirdb":
                pymlirdb_func = node.func.attr
                if pymlirdb_func == "read_table":
                    table_name = node.args[0].value
                    scope_name = self.codegen.getUniqueName(table_name)
                    table=getTable(table_name,scope_name)
                    var = self.codegen.create_relalg_base_table(table)
                    self.dataframes[var] = table.columns
                    return var
            objval = self.visit(node.func.value)
            if self.isDataFrame(objval):
                method_name = node.func.attr
                if method_name == "count":
                    return self.codegen.create_relalg_count_rows(objval)
                if method_name == "join":
                    otherdf=self.visit(node.args[0])
                    on=node.keywords[0].value

                    tree_var, tuple = self.codegen.startJoin(False,"innerjoin",objval,otherdf)
                    self.curr_tuple = tuple
                    sel_res = self.visit(on)
                    self.codegen.endJoin(sel_res)
                    self.dataframes[tree_var] = self.dataframes[objval].copy()
                    self.dataframes[tree_var].update(self.dataframes[otherdf])
                    return tree_var
        if type(node.func) is ast.Name:
            if self.functions.contains(node.func.id):
                params=list(map(lambda x:self.visit(x),node.args))
                return self.codegen.create_db_func_call(node.func.id,params)
                #return self.codegen.create
        raise NotImplementedError



    def visit_Assign(self, node):
        assert len(node.targets) == 1
        t = node.targets[0]
        if type(t) is ast.Name:
            var = t.id
            val = self.visit(node.value)
            self.resolver.set(var, val)
            return
        elif type(t) is ast.Subscript:
            if t.slice:
                left_val = self.visit(t.value)
                slice=t.slice
                if self.isDataFrame(left_val):
                    map_attr_name=slice.value
                    scope_name=self.codegen.getUniqueName("map")
                    tree_var, tuple = self.codegen.startMap(scope_name, left_val)
                    self.curr_tuple = tuple
                    val=self.visit(node.value)
                    map_attr_type=self.codegen.getType(val)
                    attr=Attribute(scope_name,map_attr_name,map_attr_type)
                    tuple=self.codegen.create_relalg_addattr(val,attr,tuple)
                    self.codegen.endMap(tuple)
                    self.dataframes[tree_var] = self.dataframes[left_val].copy()
                    self.dataframes[tree_var][map_attr_name]=attr
                    if type(t.value) is ast.Name:
                        self.resolver.set(t.value.id,tree_var)
                    return tree_var
            pass
        raise NotImplementedError

    def visit_FunctionDef(self, node):
        self.resolver = VarResolver()
        func = self.functions.get(node.name)
        func_params = []
        idx = 0
        for argobj in node.args.args:
            argname = argobj.arg
            p = self.codegen.newParam(func.operandTypes[idx])
            func_params.append(p + ": " + func.operandTypes[idx].to_string())
            idx += 1
            self.resolver.set(argname, p)
        self.codegen.startFunction(node.name, func_params)
        ret = None
        for stmt in node.body:
            if type(stmt) is ast.Return:
                ret = self.visit(stmt.value)
            else:
                self.visit(stmt)
        self.codegen.endFunction(ret)


    def isDataFrame(self, var):
        return var in self.dataframes
    def isDataFrameTuple(self,var):
        return var in self.df_tuples
    def visit_Subscript(self, node):
        if isinstance(node.ctx, ast.Load):
            if node.slice:
                val = self.visit(node.value)
                slice = node.slice
                if self.isDataFrame(val):
                    if type(slice) is ast.Constant:
                        attr_name=slice.value
                        attr=self.dataframes[val][attr_name]

                        var= self.codegen.create_relalg_getattr(self.curr_tuple,attr)
                        return var
                    else:
                        tree_var, tuple = self.codegen.startSelection(val)
                        self.curr_tuple=tuple
                        sel_res=self.visit(slice)
                        self.codegen.endSelection(sel_res)
                        self.dataframes[tree_var] = self.dataframes[val].copy()
                        return tree_var
                if self.isDataFrameTuple(val):
                    if type(slice) is ast.Constant:
                        attr_name=slice.value
                        idx =self.df_tuples[val].resolve(attr_name)
                        var= self.codegen.create_get_tuple(val,idx)
                        return var
        raise NotImplementedError

    def visit_For(self, node):
        if type(node.target) is ast.Name:
            param_name=node.target.id
            iter = self.visit(node.iter)
            param_type = TupleType([])
            iterateDF=False
            param=None
            if self.isDataFrame(iter):
                param= self.codegen.newParam(param_type)
                self.df_tuples[param]=AttrTupleManager()
                ba =BlockAnalyzer(self.dataframes,self.df_tuples,self.resolver)
                self.resolver.set(param_name,param)
                ba.visitStatements(node.body)
                iter_args=[]
                iter_args_initial=[]
                iter_vars=list(ba.block_vars)
                for x in iter_vars:
                    resolved=self.resolver.resolve(x)
                    iter_args.append(self.codegen.newParam(self.codegen.getType(resolved)))
                    iter_args_initial.append(resolved)


                attrs=list(map(lambda x:self.dataframes[iter][x],self.df_tuples[param].attrs()))
                for attr in attrs:
                    param_type.types.append(attr.type)
                iter=self.codegen.create_relalg_getlist(iter,attrs)
                self.codegen.startFor(param,iter,iter_args,iter_args_initial)
                for x in zip(iter_vars,iter_args):
                    self.resolver.set(x[0],x[1])
                for stmt in node.body:
                    self.visit(stmt)
                yield_values=list(map(lambda x:self.resolver.resolve(x),iter_vars))
                for_results=self.codegen.endFor(yield_values)
                for (x,y) in zip(iter_vars,for_results):
                    self.resolver.set(x,y)
            return
        raise NotImplementedError
    def visit_If(self, node: ast.If):
        condition=self.visit(node.test)
        self.codegen.startIf(condition)
        ba = BlockAnalyzer(self.dataframes, self.df_tuples, self.resolver)
        ba.visitStatements(node.body)
        if node.orelse:
            ba.visitStatements(node.orelse)
        iter_vars = list(ba.block_vars)
        captured_resolver=self.resolver
        self.resolver=copy.deepcopy(captured_resolver)
        for stmt in node.body:
            self.visit(stmt)
        yield_values = list(map(lambda x: self.resolver.resolve(x), iter_vars))
        self.codegen.addElse(yield_values)
        self.resolver=copy.deepcopy(captured_resolver)
        if node.orelse:
            for stmt in node.orelse:
                self.visit(stmt)
        yield_values = list(map(lambda x: self.resolver.resolve(x), iter_vars))
        if_results = self.codegen.endIf(yield_values)
        self.resolver=captured_resolver
        for (x, y) in zip(iter_vars, if_results):
            self.resolver.set(x, y)
        return

        raise NotImplementedError

    def visit_Compare(self,node):
        left=self.visit(node.left)
        right=self.visit(node.comparators[0])
        op=node.ops[0]
        strop=None
        if type(op) is ast.Eq:
            strop="eq"
        elif type(op) is ast.Gt:
            strop="gt"
        else:
            raise NotImplementedError(ast.dump(node))
        return self.codegen.create_db_cmp(strop,[left,right])
    def visit_Constant(self, node:ast.Constant):
        if type(node.value) is str:
            return self.codegen.create_db_const(node.value,DBType("string"))
        if type(node.value) is bool:
            return self.codegen.create_db_const(1 if node.value else 0,DBType("bool"))
        raise NotImplementedError(ast.dump(node))
    def visit_BoolOp(self, node: ast.BoolOp):
        if type(node.op) is ast.And:
            values = list(map(self.visit, node.values))
            return self.codegen.create_db_and(values)
        if type(node.op) is ast.Or:
            values = list(map(self.visit, node.values))
            return self.codegen.create_db_or(values)
        raise NotImplementedError(ast.dump(node))


    def generic_visit(self, node):
        raise NotImplementedError(ast.dump(node))

def compileFn(func,codegen):
        transformer = PythonVisitor()
        transformer(func, codegen)




