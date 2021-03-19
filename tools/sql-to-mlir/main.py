from moz_sql_parser import parse
from tables import getTPCHTable
from mlir import CodeGen, DBType, Attribute
for query_number in range(1,23):
    file="./sql/tpch/"+str(query_number)+".sql"
    print(file)
    query=""
    with open(file) as f:
         query = f.read()
         f.close()

    parsed=parse(query)

    def ensure_list(maybe_list):
        if type(maybe_list) is list:
            return maybe_list
        else:
            return [maybe_list]

    def ensure_value_dict(maybe_dict):
        if type(maybe_dict) is dict:
            return maybe_dict
        else:
            return {"value":maybe_dict}

    class Resolver:
        def __init__(self):
            self.lookup_table={}
        def resolve(self,name):
            if name in self.lookup_table:
                return self.lookup_table[name]
            else:
                raise Exception("could not resolve"+ name)
        def add_(self,name,attr):
            if name in self.lookup_table:
                del self.lookup_table[name]
            else:
                self.lookup_table[name]=attr


        def add(self,prefixes,name,attr):
            self.add_(name, attr)
            for prefix in prefixes:
               self.add_(prefix+"."+name,attr)



    class StackedResolver:
        def __init__(self):
            self.stack= []
        def push(self,tuple,resolver=Resolver()):
            self.stack.append((tuple,resolver))
        def pop(self):
            self.stack.pop()
        def resolve(self,name):
            resolved=None
            responsible_tuple=""
            for tuple,resolver in reversed(self.stack):
                try:
                    resolved=resolver.resolve(name)
                    responsible_tuple=tuple
                    break
                except:
                    continue
            if resolved==None:
                raise Exception("could not resolve"+ name)
            else:
                return resolved,responsible_tuple
        def add(self,prefixes,name,attr):
            self.stack[-1].add(prefixes,name,attr)


    class AggrFuncManager:

        def __init__(self):
            self.aggname="agg1"
            self.mapname_before="map1"
            self.mapname_after="map2"

            self.evaluate_before_agg={}
            self.aggr={}
            self.evaluate_after_agg={}

            self.namectr=0

        def gen_name(self):
            self.namectr+=1
            return "aggfmname"+str(self.namectr)

        def handleAggrFunc(self,t,exprs):
            names_for_agg=[]
            for expr in exprs:
                if type(expr) is dict:
                    name=self.gen_name()
                    self.evaluate_before_agg[name]=expr
                    names_for_agg.append(name)
                else:
                    names_for_agg.append(expr)
            aggr_name=self.gen_name()
            self.aggr[aggr_name]={ "type":t, "names":names_for_agg}
            return aggr_name
        def substituteAggrFuncs(self, obj):
            if type(obj) is list:
                res=[]
                for el in obj:
                    res.append(self.substituteAggrFuncs(el))
                return res
            elif type(obj) is dict:
                t=list(obj.keys())[0]
                v=obj[t]
                if t=="select":
                    return obj
                if t in ["count","min","max","sum","avg"]:
                    exprs=[]
                    if v!='*':
                        exprs.append(v)
                    return self.handleAggrFunc(t,exprs)
                return {t:self.substituteAggrFuncs(v)}
            else:
                return obj
        def substituteComplexExpressions(self, exprs):
            exprs=ensure_list(exprs)
            result = []
            for expr in exprs:
                expr_without_aggr=ensure_value_dict(self.substituteAggrFuncs(expr))["value"]
                if type(expr_without_aggr) is dict:
                    name = self.gen_name()
                    self.evaluate_after_agg[name] = expr_without_aggr
                    result.append(name)
                else:
                    result.append(expr_without_aggr)
            return result



    def translateExpression(expr,codegen,stacked_resolver):
        def translateSubExpressions(subexprs):
            return list(map(lambda subexpr: translateExpression(subexpr, codegen, stacked_resolver), subexprs))
        if type(expr) is dict:
            key=list(expr.keys())[0]
            if key=="and":
                return codegen.create_db_and(translateSubExpressions(expr["and"]))
            elif key in ["lt","gt","lte","gte","eq"]:
                return codegen.create_db_cmp(key,translateSubExpressions(expr[key]))
            elif key=="date":
                return codegen.create_db_const(expr["date"]["literal"],DBType("date",[],False))
            elif key=="exists":
                subquery,_=translateSelectStmt(expr["exists"],codegen,stacked_resolver)
                return codegen.create_relalg_exists(subquery)
            elif key in ["add","mul","sub"]:
                return codegen.create_db_binary_op(key,translateSubExpressions(expr[key]))
            elif key=="interval":
                return codegen.create_db_const("%d %s" % (expr[key][0],expr[key][1]),DBType("interval",[],False))
            else:
                print("unknown expression:",expr)
                return "%unknown"
        elif type(expr) is int:
            return codegen.create_db_const(expr, DBType("int", ["32"], False))
        else:
            attr,tuple=stacked_resolver.resolve(expr)
            return codegen.create_relalg_getattr(tuple,attr)

    def getAttributeList(resolver,maybe_list):
        l=ensure_list(maybe_list)
        res=list(map(lambda x: resolver.resolve(ensure_value_dict(x)["value"]),l))
        return res

    def translateSelectStmt(stmt,codegen,stacked_resolver=StackedResolver()):

        resolver=Resolver()
        tree_var = ""
        star_attributes=[]
        for from_val in ensure_list(stmt["from"]):
            from_value = ensure_value_dict(from_val)
            var = None
            if type(from_value["value"]) is str:
                table = getTPCHTable(from_value["value"], from_value["value"])
                prefixes = [from_value["value"]]
                if "name" in from_value:
                    prefixes.append(from_value["name"])
                for col_name in table.columns:
                    resolver.add(prefixes, col_name, table.columns[col_name])
                var = table.createBaseTableInstr(codegen)
            else:
                var = translateSelectStmt(from_value["value"],codegen)

            if tree_var == "":
                tree_var = var
            else:
                tree_var = codegen.create_relalg_crossproduct(tree_var, var)

        if "where" in stmt:
            tree_var,tuple=codegen.startSelection(tree_var)
            stacked_resolver.push(tuple,resolver)
            sel_res=translateExpression(stmt["where"],codegen,stacked_resolver)
            stacked_resolver.pop()
            codegen.endSelection(sel_res)
        AFM=AggrFuncManager()
        if "having" in stmt:
            having_expr=AFM.substituteAggrFuncs(stmt["having"])
        select_names=AFM.substituteComplexExpressions(stmt["select"])
        if len(AFM.evaluate_before_agg)>0:
            scope_name=AFM.mapname_before
            tree_var,tuple=codegen.startMap(scope_name,tree_var)
            stacked_resolver.push(tuple,resolver)
            for name,expr in AFM.evaluate_before_agg.items():
                res=translateExpression(expr,codegen,stacked_resolver)
                attr=Attribute(scope_name,name,codegen.getType(res))
                codegen.create_relalg_addattr(res,attr)
                resolver.add([],name,attr)
            stacked_resolver.pop()
            codegen.endMap()
        if "groupby" in stmt or len(AFM.aggr)>0:
            scope_name=AFM.aggname
            attributeList=getAttributeList(resolver,stmt["groupby"]) if "groupby" in stmt else []
            tree_var,relation=codegen.startAggregation("agg1",tree_var,attributeList)
            for name,aggrfunc in AFM.aggr.items():
                if len(aggrfunc["names"]) > 0:
                    res=codegen.create_relalg_aggr_func(aggrfunc["type"],resolver.resolve(aggrfunc["names"][0]),relation)
                elif aggrfunc["type"]=="count":
                    res=codegen.create_relalg_count_rows(relation)
                attr=Attribute(scope_name,name,codegen.getType(res))
                codegen.create_relalg_addattr(res,attr)
                resolver.add([],name,attr)
            codegen.endAggregation()
        if "having" in stmt:
            tree_var,tuple=codegen.startSelection(tree_var)
            stacked_resolver.push(tuple,resolver)
            sel_res=translateExpression(having_expr,codegen,stacked_resolver)
            stacked_resolver.pop()
            codegen.endSelection(sel_res)
        if len(AFM.evaluate_after_agg)>0:
            scope_name=AFM.mapname_after
            tree_var,tuple=codegen.startMap(scope_name,tree_var)
            stacked_resolver.push(tuple,resolver)
            for name,expr in AFM.evaluate_after_agg.items():
                res=translateExpression(expr,codegen,stacked_resolver)
                attr=Attribute(scope_name,name,codegen.getType(res))
                codegen.create_relalg_addattr(res,attr)
                resolver.add([],name,attr)
            stacked_resolver.pop()
            codegen.endMap()
        if select_names==["*"]:
            results=[]
        else:
            results=getAttributeList(resolver,select_names)
        return tree_var,results

    codegen=CodeGen()
    codegen.startModule("querymodule")
    codegen.startFunction("query")
    var,results=translateSelectStmt(parsed,codegen)
    res=codegen.create_relalg_materialize(var,results)
    codegen.endFunction(res)
    codegen.endModule()
    print(codegen.result)
