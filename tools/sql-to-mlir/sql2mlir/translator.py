from moz_sql_parser import parse
import copy
from sql2mlir.mlir import DBType, Attribute
from sql2mlir.codegen import CodeGen
from sql2mlir.resolver import StackedResolver, Resolver
from sql2mlir.tables import getTable
from sql2mlir.utility import ensure_list, ensure_value_dict, AggrFuncManager, getAttributeList, getPrintNames


class Translator:
    def __init__(self, query):
        self.params={}
        self.query = query
        self.basetables = {}
        self.with_defs = {}


    def translateExpression(self, expr, codegen, stacked_resolver):
        def translateSubExpressions(subexprs):
            return list(map(lambda subexpr: self.translateExpression(subexpr, codegen, stacked_resolver),
                            ensure_list(subexprs)))

        if type(expr) is dict:
            key = list(expr.keys())[0]
            if key == "and":
                return codegen.create_db_and(translateSubExpressions(expr["and"]))
            if key == "or":
                return codegen.create_db_or(translateSubExpressions(expr["or"]))
            if key == "not":
                return codegen.create_db_not(translateSubExpressions(expr[key]))
            elif key in ["lt", "gt", "lte", "gte", "eq", "neq","like"]:
                return codegen.create_db_cmp(key, translateSubExpressions(expr[key]))
            elif key == "date":
                return codegen.create_db_const(expr["date"]["literal"], DBType("date", ["day"], False))
            elif key == "exists":
                subquery, _ = self.translateSelectStmt(expr["exists"], codegen, stacked_resolver)
                return codegen.create_relalg_exists(subquery)
            elif key == "not_like":
                return self.translateExpression({"not": {"like": expr[key]}}, codegen, stacked_resolver)
            elif key == "nin":
                return self.translateExpression({"not": {"in": expr[key]}}, codegen, stacked_resolver)
            elif key in ["add", "mul", "sub", "div"]:
                subexprs=translateSubExpressions(expr[key])
                if codegen.getType(subexprs[0]).name =="date":
                    return codegen.create_db_date_binary_op(key, subexprs )
                else:
                    return codegen.create_db_binary_op(key, subexprs )
            elif key == "case":
                nesting_depth=0
                return_value=None
                for case in expr[key]:
                    if type(case) is dict and "when" in case:
                        cond=translateSubExpressions(case["when"])[0]
                        codegen.startIf(cond)
                        then=translateSubExpressions(case["then"])[0]
                        codegen.addElse([then])
                        nesting_depth+=1
                    else:
                        default=translateSubExpressions(case)[0]
                        codegen.toCommonType(default,codegen.getType(then))
                        return_value=default
                for i in range(0,nesting_depth):
                    return_value=codegen.endIf([return_value])[0]
                return return_value

            elif key == "extract":
                return codegen.create_db_extract(expr[key][0], translateSubExpressions(expr[key][1])[0])
            elif key == "interval":
                val=int(expr[key][0])
                tu=expr[key][1]
                timeunit=""
                if tu=="days" or tu=="day":
                    val*=24*60*60*1000
                    timeunit="daytime"
                else:
                    timeunit="months"
                return codegen.create_db_const(val, DBType("interval", [timeunit], False))
            elif key == "literal":
                if len(expr[key])<8:
                    return codegen.create_db_const(expr[key], DBType("char", [str(len(expr[key]))], False))
                return codegen.create_db_const(expr[key], DBType("string", [], False))
            elif key == "in":
                attr = expr[key][0]
                vals = expr[key][1]
                query=vals
                immediate=False
                if "literal" in query:
                    query=query["literal"]
                if type(query)==list:
                    if(len(query)<10):
                        left=translateSubExpressions(attr)[0]
                        comparisons=[]
                        for val in query:
                            right=codegen.create_db_const(val, codegen.getType(left))
                            comparisons.append(codegen.create_db_cmp("eq",[left,right]))
                        return codegen.create_db_or(comparisons)
                    else:
                        query={'select': '*', 'from': {'value': {'values': {"literal":vals}}}}
                        immediate=True
                rel, attrs = self.translateSelectStmt(query, codegen, stacked_resolver)
                if not immediate:
                    rel=codegen.create_relalg_projection("all",rel,attrs)
                return codegen.create_relalg_in(translateSubExpressions(attr)[0],rel)

            elif key == "between":
                subexpressions = translateSubExpressions(expr[key])
                greaterequal = codegen.create_db_cmp("gte", [subexpressions[0], subexpressions[1]])
                lessequal = codegen.create_db_cmp("lte", [subexpressions[0], subexpressions[2]])
                return codegen.create_db_and([greaterequal, lessequal])

            elif "select" in expr:
                subquery, attr = self.translateSelectStmt(expr, codegen, stacked_resolver)
                return codegen.create_relalg_getscalar(subquery, attr[0])

            else:
                funcname=key
                params=translateSubExpressions(expr[key])
                return codegen.create_db_func_call(funcname,params)
                #raise Exception("unknown expression:", expr)
        elif type(expr) is int:
            return codegen.create_db_const(expr, DBType("int", ["64"], False))
        elif type(expr) is float:
            asstring=str(expr)
            parts=asstring.split('.');
            afterComma=len(parts[1])
            return codegen.create_db_const(expr, DBType("decimal", ["15",str(afterComma)], False))
        elif type(expr) is str and expr.startswith("@"):
            return self.params[expr[1:len(expr)]]
        else:
            attr, tuple = stacked_resolver.resolve(expr)
            return codegen.create_relalg_getattr(tuple, attr)

    def translateSelectStmt(self, stmt, codegen, stacked_resolver=StackedResolver()):
        resolver = Resolver()
        tree_var = ""
        all_from_attributes = []
        for from_val in ensure_list(stmt["from"]):
            from_value = ensure_value_dict(from_val)
            var = None
            if "value" in from_value:
                var = self.addJoinTable(codegen, from_value, resolver,all_from_attributes)
                if tree_var == "":
                    tree_var = var
                else:
                    tree_var = codegen.create_relalg_crossproduct(tree_var, var)
            else:
                key = list(from_value.keys())[0]
                if key.endswith("join"):
                    join_table_desc = ensure_value_dict(from_value[key])
                    left_attributes=all_from_attributes
                    right_attributes=[]
                    var = self.addJoinTable(codegen, join_table_desc, resolver,right_attributes)
                    all_from_attributes.extend(right_attributes)
                    outer = "outer" in key
                    left=tree_var
                    right=var
                    jointype = key.split(" ")[0]
                    if jointype == "left":
                        jointype=""
                    elif jointype == "right":
                        jointype=""
                        left,right=right,left
                    name=codegen.getUniqueName("outerjoin") if outer else ""
                    tree_var, tuple = codegen.startJoin(outer, jointype, left , right, name)

                    stacked_resolver.push(tuple, resolver)
                    sel_res = self.translateExpression(from_value["on"], codegen, stacked_resolver)
                    stacked_resolver.pop()
                    mapping=[]
                    if outer:
                        for attr in right_attributes:
                            new_type=copy.copy(attr.type)
                            new_type.nullable=True
                            new_attr=Attribute(name,attr.name,new_type,[attr],attr.print_name)
                            mapping.append(new_attr)
                            resolver.replace(attr,new_attr)
                    codegen.endJoin(sel_res,mapping)

        if "where" in stmt:
            tree_var, tuple = codegen.startSelection(tree_var)
            stacked_resolver.push(tuple, resolver)
            sel_res = self.translateExpression(stmt["where"], codegen, stacked_resolver)
            stacked_resolver.pop()
            codegen.endSelection(sel_res)
        AFM = AggrFuncManager(codegen.getUniqueName("aggr"),codegen.getUniqueName("map"),codegen.getUniqueName("map"))
        if "having" in stmt:
            having_expr = AFM.substituteAggrFuncs(stmt["having"])
        select_names = AFM.substituteComplexExpressions(stmt["select"])
        select_print_names = getPrintNames(stmt["select"])
        if len(AFM.evaluate_before_agg) > 0:
            scope_name = AFM.mapname_before
            tree_var, tuple = codegen.startMap(scope_name, tree_var)
            stacked_resolver.push(tuple, resolver)
            for name, expr in AFM.evaluate_before_agg.items():
                res = self.translateExpression(expr, codegen, stacked_resolver)
                attr = Attribute(scope_name, name, codegen.getType(res))
                tuple=codegen.create_relalg_addattr(res, attr,tuple)
                resolver.add([], name, attr)
            stacked_resolver.pop()
            codegen.endMap(tuple)
        if "groupby" in stmt or len(AFM.aggr) > 0:
            scope_name = AFM.aggname
            attributeList = getAttributeList(resolver, stmt["groupby"]) if "groupby" in stmt else []
            tree_var, relation,tuple = codegen.startAggregation(scope_name, tree_var, attributeList)
            for name, aggrfunc in AFM.aggr.items():
                if len(aggrfunc["names"]) > 0:
                    if aggrfunc["distinct"] == True:
                        rel=codegen.create_relalg_projection("distinct",relation,[resolver.resolve(aggrfunc["names"][0])])
                    else:
                        rel=relation
                    aggr_is_nullable=False
                    if aggrfunc["type"]!="count":
                        if len(attributeList)==0:
                            aggr_is_nullable=True
                        else:
                            aggr_is_nullable=resolver.resolve(aggrfunc["names"][0]).type.nullable
                    res = codegen.create_relalg_aggr_func(aggrfunc["type"], resolver.resolve(aggrfunc["names"][0]),
                                                          rel,aggr_is_nullable)
                elif aggrfunc["type"] == "count":
                    res = codegen.create_relalg_count_rows(relation)
                attr = Attribute(scope_name, name, codegen.getType(res))
                tuple=codegen.create_relalg_addattr(res, attr,tuple)
                resolver.add([], name, attr)
            codegen.endAggregation(tuple)
        if "having" in stmt:
            tree_var, tuple = codegen.startSelection(tree_var)
            stacked_resolver.push(tuple, resolver)
            sel_res = self.translateExpression(having_expr, codegen, stacked_resolver)
            stacked_resolver.pop()
            codegen.endSelection(sel_res)
        if len(AFM.evaluate_after_agg) > 0:
            scope_name = AFM.mapname_after
            tree_var, tuple = codegen.startMap(scope_name, tree_var)
            stacked_resolver.push(tuple, resolver)
            for name, expr in AFM.evaluate_after_agg.items():
                res = self.translateExpression(expr, codegen, stacked_resolver)
                attr = Attribute(scope_name, name, codegen.getType(res))
                tuple=codegen.create_relalg_addattr(res, attr,tuple)
                resolver.add([], name, attr)
            stacked_resolver.pop()
            codegen.endMap(tuple)
        if select_names == ["*"]:
            results = all_from_attributes
        else:
            results = getAttributeList(resolver, select_names)
            for i in range(0, len(results)):
                results[i].print_name = select_print_names[i]
                resolver.addOverride(select_print_names[i],results[i])
        if "orderby" in stmt:
            sortSpecifications=[]
            for v in ensure_list(stmt["orderby"]):
                attr=resolver.resolve(v["value"])
                sortSpec=v["sort"] if "sort" in v else "asc"
                sortSpecifications.append((attr,sortSpec))
            tree_var=codegen.create_relalg_sort(tree_var,sortSpecifications)
        if "limit" in stmt:
            tree_var=codegen.create_relalg_limit(tree_var,stmt["limit"])

        return tree_var, results

    def estimateType(self,val):
        return DBType("string") if type(val) == str else DBType("int", ["32"])
    def addJoinTable(self, codegen, from_value, resolver,all_from_attributes):
        if type(from_value["value"]) is str and not from_value["value"] in self.with_defs:
            scope_name=codegen.getUniqueName(from_value["value"])


            table = getTable(from_value["value"],scope_name)
            prefixes = [from_value["value"]]
            if "name" in from_value:
                prefixes.append(from_value["name"])
            for col_name in table.columns:
                table.columns[col_name].print_name=col_name
                resolver.add(prefixes, col_name, table.columns[col_name])
                all_from_attributes.append(table.columns[col_name]);
            var = codegen.create_relalg_base_table(table)
        elif  "values" in from_value["value"]:
            attrs= []
            base =from_value["value"]["values"]
            base =base["literal"] if (type(base)==dict) and "literal" in base else base
            probeelement=base[0]
            scope_name=codegen.getUniqueName("constrel")
            if type(probeelement)==list:
                for val in probeelement:
                    attrname = codegen.getUniqueName("attr")
                    attr = Attribute(scope_name, attrname, self.estimateType(val), [], attrname)
                    attrs.append(attr)
            else:
                attrname = codegen.getUniqueName("attr")
                attr=Attribute(scope_name,attrname,self.estimateType(probeelement),[],attrname)
                attrs.append(attr)
            var = codegen.create_relalg_const_relation(scope_name,attrs,base)
            if "name" in from_value:
                prefixes=list(from_value['name'].keys())
                column_names = from_value['name'][prefixes[0]]
                i=0
                for attr in attrs:
                    attr.print_name=column_names[i]
                    resolver.add(prefixes, column_names[i], attr)
                    i+=1
            for attr in attrs:
                all_from_attributes.append(attr)
        else:
            if type(from_value["value"]) is str and from_value["value"] in self.with_defs:
                with_def = self.with_defs[from_value["value"]]
                from_value = {"value": with_def, "name": from_value["value"]}
            var, attrs = self.translateSelectStmt(from_value["value"], codegen)
            prefixes = []
            if "name" in from_value:
                prefixes.append(from_value["name"])
            for attr in attrs:
                resolver.add(prefixes, attr.print_name, attr)
                all_from_attributes.append(attr)
        return var
    def setParam(self,name,val):
        self.params[name]=val
    def translate(self,codegen):
        parsed = parse(self.query)
        if "with" in parsed:
            for with_query in ensure_list(parsed["with"]):
                self.with_defs[with_query["name"]] = with_query["value"]
        var, results = self.translateSelectStmt(parsed, codegen)
        res = codegen.create_relalg_materialize(var, results)
        return res
    def translateIntoFunction(self,codegen,name, params,mode="private"):
        func_params=[]
        for param_name in params:
            p=codegen.newParam(params[param_name])
            func_params.append(p+": "+params[param_name].to_string())
            self.setParam(param_name,p)
        codegen.startFunction(name,func_params,mode)
        res= self.translate(codegen)
        codegen.endFunction(res)
    def translateModule(self,params,functions={}):

        codegen = CodeGen(functions)
        codegen.startModule("querymodule")
        self.translateIntoFunction(codegen,"main",params,"")
        codegen.endModule()
        return codegen.getResult()
