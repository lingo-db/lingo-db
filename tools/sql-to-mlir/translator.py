from moz_sql_parser import parse

from mlir import DBType, Attribute, CodeGen
from resolver import StackedResolver, Resolver
from tables import getTPCHTable
from utility import ensure_list, ensure_value_dict, AggrFuncManager, getAttributeList, getPrintNames


class Translator:
    def __init__(self, query):
        self.query = query
        self.with_defs = {}
        self.mapnumber=0
        self.aggrnumber=0
    def uniqueMapName(self):
        self.mapnumber+=1
        return "map"+str(self.mapnumber)

    def uniqueAggrName(self):
        self.aggrnumber += 1
        return "aggr" + str(self.aggrnumber)

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
                return codegen.create_db_const(expr["date"]["literal"], DBType("date", [], False))
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
                return_values=[]
                for case in expr[key]:
                    if type(case) is dict and "when" in case:
                        cond=translateSubExpressions(case["when"])[0]
                        return_value=codegen.startIf(cond)
                        then=translateSubExpressions(case["then"])[0]
                        codegen.addElse(then)
                        return_values.append(return_value)
                    else:
                        default=translateSubExpressions(case)[0]
                        return_values.append(default)
                while len(return_values)>1:
                    return_value=return_values.pop()
                    codegen.endIf(return_value)
                return return_values[0]

            elif key == "extract":
                return codegen.create_db_extract(expr[key][0], translateSubExpressions(expr[key][1])[0])
            elif key == "interval":
                return codegen.create_db_const("%d %s" % (expr[key][0], expr[key][1]), DBType("interval", [], False))
            elif key == "literal":
                return codegen.create_db_const(expr[key], DBType("string", [], False))
            elif key == "in":
                attr = expr[key][0]
                vals = expr[key][1]
                if "literal" in vals:
                    return self.translateExpression({"in": [attr, {'select': '*', 'from': {'value': {'values': vals}}}
                                                            ]}, codegen, stacked_resolver)
                elif type(vals)==list:
                    return self.translateExpression({"in": [attr, {'select': '*', 'from': {'value': {'values': {"literal":vals}}}}
                                                            ]}, codegen, stacked_resolver)
                else:
                    subquery, _ = self.translateSelectStmt(vals, codegen, stacked_resolver)
                    return codegen.create_relalg_in(translateSubExpressions(attr)[0],subquery)

            elif key == "between":
                subexpressions = translateSubExpressions(expr[key])
                greaterequal = codegen.create_db_cmp("gte", [subexpressions[0], subexpressions[1]])
                lessequal = codegen.create_db_cmp("lte", [subexpressions[0], subexpressions[2]])
                return codegen.create_db_and([greaterequal, lessequal])

            elif "select" in expr:
                subquery, attr = self.translateSelectStmt(expr, codegen, stacked_resolver)
                return codegen.create_relalg_getscalar(subquery, attr[0])

            else:
                raise Exception("unknown expression:", expr)
        elif type(expr) is int:
            return codegen.create_db_const(expr, DBType("int", ["32"], False))
        elif type(expr) is float:
            return codegen.create_db_const(expr, DBType("float", ["32"], False))
        else:
            attr, tuple = stacked_resolver.resolve(expr)
            return codegen.create_relalg_getattr(tuple, attr)

    def translateSelectStmt(self, stmt, codegen, stacked_resolver=StackedResolver()):
        resolver = Resolver()
        tree_var = ""
        star_attributes = []
        for from_val in ensure_list(stmt["from"]):
            from_value = ensure_value_dict(from_val)
            var = None
            if "value" in from_value:
                var = self.addJoinTable(codegen, from_value, resolver)
                if tree_var == "":
                    tree_var = var
                else:
                    tree_var = codegen.create_relalg_crossproduct(tree_var, var)
            else:
                key = list(from_value.keys())[0]
                if key.endswith("join"):
                    join_table_desc = ensure_value_dict(from_value[key])
                    var = self.addJoinTable(codegen, join_table_desc, resolver)
                    outer = "outer" in key
                    jointype = key.split(" ")[0]
                    tree_var, tuple = codegen.startJoin(outer, jointype, tree_var, var)
                    stacked_resolver.push(tuple, resolver)
                    sel_res = self.translateExpression(from_value["on"], codegen, stacked_resolver)
                    stacked_resolver.pop()
                    codegen.endSelection(sel_res)

        if "where" in stmt:
            tree_var, tuple = codegen.startSelection(tree_var)
            stacked_resolver.push(tuple, resolver)
            sel_res = self.translateExpression(stmt["where"], codegen, stacked_resolver)
            stacked_resolver.pop()
            codegen.endSelection(sel_res)
        AFM = AggrFuncManager(self.uniqueAggrName(),self.uniqueMapName(),self.uniqueMapName())
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
                codegen.create_relalg_addattr(res, attr)
                resolver.add([], name, attr)
            stacked_resolver.pop()
            codegen.endMap()
        if "groupby" in stmt or len(AFM.aggr) > 0:
            scope_name = AFM.aggname
            attributeList = getAttributeList(resolver, stmt["groupby"]) if "groupby" in stmt else []
            tree_var, relation = codegen.startAggregation(scope_name, tree_var, attributeList)
            for name, aggrfunc in AFM.aggr.items():
                if len(aggrfunc["names"]) > 0:
                    if aggrfunc["distinct"] == True:
                        rel=codegen.create_relalg_distinct(relation,[resolver.resolve(aggrfunc["names"][0])])
                    else:
                        rel=relation
                    res = codegen.create_relalg_aggr_func(aggrfunc["type"], resolver.resolve(aggrfunc["names"][0]),
                                                          rel)
                elif aggrfunc["type"] == "count":
                    res = codegen.create_relalg_count_rows(relation)
                attr = Attribute(scope_name, name, codegen.getType(res))
                codegen.create_relalg_addattr(res, attr)
                resolver.add([], name, attr)
            codegen.endAggregation()
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
                codegen.create_relalg_addattr(res, attr)
                resolver.add([], name, attr)
            stacked_resolver.pop()
            codegen.endMap()
        if select_names == ["*"]:
            results = []
        else:
            results = getAttributeList(resolver, select_names)
            for i in range(0, len(results)):
                results[i].print_name = select_print_names[i]
        return tree_var, results

    def addJoinTable(self, codegen, from_value, resolver):
        if type(from_value["value"]) is str and not from_value["value"] in self.with_defs:
            table = getTPCHTable(from_value["value"], from_value["value"])
            prefixes = [from_value["value"]]
            if "name" in from_value:
                prefixes.append(from_value["name"])
            for col_name in table.columns:
                resolver.add(prefixes, col_name, table.columns[col_name])
            var = codegen.create_relalg_base_table(table)
        elif  "values" in from_value["value"]:
            var = codegen.create_relalg_const_relation(from_value["value"]["values"]["literal"])
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
        return var

    def translate(self):
        parsed = parse(self.query)
        #print(parsed)
        if "with" in parsed:
            for with_query in ensure_list(parsed["with"]):
                self.with_defs[with_query["name"]] = with_query["value"]
        codegen = CodeGen()
        codegen.startModule("querymodule")
        codegen.startFunction("query")
        var, results = self.translateSelectStmt(parsed, codegen)
        res = codegen.create_relalg_materialize(var, results)
        codegen.endFunction(res)
        codegen.endModule()
        codegen.fixTypes()
        return codegen.getResult()
