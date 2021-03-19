from moz_sql_parser import parse

from mlir import DBType, Attribute, CodeGen
from resolver import StackedResolver, Resolver
from tables import getTPCHTable
from utility import ensure_list, ensure_value_dict, AggrFuncManager, getAttributeList, getPrintNames


class Translator:
    def __init__(self,query):
        self.query=query

    def translateExpression(self,expr, codegen, stacked_resolver):
        def translateSubExpressions(subexprs):
            return list(map(lambda subexpr: self.translateExpression(subexpr, codegen, stacked_resolver), subexprs))

        if type(expr) is dict:
            key = list(expr.keys())[0]
            if key == "and":
                return codegen.create_db_and(translateSubExpressions(expr["and"]))
            elif key in ["lt", "gt", "lte", "gte", "eq"]:
                return codegen.create_db_cmp(key, translateSubExpressions(expr[key]))
            elif key == "date":
                return codegen.create_db_const(expr["date"]["literal"], DBType("date", [], False))
            elif key == "exists":
                subquery, _ = self.translateSelectStmt(expr["exists"], codegen, stacked_resolver)
                return codegen.create_relalg_exists(subquery)
            elif key in ["add", "mul", "sub"]:
                return codegen.create_db_binary_op(key, translateSubExpressions(expr[key]))
            elif key == "interval":
                return codegen.create_db_const("%d %s" % (expr[key][0], expr[key][1]), DBType("interval", [], False))
            else:
                print("unknown expression:", expr)
                return "%unknown"
        elif type(expr) is int:
            return codegen.create_db_const(expr, DBType("int", ["32"], False))
        else:
            attr, tuple = stacked_resolver.resolve(expr)
            return codegen.create_relalg_getattr(tuple, attr)

    def translateSelectStmt(self,stmt, codegen, stacked_resolver=StackedResolver()):
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
                    var =self.addJoinTable(codegen,join_table_desc,resolver)
                    outer= "outer" in key
                    jointype=key.split(" ")[0]
                    tree_var, tuple = codegen.startJoin(outer,jointype,tree_var,var)
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
        AFM = AggrFuncManager()
        if "having" in stmt:
            having_expr = AFM.substituteAggrFuncs(stmt["having"])
        select_names = AFM.substituteComplexExpressions(stmt["select"])
        select_print_names=getPrintNames(stmt["select"])
        print(select_print_names)
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
            tree_var, relation = codegen.startAggregation("agg1", tree_var, attributeList)
            for name, aggrfunc in AFM.aggr.items():
                if len(aggrfunc["names"]) > 0:
                    res = codegen.create_relalg_aggr_func(aggrfunc["type"], resolver.resolve(aggrfunc["names"][0]),
                                                          relation)
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
                results[i].print_name=select_print_names[i]
        return tree_var, results

    def addJoinTable(self, codegen, from_value, resolver):
        if type(from_value["value"]) is str:
            table = getTPCHTable(from_value["value"], from_value["value"])
            prefixes = [from_value["value"]]
            if "name" in from_value:
                prefixes.append(from_value["name"])
            for col_name in table.columns:
                resolver.add(prefixes, col_name, table.columns[col_name])
            var = table.createBaseTableInstr(codegen)
        else:
            var, attrs = self.translateSelectStmt(from_value["value"], codegen)
            prefixes = []
            if "name" in from_value:
                prefixes.append(from_value["name"])
            for attr in attrs:
                resolver.add(prefixes, attr.print_name, attr)
        return var

    def translate(self):
        parsed=parse(self.query)
        codegen = CodeGen()
        codegen.startModule("querymodule")
        codegen.startFunction("query")
        var, results = self.translateSelectStmt(parsed, codegen)
        res = codegen.create_relalg_materialize(var, results)
        codegen.endFunction(res)
        codegen.endModule()
        print(codegen.result)