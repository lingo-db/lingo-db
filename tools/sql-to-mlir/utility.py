def ensure_list(maybe_list):
    if type(maybe_list) is list:
        return maybe_list
    else:
        return [maybe_list]


def ensure_value_dict(maybe_dict):
    if type(maybe_dict) is dict:
        return maybe_dict
    else:
        return {"value": maybe_dict}


def getAttributeList(resolver, maybe_list):
    l = ensure_list(maybe_list)
    res = list(map(lambda x: resolver.resolve(ensure_value_dict(x)["value"]), l))
    return res


def getPrintNames(l):
    res=[]
    if type(l) !=list:
        l=[l]
    for expr_ in l:
        expr=ensure_value_dict(expr_)
        if "name" in expr:
            res.append(expr["name"])
        else:
            v=expr["value"]
            if type(v) is dict:
                res.append(list(v.keys())[0])
            else:
                res.append(v)
    return res

class AggrFuncManager:

    def __init__(self,aggname,mapname_before,mapname_after):
        self.aggname=aggname
        self.mapname_before=mapname_before
        self.mapname_after=mapname_after

        self.evaluate_before_agg={}
        self.aggr={}
        self.evaluate_after_agg={}

        self.namectr=0

    def gen_name(self):
        self.namectr+=1
        return "aggfmname"+str(self.namectr)

    def handleAggrFunc(self,t,exprs):
        distinct=False
        names_for_agg=[]
        for expr in exprs:
            if type(expr) is dict and "distinct" in expr:
                distinct = True
                expr = expr["distinct"]["value"]
            if type(expr) is dict:
                name=self.gen_name()
                self.evaluate_before_agg[name]=expr
                names_for_agg.append(name)
            else:
                names_for_agg.append(expr)
        aggr_name=self.gen_name()
        self.aggr[aggr_name]={ "type":t,"distinct":distinct,"names":names_for_agg}
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
