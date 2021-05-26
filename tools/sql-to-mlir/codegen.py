import functools
from mlir import DBType,Attribute


class ValueRef:
    def __init__(self,var):
        self.var=var
    def print(self,codegen):
        codegen.print(str(self.var))
class TypeRef:
    def __init__(self,var):
        self.var=var
    def print(self,codegen):
        codegen.print(codegen.getType(self.var).to_string())
class Operation:
    def __init__(self,var,type, print_args):
        self.var=var
        self.type=type
        self.print_args=print_args
    def print(self,codegen):
        if self.var != None:
            codegen.print(self.var + " = ")
        for print_arg in self.print_args:
            if type(print_arg) is str:
                codegen.print(print_arg)
            elif type(print_arg) is int or type(print_arg) is float:
                codegen.print(str(print_arg))
            else:
                print_arg.print(codegen)
        codegen.print("\n")


class BaseTableOp:
    def __init__(self,var,table):
        self.table=table
        self.var=var
        self.type="relation"

    def print(self, codegen):
        column_str=""
        first=True
        for column_name,attr in self.table.columns.items():
            if first:
                first=False
            else:
                column_str+=",\n"
            column_str+=column_name+" => "+attr.def_to_string();
        str="%s = relalg.basetable @%s { table_identifier=\"%s\" } columns: {" % (self.var,self.table.scope_name,self.table.table_identifier)
        codegen.print(str)
        codegen.indent+=1
        codegen.print(column_str)
        codegen.indent-=1
        codegen.print("\n}\n")

class Param:
    def __init__(self,type):
        self.type=type

class Region:
    def __init__(self):
        self.ops=[]
    def addOp(self,op):
        self.ops.append(op)
    def print(self,codegen):
        codegen.print("{")
        codegen.print("\n")
        codegen.indent+=1
        for op in self.ops:
            op.print(codegen)
        codegen.indent-=1
        codegen.print("}")
class RootRegion:
    def __init__(self,ops):
        self.ops=ops
    def addOp(self, op):
        self.ops.append(op)
    def print(self,codegen):
        for op in self.ops:
            op.print(codegen)
class CodeGen:


    def __init__(self):
        self.var_number = 0
        self.needs_indent=False
        self.indent=0
        self.result =""
        self.ops ={}
        self.stacked_regions=[RootRegion([])]
        self.stacked_ops=[]

    def getType(self, var):
        type = self.ops[var].type
        return type
    def getCurrentRegion(self):
        return self.stacked_regions[-1]
    def newParam(self,type):
        self.var_number+=1
        var_str='%%%d' % (self.var_number)
        self.ops[var_str]=Param(type)
        return var_str
    def newVar(self):
        self.var_number+=1
        var_str='%%%d' % (self.var_number)
        return var_str

    def print(self,str):
        for c in str:
            if c=="\n":
                self.result+="\n"
                self.needs_indent=True
            else:
                if self.needs_indent:
                    self.needs_indent=False
                    self.result+=("    " * self.indent)
                self.result+=c




    def create_relalg_crossproduct(self,left,right):
        return self.addOp("relation", ["relalg.crossproduct ",ValueRef(left),", ",ValueRef(right)])
    def create_relalg_base_table(self,table):
        var=self.newVar()
        self.addExistingOp(BaseTableOp(var,table))
        return var
    def startRegion(self):
        self.stacked_regions.append(Region())
    def endRegion(self):
        return self.stacked_regions.pop()

    def is_any_nullable(self,values):
        return any(self.getType(x).nullable for x in values)

    def create_db_and(self, values):
        args = ["db.and "]
        self.addValuesWithTypes(args, values)
        return self.addOp(DBType("bool", [], self.is_any_nullable(values)), args)
    def create_db_or(self, values):
        args = ["db.or "]
        self.addValuesWithTypes(args, values)
        return self.addOp(DBType("bool", [], self.is_any_nullable(values)), args)
    def create_db_not(self, values):
        args = ["db.not "]
        self.addValuesWithTypes(args, values)
        return self.addOp(DBType("bool", [], self.is_any_nullable(values)), args)
    def create_db_cmp(self, type,values):
        args=["db.compare ",type," "]
        self.addValuesWithTypes(args,values)
        return self.addOp(DBType("bool",[],self.is_any_nullable(values)),args)
    def addValuesWithTypes(self,arr,values):
        first=True
        for v in values:
            if first:
                first=False
            else:
                arr.append(",")
            arr.append(ValueRef(v))
            arr.append(" : ")
            arr.append(TypeRef(v))

    def create_db_binary_op(self, name,values:list):
        bin_values=[values[0],values[1]]
        args=["db."+name+" "]
        common_type=self.getCommonType(self.getTypesForValues(bin_values))
        self.addValuesWithTypes(args,self.toCommonTypes(bin_values,common_type))
        res= self.addOp(common_type,args)
        if len(values)>2:
            rec_vals=values[2:]
            rec_vals.append(res)
            return self.create_db_binary_op(name,rec_vals)
        else:
            return res
    def create_db_date_binary_op(self, name,values):
        args=["db.date_"+name+" "]
        self.addValuesWithTypes(args,values)
        return self.addOp(DBType("date",["day"]),args)
    def create_db_extract(self, key,value):
        return self.addOp(DBType("int",["64"],self.getType(value).nullable),["db.date_extract"," ",key,", ",ValueRef(value)," : ",TypeRef(value)])
    def create_db_cast(self, value, targetType):
        return self.addOp(targetType,["db.cast ",ValueRef(value)," : ",TypeRef(value)," -> ",targetType.to_string()])
    def create_relalg_getattr(self,tuple,attr):
        return self.addOp(attr.type,["relalg.getattr ",ValueRef(tuple), " ",attr.ref_to_string()," : ",attr.type.to_string()])
    def create_relalg_addattr(self,val,attr):
        return self.addExistingOp(Operation(None,None,["relalg.addattr ",attr.def_to_string()," ", ValueRef(val)]))
    def create_relalg_materialize(self,rel,attrs):
        attr_refs=list(map(lambda val:val.ref_to_string(),attrs))
        columns=list(map(lambda val:'"'+val.print_name+'"',attrs))
        t=DBType("table",[])
        return self.addOp(t,["relalg.materialize ",ValueRef(rel)," [",",".join(attr_refs),"]"," => [",",".join(columns),"]", " : ",t.to_string()])
    def create_relalg_projection(self,setsem,rel,attrs):
        attr_refs=list(map(lambda val:val.ref_to_string(),attrs))
        return self.addOp("relation",["relalg.projection ",setsem," [",",".join(attr_refs),"]",ValueRef(rel)])

    def create_relalg_exists(self,rel):
        return self.addOp(DBType("bool"),["relalg.exists", ValueRef(rel)])
    def create_relalg_in(self,val,rel):
        return self.addOp(DBType("bool",[],self.is_any_nullable([val])),["relalg.in ",ValueRef(val)," : ",TypeRef(val),", ",ValueRef(rel)])
    def create_relalg_getscalar(self,rel,attr):
        return self.addOp(attr.type,["relalg.getscalar ",attr.ref_to_string()," ",ValueRef(rel)," : ",attr.type.to_string()])
    def create_relalg_aggr_func(self,type,attr,rel):
        return self.addOp(DBType(attr.type.name,attr.type.baseprops,True),["relalg.aggrfn ",type," ",attr.ref_to_string()," ", ValueRef(rel), " : ",DBType(attr.type.name,attr.type.baseprops,True).to_string()])
    def create_relalg_count_rows(self,rel):
        return self.addOp(DBType("int",["64"]),["relalg.count ",ValueRef(rel)])

    def create_relalg_sort(self,rel,sortSpecifications):
        return self.addOp("relation",["relalg.sort ",ValueRef(rel)," [%s]" % (",".join(map(lambda x: "("+x[0].ref_to_string()+","+x[1]+")",sortSpecifications)))])
    def create_relalg_limit(self,rel,limit):
        return self.addOp("relation",["relalg.limit ",limit," ",ValueRef(rel)])

    def create_db_const(self,const,type):
        var=self.newVar()
        const_str="(\""+str(const)+"\")"
        if type.name=="int":
            const_str="("+str(const)+")"
        self.addExistingOp(Operation(var,type,["db.constant ",const_str," :", TypeRef(var)]))
        return var
    def create_relalg_const_relation(self,scope,attrs,values):
        attr_defs=",".join(list(map(lambda val:val.def_to_string(),attrs)))
        return self.addOp("relation",["relalg.const_relation ","@",scope, " attributes:[%s]" %(attr_defs) ," values: [%s]" % (",".join(map(lambda x: "\""+x+"\"" if type(x) == str else str(x),values)))])



    def addExistingOp(self,op):
        self.getCurrentRegion().addOp(op)
        if op.var != None:
            self.ops[op.var]=op
            return op.var
    def addOp(self,type,print_args):
        var=self.newVar()
        op=Operation(var,type,print_args)
        return self.addExistingOp(op)
    def startModule(self,name):
        self.stacked_ops.append(Operation(None,None,["module ","@",name]))
        self.startRegion()
    def endModule(self):
        op=self.stacked_ops.pop()
        op.print_args.append(self.endRegion())
        self.getCurrentRegion().addOp(op)
    def startFunction(self,name,params):
        self.stacked_ops.append(Operation(None,None,["func ","@",name," (",",".join(params),") "]))
        self.startRegion()
    def endFunction(self,res):
        op = self.stacked_ops.pop()
        op.print_args.append(" -> ")
        op.print_args.append(TypeRef(res))
        return_op=Operation(None, None, ["return ", ValueRef(res)," : ",TypeRef(res)])
        current_region = self.getCurrentRegion()
        current_region.addOp(return_op)
        op.print_args.append(self.endRegion())
        self.addExistingOp(op)
    def startRegionOp(self,type,args):
        var=self.newVar()
        self.stacked_ops.append(Operation(var,type,args))
        self.startRegion()
        return var
    def endRegionOp(self):
        op = self.stacked_ops.pop()
        op.print_args.append(self.endRegion())
        self.addExistingOp(op)
    def endRegionOpWith(self,op):
        current_region = self.getCurrentRegion()
        current_region.addOp(op)
        self.endRegionOp()



    def startSelection(self,rel):
        tuple=self.newParam("tuple")
        return self.startRegionOp("relation",["relalg.selection ",ValueRef(rel),"(",tuple, ": !relalg.tuple) "]),tuple
    def endSelection(self,res):
        self.endRegionOpWith(Operation(None,None,["relalg.return ",ValueRef(res)," : ",TypeRef(res)]))
    def startMap(self,scope_name,rel):
        tuple=self.newParam("tuple")
        return self.startRegionOp("relation",["relalg.map ","@",scope_name," ",ValueRef(rel)," (",tuple, ": !relalg.tuple) "]),tuple
    def endMap(self):
        self.endRegionOpWith(Operation(None, None, ["relalg.return"]))

    def startSelection(self,rel):
        tuple=self.newParam("tuple")
        return self.startRegionOp("relation",["relalg.selection ",ValueRef(rel),"(",tuple, ": !relalg.tuple) "]),tuple
    def endSelection(self,res):
        self.endRegionOpWith(Operation(None,None,["relalg.return ",ValueRef(res)," : ",TypeRef(res)]))
    def startIf(self,val):
        return self.startRegionOp(None,["db.if ", ValueRef(val)," : ",TypeRef(val)," "])
    def addElse(self, yieldval=None):
        if yieldval != None:
            ifop = self.stacked_ops[-1]
            ifop.print_args.append(" -> ")
            ifop.print_args.append(TypeRef(yieldval))
            ifop.print_args.append(" ")
            ifop.type=self.getType(yieldval)
            yieldOp=Operation(None,None,["db.yield ",ValueRef(yieldval)," : ",TypeRef(yieldval)])
        else:
            yieldOp=Operation(None,None,["db.yield"])
        current_region = self.getCurrentRegion()
        current_region.addOp(yieldOp)
        ifop=self.stacked_ops[-1]
        ifop.print_args.append(self.endRegion())
        ifop.print_args.append(" else ")
        self.startRegion()
    def endIf(self, yieldval=None):
        if yieldval != None:
            ifop = self.stacked_ops[-1]
            if ifop.type == None:
                ifop = self.stacked_ops[-1]
                ifop.print_args.append(" -> ")
                ifop.print_args.append(TypeRef(yieldval))
                ifop.print_args.append(" ")
                ifop.type = self.getType(yieldval)
            yieldOp=Operation(None,None,["db.yield ",ValueRef(yieldval)," : ",TypeRef(yieldval)])
        else:
            yieldOp=Operation(None,None,["db.yield"])
        self.endRegionOpWith(yieldOp)


    def startJoin(self, outer,type,left,right):
        tuple = self.newParam("tuple")
        joinop= "outerjoin" if outer else "join"
        if type =="full":
        	type=""
        	joinop="fullouterjoin"
        return self.startRegionOp("relation",
                                  ["relalg."+joinop," ",type," ", ValueRef(left),", ",ValueRef(right), "(", tuple, ": !relalg.tuple) "]), tuple

    def endJoin(self, res):
        self.endRegionOpWith(Operation(None, None, ["relalg.return ", ValueRef(res), " : ", TypeRef(res)]))
    def startAggregation(self,name,rel, attributes):
        relation=self.newParam("relation")
        attr_refs=list(map(lambda val:val.ref_to_string(),attributes))
        return self.startRegionOp("relation",["relalg.aggregation ","@",name," ",ValueRef(rel)," [",",".join(attr_refs),"] (",relation," : !relalg.relation) "]),relation
    def endAggregation(self):
        self.endRegionOpWith(Operation(None, None, ["relalg.return"]))
    def getResult(self):
        self.getCurrentRegion().print(self)
        return self.result
    def getHigherType(self,left:DBType,right:DBType):
        if left==None:
            return right
        nullable=left.nullable or right.nullable
        if left.name > right.name:
            tmp=left
            left=right
            right=tmp
        # int -> decimal / float -> decimal / int -> float
        if (left.name=="decimal" and right.name=="int") or \
                (left.name=="decimal" and right.name=="float") or\
                (left.name =="float" and right.name=="int"):
            return DBType(left.name,left.baseprops,nullable)
        if left.name==right.name:
            if left.name=="int" or left.name=="float":
                width=max(int(left.baseprops[0]),int(right.baseprops[0]))
                return DBType(left.name,[str(width)],nullable)
            if left.name=="decimal":
                p=max(int(left.baseprops[0]),int(right.baseprops[0]))
                s=max(int(left.baseprops[1]),int(right.baseprops[1]))
                return DBType(left.name,[str(p),str(s)],nullable)
        if right.name=="string" or left.name=="string":
            return DBType("string",[],nullable)
        return None
    def getTypesForValues(self,values):
        return list(map(lambda x:self.getType(x),values))
    def getCommonType(self,types):
        return functools.reduce(self.getHigherType,types,None)
    def toCommonType(self,value,common_type):
        op=self.ops[value]
        if type(op) == Operation:
            if op.type.name==common_type.name and op.type.baseprops==common_type.baseprops:
                return value
            if op.print_args[0].startswith("db.constant"):
                op.type=DBType(common_type.name,common_type.baseprops)
                return value
        return self.create_db_cast(value,common_type)

    def toCommonTypes(self,values,common_type):
        return list(map(lambda x:self.toCommonType(x,common_type),values))



