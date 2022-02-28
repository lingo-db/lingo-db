import functools
from sql2mlir.mlir import DBType, Attribute, Function, TupleType, DBVectorType


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
        self.print_vars= [var] if var != None else []
        self.type=type
        self.print_args=print_args
        self.metadata={}
    def print(self,codegen):
        if len(self.print_vars)>0:
            codegen.print(",".join(self.print_vars) + " = ")
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
        self.type="tuplestream"

    def print(self, codegen):
        column_str=""
        first=True
        for column_name,attr in self.table.columns.items():
            if first:
                first=False
            else:
                column_str+=",\n"
            column_str+=column_name+" => "+attr.def_to_string();
        pkeys= ", pkey=[\""+("\",\"".join(self.table.pkey))+"\"]" if self.table.pkey else ""
        str="%s = relalg.basetable @%s { table_identifier=\"%s\", rows=%d %s} columns: {" % (self.var,self.table.scope_name,self.table.table_identifier,self.table.row_count,pkeys)
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


    def __init__(self,functions):
        self.var_number = 0
        self.needs_indent=False
        self.indent=0
        self.result =""
        self.ops ={}
        self.stacked_regions=[RootRegion([])]
        self.stacked_ops=[]
        self.functions=functions
        self.uniquer={}
    def getUniqueName(self,base):
        if base in self.uniquer:
            self.uniquer[base] += 1
            base += str(self.uniquer[base])
        else:
            self.uniquer[base] = 0
        return base
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
        return self.addOp("tuplestream", ["relalg.crossproduct ",ValueRef(left),", ",ValueRef(right)])
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
        common_type=self.getCommonType(self.getTypesForValues(values))
        self.addValuesWithTypes(args,self.toCommonTypes(values,common_type))
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
    def create_db_func_call(self,funcname,params):
        func=self.functions.get(funcname)
        casted_params=[]

        for i in range(0,len(func.operandTypes)):
            casted_params.append(self.toCommonType(params[i],func.operandTypes[i]))
        types_as_string=list(map(lambda val:val.to_string(),func.operandTypes))
        return self.addOp(func.resultType,["call"," @",func.name,"(",",".join(casted_params),") : (",",".join(types_as_string),") -> ", func.resultType.to_string()])
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
    def create_relalg_addattr(self,val,attr,tuple):
        return self.addExistingOp(Operation(self.newVar(),DBType("tuple"),["relalg.addattr ",tuple,", ",attr.def_to_string()," ", ValueRef(val)]))
    def create_relalg_materialize(self,rel,attrs):
        attr_refs=list(map(lambda val:val.ref_to_string(),attrs))
        columns=list(map(lambda val:'"'+val.print_name+'"',attrs))
        t=DBType("table",[])
        return self.addOp(t,["relalg.materialize ",ValueRef(rel)," [",",".join(attr_refs),"]"," => [",",".join(columns),"]", " : ",t.to_string()])
    def create_relalg_projection(self,setsem,rel,attrs):
        attr_refs=list(map(lambda val:val.ref_to_string(),attrs))
        return self.addOp("tuplestream",["relalg.projection ",setsem," [",",".join(attr_refs),"]",ValueRef(rel)])

    def create_relalg_exists(self,rel):
        return self.addOp(DBType("bool"),["relalg.exists", ValueRef(rel)])
    def create_relalg_in(self,val,rel):
        return self.addOp(DBType("bool",[],self.is_any_nullable([val])),["relalg.in ",ValueRef(val)," : ",TypeRef(val),", ",ValueRef(rel)])
    def create_relalg_getscalar(self,rel,attr):
        return self.addOp(attr.type,["relalg.getscalar ",attr.ref_to_string()," ",ValueRef(rel)," : ",attr.type.to_string()])
    def create_relalg_getlist(self,rel,attrs):
        attr_refs = list(map(lambda val: val.ref_to_string(), attrs))
        tupleType=TupleType(list(map(lambda attr:attr.type,attrs)))
        t = DBVectorType(tupleType)
        return self.addOp(t, ["relalg.getlist ", ValueRef(rel), " [", ",".join(attr_refs), "] : ", t.to_string()])
    def create_get_tuple(self,tupleVar,idx):
        res_type=self.getType(tupleVar).types[idx]
        return self.addOp(res_type, ["util.get_tuple ", ValueRef(tupleVar), "[",idx,"] : (",TypeRef(tupleVar),") -> ", res_type.to_string()])

    def create_relalg_aggr_func(self,type,attr,rel,isNullable):
        count_t=DBType("int",["64"])
        default_t=DBType(attr.type.name,attr.type.baseprops,isNullable)
        result_t=count_t if type=="count" else default_t
        return self.addOp(result_t,["relalg.aggrfn ",type," ",attr.ref_to_string()," ", ValueRef(rel), " : ",result_t.to_string()])
    def create_relalg_count_rows(self,rel):
        return self.addOp(DBType("int",["64"]),["relalg.count ",ValueRef(rel)])

    def create_relalg_sort(self,rel,sortSpecifications):
        return self.addOp("tuplestream",["relalg.sort ",ValueRef(rel)," [%s]" % (",".join(map(lambda x: "("+x[0].ref_to_string()+","+x[1]+")",sortSpecifications)))])
    def create_relalg_limit(self,rel,limit):
        return self.addOp("tuplestream",["relalg.limit ",limit," ",ValueRef(rel)])

    def create_db_const(self,const,type):
        var=self.newVar()
        const_str="(\""+str(const)+"\")"
        if type.name=="int" or type.name=="bool":
            const_str="("+str(const)+")"
        self.addExistingOp(Operation(var,type,["db.constant ",const_str," :", TypeRef(var)]))
        return var
    def serializeVal(self,val,addParen):
        res=""
        res += "[" if addParen else ""
        if type(val)==list:
            res += "[" if not addParen else ""
            res+=",".join(map(lambda x: self.serializeVal(x,False),val))
            res += "]" if not addParen else ""
        elif type(val)==str:
            res+="\"" + val + "\""
        else:
            res+=str(val)
        res += "]" if addParen else ""
        return res
    def create_relalg_const_relation(self,scope,attrs,values):
        attr_defs=",".join(list(map(lambda val:val.def_to_string(),attrs)))
        return self.addOp("tuplestream",["relalg.const_relation ","@",scope, " attributes:[%s]" %(attr_defs) ," values: [%s]" % (",".join(map(lambda x: self.serializeVal(x,True),values)))])



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
    def startFunction(self,name,params,mode="private"):
        self.stacked_ops.append(Operation(None,None,["func ",mode," @",name," (",",".join(params),") "]))
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
        return self.startRegionOp("tuplestream",["relalg.selection ",ValueRef(rel),"(",tuple, ": !relalg.tuple) "]),tuple
    def endSelection(self,res):
        self.endRegionOpWith(Operation(None,None,["relalg.return ",ValueRef(res)," : ",TypeRef(res)]))
    def startMap(self,scope_name,rel):
        tuple=self.newParam("tuple")
        return self.startRegionOp("tuplestream",["relalg.map ","@",scope_name," ",ValueRef(rel)," (",tuple, ": !relalg.tuple) "]),tuple
    def endMap(self,tuple):
        self.endRegionOpWith(Operation(None, None, ["relalg.return"," ",tuple, " : !relalg.tuple"]))

    def startSelection(self,rel):
        tuple=self.newParam("tuple")
        return self.startRegionOp("tuplestream",["relalg.selection ",ValueRef(rel),"(",tuple, ": !relalg.tuple) "]),tuple
    def endSelection(self,res):
        self.endRegionOpWith(Operation(None,None,["relalg.return ",ValueRef(res)," : ",TypeRef(res)]))
    def startIf(self,val):
        cond= self.addOp(DBType("bool",[],False),["db.derive_truth ",ValueRef(val)," : ",TypeRef(val)])
        self.startRegionOp(None,["scf.if ", ValueRef(cond)," "])
    def startFor(self,param, collection,iter_args,initial_vals):
        iter_arg_str=",".join(list(map(lambda tpl: tpl[0]+" = "+tpl[1],zip(iter_args,initial_vals))))

        iter_arg_types=",".join(list(map(lambda x: self.getType(x).to_string(),iter_args)))
        if len(iter_args)>0:
            iter_arg_str="iter_args("+iter_arg_str+") -> ("+iter_arg_types+")";
        return self.startRegionOp(None,["db.for ", ValueRef(param)," in ",ValueRef(collection)," : ", TypeRef(collection)," ",iter_arg_str])
    def create_db_yield(self,yieldValues):
        yieldTypes=list(map(lambda x:self.getType(x).to_string(),yieldValues))
        return self.addExistingOp(Operation(None,None,["db.yield ",",".join(yieldValues)," : " if len(yieldValues)>0 else "", ",".join(yieldTypes)]))
    def create_scf_yield(self,yieldValues):
        yieldTypes=list(map(lambda x:self.getType(x).to_string(),yieldValues))
        return self.addExistingOp(Operation(None,None,["scf.yield ",",".join(yieldValues)," : " if len(yieldValues)>0 else "", ",".join(yieldTypes)]))

    def endFor(self,yieldValues):
        forop = self.stacked_ops[-1]
        resParams=[]
        for yieldValue in yieldValues:
            resParams.append(self.newParam(self.getType(yieldValue)))
        self.create_db_yield(yieldValues)
        forop.print_vars=resParams
        self.endRegionOp()
        return resParams
    def addElse(self, yieldValues=[]):
        self.create_scf_yield(yieldValues)
        ifop=self.stacked_ops[-1]
        self.addIfReturnTypes(ifop,yieldValues)

        ifop.print_args.append(self.endRegion())
        ifop.print_args.append(" else ")
        self.startRegion()
    def addIfReturnTypes(self,ifop,yieldValues):
        if not "addedReturnTypes" in ifop.metadata:
            ifop.metadata["addedReturnTypes"]=True
            first = True
            for yieldValue in yieldValues:
                if first:
                    ifop.print_args.append(" -> (")
                    first = False
                else:
                    ifop.print_args.append(", ")
                ifop.print_args.append(TypeRef(yieldValue))
            ifop.print_args.append(") ")
    def endIf(self, yieldValues=[]):
        ifop = self.stacked_ops[-1]
        self.addIfReturnTypes(ifop,yieldValues)
        resParams=[]
        for yieldValue in yieldValues:
            resParams.append(self.newParam(self.getType(yieldValue)))
        ifop.print_vars=resParams
        self.create_scf_yield(yieldValues)
        self.endRegionOp()
        return resParams


    def startJoin(self, outer,type,left,right, name=""):
        tuple = self.newParam("tuple")
        joinop= "outerjoin" if outer else "join"
        if type =="full":
        	type=""
        	joinop="fullouterjoin"
        if len(name)>0:
            name=" @"+name
        return self.startRegionOp("tuplestream",
                                  ["relalg."+joinop,name," ", ValueRef(left),", ",ValueRef(right), "(", tuple, ": !relalg.tuple) "]), tuple

    def endJoin(self, res,mapping=None):
        current_region = self.getCurrentRegion()
        current_region.addOp(Operation(None, None, ["relalg.return ", ValueRef(res), " : ", TypeRef(res)]))
        op = self.stacked_ops.pop()
        op.print_args.append(self.endRegion())
        if not mapping is None:
            attr_defs = list(map(lambda val: val.def_to_string(), mapping))
            op.print_args.extend([" mapping: ", "{",",".join(attr_defs),"}"])
        self.addExistingOp(op)
        #if len(mapping)>0:

    def startAggregation(self,name,rel, attributes):
        relation=self.newParam("tuplestream")
        tuple=self.newParam("tuple")
        attr_refs=list(map(lambda val:val.ref_to_string(),attributes))
        return self.startRegionOp("tuplestream",["relalg.aggregation ","@",name," ",ValueRef(rel)," [",",".join(attr_refs),"] (",relation," : !relalg.tuplestream, ",tuple," : !relalg.tuple) "]),relation,tuple
    def endAggregation(self,tuple):
        self.endRegionOpWith(Operation(None, None, ["relalg.return"," ",tuple, " : !relalg.tuple"]))
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
        return left
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



