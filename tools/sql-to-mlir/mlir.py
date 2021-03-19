
class DBType:
    def __init__ (self,name,baseprops=[],nullable=False):
        self.name=name
        self.baseprops=baseprops
        self.nullable=nullable
    def to_string(self):
        type_props=",".join(self.baseprops)
        if self.nullable:
            if len(type_props)>0:
                type_props+=","
            type_props+="nullable"
        if type_props=="":
            return '!db.%s' % (self.name)
        else:
            return '!db.%s<%s>' % (self.name,type_props)

class Attribute:
    def __init__ (self,scope_name,name,type,from_existing=[],print_name=""):
        self.name=name
        self.scope_name=scope_name
        self.type=type
        self.from_existing=from_existing
        self.print_name=print_name
    def ref_to_string(self):
        return '@%s::@%s' % (self.scope_name,self.name)
    def def_to_string(self):
        props="type=%s" % (self.type.to_string())
        from_existing_def=""
        if len(self.from_existing)>0:
            from_existing_def="=["+",".join(map(lambda x:x.ref_to_string(),self.from_existing))+"]"
        return '@%s({%s})%s' % (self.name,props,from_existing_def)
class CodeGen:
    def __init__(self):
        self.var_number = 0
        self.indent=0
        self.result =""
        self.types ={"%unknown": DBType("unknown")}
    def newVar(self,type):
        self.var_number+=1
        var_str='%%%d' % (self.var_number)
        self.types[var_str]=type
        return var_str
    def add_(self,str):
        self.result+=("    "*self.indent)+str+"\n"
    def add(self,str):
        for line in str.split("\n"):
            self.add_(line)
    def create_relalg_crossproduct(self,left,right):
        return self.create("relation", "relalg.crossproduct %s,%s" % (left,right))
    def getType(self,var):
        type=self.types[var]
        return type
    def create(self,type,str):
        var=self.newVar(type)
        self.add(var +" = "+str)
        return var

    def create_db_and(self, values):
        values_with_types=list(map(lambda val:val+" : "+self.getType(val).to_string(),values))
        return self.create(DBType("bool"),"db.and %s" % (",".join(values_with_types)))
    def create_db_cmp(self, type,values):
        values_with_types=list(map(lambda val:val+" : "+self.getType(val).to_string(),values))
        return self.create(DBType("bool"),"db.compare %s %s" % (type,",".join(values_with_types)))
    def create_db_binary_op(self, name,values):
        values_with_types=list(map(lambda val:val+" : "+self.getType(val).to_string(),values))
        return self.create(DBType("bool"),"db.%s %s" % (name,",".join(values_with_types)))
    def create_relalg_getattr(self,tuple,attr):
        return self.create(attr.type,"relalg.getattr %s %s : %s" % (tuple,attr.ref_to_string(),attr.type.to_string()))
    def create_relalg_addattr(self,val,attr):
        return self.add("relalg.addattr %s %s" % (attr.def_to_string(),val))
    def create_relalg_materialize(self,rel,attrs):
        attr_refs=list(map(lambda val:val.ref_to_string(),attrs))
        return self.create("collection","relalg.materialize %s [%s]" % (rel,",".join(attr_refs)))

    def create_relalg_exists(self,rel):
        return self.create(DBType("bool"),"relalg.exists %s" % (rel))
    def create_relalg_aggr_func(self,type,attr,rel):
        return self.create(DBType("bool"),"relalg.aggr.%s %s %s" % (type,attr.ref_to_string(),rel))
    def create_relalg_count_rows(self,rel):
        return self.create(DBType("bool"),"relalg.count_rows %s" % (rel))
    def create_db_const(self,const,type):
        return self.create(type,"db.constant (\"%s\") : %s" % (const,type.to_string()))


    def startRegion(self):
        self.indent+=1
    def endRegion(self):
        self.indent-=1;
        self.add("}")

    def startModule(self,name):
        self.add("module @"+ name +" {")
        self.startRegion()
    def startFunction(self,name):
        self.add("func @" + name +" {")
        self.startRegion()
    def endFunction(self,res):
        self.add("return %s" % (res));
        self.endRegion()
    def endModule(self):
        self.endRegion()

    def startSelection(self,rel):
        tuple=self.newVar("tuple")
        var=self.create("relation","relalg.selection %s (%s : relalg.tuple) {" % (rel,tuple))
        self.startRegion()
        return var,tuple
    def endSelection(self,res):
        self.add("relalg.return %s : %s" % (res,self.getType(res).to_string()))
        self.endRegion()

    def startJoin(self,outer,type,left,right):
        tuple=self.newVar("tuple")
        joinop= "outerjoin" if outer else "join"
        var=self.create("relation","relalg.%s %s %s,%s (%s : relalg.tuple) {" % (joinop,type,left,right,tuple))
        self.startRegion()
        return var,tuple
    def endJoin(self,res):
        self.add("relalg.return %s : %s" % (res,self.getType(res).to_string()))
        self.endRegion()
    def startAggregation(self,name,rel, attributes):
        relation=self.newVar("relation")
        attr_refs=list(map(lambda val:val.ref_to_string(),attributes))

        var=self.create("relation","relalg.aggregation @%s %s [%s] (%s : relalg.relation) {" % (name,rel,",".join(attr_refs) ,relation))
        self.startRegion()
        return var,relation
    def endAggregation(self):
        self.endRegion()
    def startMap(self,scope_name,rel):
        tuple=self.newVar("tuple")
        var=self.create("relation","relalg.map @%s %s (%s : relalg.tuple) {" % (scope_name,rel,tuple))
        self.startRegion()
        return var,tuple
    def endMap(self):
        self.endRegion()


