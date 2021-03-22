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
