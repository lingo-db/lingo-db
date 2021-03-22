class Resolver:
    def __init__(self):
        self.lookup_table = {}

    def resolve(self, name):
        if name in self.lookup_table:
            return self.lookup_table[name]
        else:
            raise Exception("could not resolve" + name)

    def add_(self, name, attr):
        if name in self.lookup_table:
            del self.lookup_table[name]
        else:
            self.lookup_table[name] = attr

    def add(self, prefixes, name, attr):
        self.add_(name, attr)
        for prefix in prefixes:
            self.add_(prefix + "." + name, attr)
    def addOverride(self,name,attr):
        self.lookup_table[name]=attr


class StackedResolver:
    def __init__(self):
        self.stack = []

    def push(self, tuple, resolver=Resolver()):
        self.stack.append((tuple, resolver))

    def pop(self):
        self.stack.pop()

    def resolve(self, name):
        resolved = None
        responsible_tuple = ""
        for tuple, resolver in reversed(self.stack):
            try:
                resolved = resolver.resolve(name)
                responsible_tuple = tuple
                break
            except:
                continue
        if resolved == None:
            raise Exception("could not resolve" + name)
        else:
            return resolved, responsible_tuple

    def add(self, prefixes, name, attr):
        self.stack[-1].add(prefixes, name, attr)


