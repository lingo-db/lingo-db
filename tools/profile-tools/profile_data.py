import duckdb
import seaborn
import plotly
import random


def gen_colors(n):
    
    return sc
def gen_colors(num_colors):
    sc = seaborn.color_palette(None, )
    if len(sc)==num_colors:
        return sc
    points=[]
    for i in range(0,num_colors):
        points.append(i/float(num_colors))
    colors=plotly.colors.sample_colorscale(plotly.colors.sequential.Rainbow, points , low=0.0,
                                                        high=1.0,colortype='tuple')
    random.shuffle(colors)
    return colors

def createOpNameFromRepr(repr):
    splitted=repr.split(" ")
    if len(splitted)>1 and splitted[1].startswith("@"):
        return splitted[0]+splitted[1]
    else:
        return splitted[0]

class OpColorMap:
    def __init__(self, op_ids):
        self.map = {}
        colors = gen_colors(len(op_ids))
        for p in zip(op_ids, colors):
            self.map[self.normalizeId(p[0][0])] = p[1]

    def normalizeId(self, id):
        if type(id) is not int:
            if id.startswith("id_"):
                id = int(id[3:])
            else:
                raise RuntimeError("problem")
        return id

    def lookup(self, id):
        id = self.normalizeId(id)
        if id in self.map:
            c = self.map[id]
            return '#%02x%02x%02x' % (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
        else:
            return None

    def lookupRGBA(self, id, alpha):
        id = self.normalizeId(id)
        if id in self.map:
            c = self.map[id]
            return 'rgba(%d,%d,%d,%f)' % (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), alpha)
        else:
            return None


class ProfileData:
    def __init__(self):
        self.con = duckdb.connect(database=":memory:")
        self.con.execute("CREATE TABLE operation AS SELECT * FROM 'operations.parquet'")
        self.con.execute("CREATE TABLE event AS SELECT * FROM 'events.parquet'")
        self.con.execute("select min(level),max(level) from operation")
        res = self.con.fetchone()
        self.min_level = res[0]
        self.max_level = res[1]
        self.sourcefiles = {}
        for i in range(self.min_level, self.max_level + 1):
            with open('snapshot-' + str(i) + '.mlir', 'r') as file:
                self.sourcefiles[i] = file.read()

    def getOperations(self, cols, level=None, nested_inside=None, order=None):
        o = self.con.table("operation").set_alias('o')
        if level is not None:
            o = o.filter('level=' + str(level))
        if nested_inside is not None:
            n = self.con.table("operation").set_alias('n')
            n = n.filter('repr like \'' + nested_inside + "%\'")
            o = o.join(n, "o.parent=n.id")
        o = o.project("o.*")
        o = o.project(", ".join(cols))
        if order is not None:
            o = o.order(order)
        o.execute()
        return o.fetchall()


