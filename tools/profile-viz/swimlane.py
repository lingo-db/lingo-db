import pandas as pd
import numpy
import plotly.express as px
from profile_data import createOpNameFromRepr


def create_swimline_chart(data, colo_map, cond="",normalized=None):
    if len(cond) > 0:
        cond = " and " + cond
    relevant_ops = data.getOperations(cols=["id", "repr"], level=0, nested_inside="func")
    all_ops = data.getOperations(cols=["id", "repr"], level=0)

    ops = {}
    dcm={}
    for o in relevant_ops:
        name=createOpNameFromRepr(o[1])
        ops[o[0]] = {"parent": o[0], "name": createOpNameFromRepr(o[1])}
        dcm[name+"("+str(o[0])+")"]=colo_map.lookup(o[0])
    for o in all_ops:
        if o[0] not in ops:
            ops[o[0]] = {"parent": None, "name": createOpNameFromRepr(o[1])}
    for p in data.getOperations(["id", "parent"], level=0, order="id desc"):
        if p[1] is not None:
            parent = p[1]
            curr = ops[p[0]]
            if curr["parent"] is None:
                curr["parent"] = ops[parent]["parent"]
    data.con.execute("select min(time),max(time) from event")
    res = data.con.fetchone()
    min = res[0]
    max = res[1]
    stepsize = (max - min) / 100

    data.con.execute("""select (e.time-?)/? as t,op.id, count(*) 
                        from event e, operation op, operation op1, operation op2, operation op3 
                        where op1.mapping=op.id and op2.mapping=op1.id and op3.mapping=op2.id and op3.loc=e.jit_srcline  """ + cond + """ 
                        group by t,op.id""", [min, stepsize])

    res = data.con.fetchall()
    x = []
    y = []
    ids =[]
    names=[]
    grouped_data = {}
    for r in res:
        x.append(r[0])
        y.append(r[2])
        p_id = ops[r[1]]["parent"]
        if p_id is not None:
            id = p_id
        else:
            id = r[1]
        name=ops[id]["name"]+"("+str(id)+")"
        names.append(name)
        if name not in grouped_data:
            grouped_data[name] = []
        grouped_data[name].append(r[2])

    sorted = []
    for k in grouped_data:
        sorted.append((k, numpy.var(grouped_data[k])))
    sorted.sort(key=lambda x: x[1])
    print(sorted)
    insertion_order = [i for i, j in sorted]

    df = pd.DataFrame.from_dict({"x": x, "y": y,"names":names})
    fig = px.area(df, x="x", y="y", color="names", color_discrete_map=dcm,
                  line_group="names", line_shape="linear",
                  category_orders={"names": insertion_order}, groupnorm=normalized)
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig
