from profile_data import createOpNameFromRepr
def hierarchical_sunburst(profile_data, colo_map,cond=""):
    con = profile_data.con
    if len(cond) > 0:
        cond = " and " + cond

    nested_op_stats = {}
    ops=profile_data.getOperations(["id", "repr"], level=0)

    for op in ops:
        nested_op_stats[op[0]] = {"operation": createOpNameFromRepr(op[1]), "count": 0, "parent": None,"color":colo_map.lookup(op[0])}
    con.execute(
        """select op.id, count(*) as cnt
           from operation op, operation op1, operation op2, operation op3, event e
           where op1.mapping=op.id and op2.mapping=op1.id and op3.mapping=op2.id and op3.loc=e.jit_srcline and e.loc_type=='jit' """ + cond + """ group by op.id order by cnt desc""")

    stats = con.fetchall()
    for stat in stats:
        if stat[0] in nested_op_stats:
            nested_op_stats[stat[0]]["count"] = stat[1]

    con.execute(
        """select op.id,e.symbol, count(*) as cnt
           from operation op, operation op1, operation op2, operation op3, event e
           where op1.mapping=op.id and op2.mapping=op1.id and op3.mapping=op2.id and op3.loc=e.jit_srcline and e.loc_type=='rt' """ + cond + """ group by op.id,e.symbol order by cnt desc""")
    rt_stats = con.fetchall()
    fake_id = 100000000
    rt_ids=[]
    for stat in rt_stats:
        if stat[0] in nested_op_stats:
            nested_op_stats[stat[0]]["count"] += stat[2]
            description = stat[1] + "()" if stat[1] is not None else "<unknown>"
            description = description[0:min(len(description), 30)]
            nested_op_stats[fake_id] = {"operation": description, "count": stat[2], "parent": stat[0],"color":None}
            rt_ids.append(fake_id)
            fake_id += 1

    for p in profile_data.getOperations(["id", "parent"], level=0, order="id desc"):
        if p[1] is not None:
            parent = p[1]
            curr = nested_op_stats[p[0]]
            nested_op_stats[parent]["count"] += curr["count"]
            curr["parent"] = parent
            if curr["color"] is None:
                curr["color"]=nested_op_stats[parent]["color"]
    for rt in rt_ids:
        curr = nested_op_stats[rt]
        if curr["color"] is None and curr["parent"] is not None:
            curr["color"]=nested_op_stats[curr["parent"]]["color"]

    parents = []
    values = []
    names = []
    ids = []
    colors=[]
    for k in nested_op_stats:
        id = "id_" + str(k)
        curr = nested_op_stats[k]
        names.append(curr["operation"])
        ids.append(id)
        values.append(curr["count"])
        if curr["parent"] is None:
            parents.append("")
        else:
            parents.append("id_" + str(curr["parent"]))
        color=curr["color"]
        colors.append("#ffffff" if color is None else color)

    import plotly.express as px

    fig = px.sunburst(
        names=names,
        parents=parents,
        values=values,
        ids=ids,
        branchvalues="total",
        color_discrete_sequence=colors,
        color=ids,
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig
