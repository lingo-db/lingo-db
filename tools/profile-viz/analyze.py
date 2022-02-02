import duckdb
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

con = duckdb.connect(database=":memory:")
con.execute("CREATE TABLE operation AS SELECT * FROM 'operations.parquet'")
con.execute("CREATE TABLE event AS SELECT * FROM 'events.parquet'")
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# con.execute(
#    """select e.loc_type, e.symbol,e.rt_srcline, e.jit_srcline, count(*) as cnt
#       from event e
#       where e.loc_type=='rt'  group by e.loc_type, e.symbol,e.rt_srcline, e.jit_srcline order by cnt desc""")
# print(con.fetchdf())
# exit(0)


con.execute("SELECT op.id,op.repr from operation op where op.loc like 'snapshot-0.mlir%'")
operations = con.fetchall()
nested_op_stats = {}
for op in operations:
    nested_op_stats[op[0]] = {"operation": op[1].split(" ")[0], "count": 0, "children": []}
con.execute(
    """select op.id, count(*) as cnt
       from operation op, operation op1, operation op2, operation op3, event e
       where op1.mapping=op.id and op2.mapping=op1.id and op3.mapping=op2.id and op3.loc=e.jit_srcline and e.loc_type=='jit' group by op.id order by cnt desc""")

stats = con.fetchall()
for stat in stats:
    if stat[0] in nested_op_stats:
        nested_op_stats[stat[0]]["count"] = stat[1]

con.execute(
    """select op.id,e.symbol, count(*) as cnt
       from operation op, operation op1, operation op2, operation op3, event e
       where op1.mapping=op.id and op2.mapping=op1.id and op3.mapping=op2.id and op3.loc=e.jit_srcline and e.loc_type=='rt' group by op.id,e.symbol order by cnt desc""")
# print(con.fetchdf())
rt_stats = con.fetchall()
for stat in rt_stats:
    if stat[0] in nested_op_stats:
        nested_op_stats[stat[0]]["count"] += stat[2]
        description = stat[1] + "()" if stat[1] is not None else "<unknown>"
        description = description[0:min(len(description), 30)]
        nested_op_stats[stat[0]]["children"].append({"operation": description, "count": stat[2], "children": []})

print(nested_op_stats)
# con.execute(
#    "SELECT m3.created_from, f.file  as locid, count(*) as cnt from event e  join mapping m1 on e.event_loc=m1.loc join mapping m2 on #m2.loc=m1.created_from join mapping m3 on m3.loc=m2.created_from join file_loc f on f.id=e.event_loc group by m3.created_from,f.file #order by cnt desc")
# con.execute(
#    "SELECT * from event where rt_symbol is not null")
# print(con.fetchdf())


con.execute("SELECT op.id,op.parent from operation op where op.loc like 'snapshot-0.mlir%' order by op.id desc")
parents = con.fetchall()
root = None
for p in parents:
    if p[1] is None:
        root = p[0]
    else:
        parent = p[1]
        curr = nested_op_stats[p[0]]
        nested_op_stats[parent]["count"] += curr["count"]
        nested_op_stats[parent]["children"].append(curr)
print(nested_op_stats)

# con.execute(
#    "SELECT e.rt_symbol,e.event_symbol,count(*) as cnt from event e join file_loc f on f.id=e.event_loc where f.file not like '%.mlir' and e.jit_loc is null group by e.rt_symbol,e.event_symbol")
# lonely_rt=con.fetchall()
# lonely_rt_dict={}

# for e in lonely_rt:
#    name=e[0] or e[1]
#    count=e[2]
#    if name in lonely_rt_dict:
#        lonely_rt_dict[name]["count"]+=count
#    else:
#        lonely_rt_dict[name] = {"operation": "'"+name+"'", "count": count, "children": []}

# for el  in lonely_rt_dict:
#    nested_op_stats[root]["children"].append(lonely_rt_dict[el])
#    nested_op_stats[root]["count"]+=lonely_rt_dict[el]["count"]
con.execute("SELECT count(*) from event where loc_type is not null")
total_events = con.fetchone()[0]
min_cnt = max(1, total_events / (360 * 2))


def convert_to_sunburst_format(obj):
    if obj["count"] < min_cnt:
        return None
    children = []
    for c in obj["children"]:
        converted = convert_to_sunburst_format(c)
        print(converted)
        if not converted is None:
            children.append(converted)

    return (obj["operation"], obj["count"], children)


plot_data = convert_to_sunburst_format(nested_op_stats[root])


def sunburst(nodes, total=np.pi * 2, offset=0, level=0, ax=None):
    ax = ax or plt.subplot(111, projection='polar')

    if level == 0 and len(nodes) == 1:
        label, value, subnodes = nodes[0]
        ax.bar([0], [0.5], [np.pi * 2])
        ax.text(0, 0, label, ha='center', va='center')
        sunburst(subnodes, total=value, level=level + 1, ax=ax)
    elif nodes:
        d = np.pi * 2 / total
        labels = []
        widths = []
        local_offset = offset
        for label, value, subnodes in nodes:
            labels.append(label)
            widths.append(value * d)
            sunburst(subnodes, total=total, offset=local_offset,
                     level=level + 1, ax=ax)
            local_offset += value
        values = np.cumsum([offset * d] + widths[:-1])
        heights = [1] * len(nodes)
        bottoms = np.zeros(len(nodes)) + level - 0.5
        rects = ax.bar(values, heights, widths, bottoms, linewidth=1,
                       edgecolor='white', align='edge')
        for rect, label in zip(rects, labels):
            x = rect.get_x() + rect.get_width() / 2
            y = rect.get_y() + rect.get_height() / 2
            rotation = (90 + (360 - np.degrees(x) % 180)) % 360
            ax.text(x, y, label, rotation=rotation, ha='center', va='center')

    if level == 0:
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.set_axis_off()


fig1 = plt.figure(figsize=(20, 20))
print(plot_data)

sunburst([plot_data])
plt.show()
fig1.savefig("test.pdf")
# plt.figure(figsize=(9,9))
# plot chart
# ax1 = plt.subplot(111)
# plot =df.plot(kind="pie",y="cnt",ax=ax1,labels=df["line"])
# plt.show()
