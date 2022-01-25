import duckdb
import json

import pandas as pd
import pyarrow

con = duckdb.connect(database=":memory:")
con.execute("CREATE SEQUENCE file_loc_ids START 1")
con.execute("CREATE TABLE file_loc (id INTEGER, file VARCHAR, line INTEGER)")
con.execute("CREATE TABLE mapping (created_from INTEGER, loc INTEGER)")
con.execute("CREATE TABLE nested(outer_op INTEGER, inner_op INTEGER)")
con.execute("CREATE TABLE operations(loc INTEGER,op VARCHAR)")
con.execute(
    "CREATE TABLE event(type VARCHAR, time BIGINT, event_loc INTEGER, event_symbol VARCHAR, jit_loc INTEGER, jit_symbol VARCHAR, rt_loc INTEGER, rt_symbol VARCHAR)")

file_loc_cache = {}


def get_file_loc(f, l):
    if (f, l) in file_loc_cache:
        return file_loc_cache[(f, l)]
    con.execute("SELECT id from file_loc where file=? and line=?", [f, l])
    res = con.fetchone()
    if res is None:
        con.execute("INSERT INTO file_loc VALUES(nextval('file_loc_ids'),?,?) ", [f, l])
        con.execute("SELECT id from file_loc where file=? and line=?", [f, l])
        res = con.fetchone()
    file_loc_cache[(f, l)] = res[0]
    return res[0]


def get_opt_loc(srcline):
    if srcline is None:
        return None
    splitted = srcline.split(":")
    if len(splitted) == 2:
        return get_file_loc(splitted[0], splitted[1])
    return None


def get_or_none(obj, key):
    if obj is None: return None
    if key in obj:
        return obj[key]
    return None


for i in range(0, 4):
    with open("sourcemap-" + str(i) + ".json", "r") as sourcemap_file:
        current_file = "snapshot-" + str(i) + ".mlir"
        sourcemap = json.load(sourcemap_file)
        for mapping in sourcemap:
            from_loc = get_file_loc(mapping["created_from"]["file"], mapping["created_from"]["line"])
            to_loc = get_file_loc(current_file, mapping["start"]["line"])
            con.execute("INSERT INTO mapping VALUES(?,?)", [from_loc, to_loc])
            con.execute("INSERT INTO nested SELECT ?, f.id from file_loc f where id != ? and not exists(select * from nested f2 where f2.inner_op=f.id) and file=? and line between ? and ?",
                        [to_loc,to_loc, current_file, mapping["start"]["line"], mapping["end"]["line"]])
            con.execute("INSERT INTO operations VALUES(?,?) ", [to_loc, mapping["operation"]])

dict = {"type": [], "time": [], "event_loc": [], "event_symbol": [], "jit_loc": [], "jit_symbol": [], "rt_loc": [],
        "rt_symbol": []}
with open("perf-result.json", "r") as event_file:
    events = json.load(event_file)
    num_events = len(events)
    prog = 0
    for event in events:
        event_location = get_or_none(event, "event_loc")
        jit_location = get_or_none(event, "jit_loc")
        rt_location = get_or_none(event, "rt_loc")
        event_symbol = get_or_none(event_location, "symbol")
        jit_symbol = get_or_none(jit_location, "symbol")
        rt_symbol = get_or_none(rt_location, "symbol")
        event_loc = get_opt_loc(get_or_none(event_location, "srcline"))
        jit_loc = get_opt_loc(get_or_none(jit_location, "srcline"))
        rt_loc = get_opt_loc(get_or_none(rt_location, "srcline"))
        dict["type"].append(event["event"])
        dict["time"].append(event["time"])
        dict["jit_loc"].append(jit_loc)
        dict["jit_symbol"].append(jit_symbol)
        dict["rt_loc"].append(rt_loc)
        dict["rt_symbol"].append(rt_symbol)
        dict["event_loc"].append(event_loc)
        dict["event_symbol"].append(event_symbol)
        if not rt_symbol is None:
            print(rt_symbol)

        print(str(prog) + "/" + str(num_events))
        prog += 1

events_df = pd.DataFrame.from_dict(dict)
con.register('events_df', events_df)
con.execute("INSERT INTO event SELECT * from events_df")

con2 = duckdb.connect(database="profile.db")
con2.from_arrow_table(con.table("event").arrow()).create("event")
con2.from_arrow_table(con.table("operations").arrow()).create("operations")
con2.from_arrow_table(con.table("nested").arrow()).create("nested")
con2.from_arrow_table(con.table("mapping").arrow()).create("mapping")
con2.from_arrow_table(con.table("file_loc").arrow()).create("file_loc")
