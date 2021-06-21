from mlir import Attribute, DBType


class Table:
    def __init__(self, table_identifier, scope_name,row_count,pkey, columns):
        self.table_identifier = table_identifier
        self.scope_name = scope_name
        self.columns = columns
        self.row_count=row_count
        self.pkey=pkey


def getTPCHTable(name, scope_name):
    if name == "nation":
        return Table("nation", scope_name, 25,["n_nationkey"],{
            "n_nationkey": Attribute(scope_name, "n_nationkey", DBType("int", ["64"])),
            "n_name": Attribute(scope_name, "n_name", DBType("string", [])),
            "n_regionkey": Attribute(scope_name, "n_regionkey", DBType("int", ["64"])),
            "n_comment": Attribute(scope_name, "n_comment", DBType("string", ["nullable"])),
        })

    if name == "region":
        return Table("region", scope_name,5, ["r_regionkey"],{
            "r_regionkey": Attribute(scope_name, "r_regionkey", DBType("int", ["64"])),
            "r_name": Attribute(scope_name, "r_name", DBType("string", [])),
            "r_comment": Attribute(scope_name, "r_comment", DBType("string", ["nullable"])),
        })

    if name == "part":
        return Table("part", scope_name,20000, ["p_partkey"],{
            "p_partkey": Attribute(scope_name, "p_partkey", DBType("int", ["64"])),
            "p_name": Attribute(scope_name, "p_name", DBType("string", [])),
            "p_mfgr": Attribute(scope_name, "p_mfgr", DBType("string", [])),
            "p_brand": Attribute(scope_name, "p_brand", DBType("string", [])),
            "p_type": Attribute(scope_name, "p_type", DBType("string", [])),
            "p_size": Attribute(scope_name, "p_size", DBType("int", ["32"])),
            "p_container": Attribute(scope_name, "p_container", DBType("string", [])),
            "p_retailprice": Attribute(scope_name, "p_retailprice", DBType("decimal", ["15", "2"])),
            "p_comment": Attribute(scope_name, "p_comment", DBType("string", [])),
        })

    if name == "supplier":
        return Table("supplier", scope_name,1000,["s_suppkey"], {
            "s_suppkey": Attribute(scope_name, "s_suppkey", DBType("int", ["64"])),
            "s_name": Attribute(scope_name, "s_name", DBType("string", [])),
            "s_address": Attribute(scope_name, "s_address", DBType("string", [])),
            "s_nationkey": Attribute(scope_name, "s_nationkey", DBType("int", ["64"])),
            "s_phone": Attribute(scope_name, "s_phone", DBType("string", [])),
            "s_acctbal": Attribute(scope_name, "s_acctbal", DBType("decimal", ["15", "2"])),
            "s_comment": Attribute(scope_name, "s_comment", DBType("string", [])),
        })

    if name == "partsupp":
        return Table("partsupp", scope_name, 80000, ["ps_partkey","ps_suppkey"],{
            "ps_partkey": Attribute(scope_name, "ps_partkey", DBType("int", ["64"])),
            "ps_suppkey": Attribute(scope_name, "ps_suppkey", DBType("int", ["64"])),
            "ps_availqty": Attribute(scope_name, "ps_availqty", DBType("int", ["32"])),
            "ps_supplycost": Attribute(scope_name, "ps_supplycost", DBType("decimal", ["15", "2"])),
            "ps_comment": Attribute(scope_name, "ps_comment", DBType("string", [])),
        })

    if name == "customer":
        return Table("customer", scope_name,15000, ["c_custkey"],{
            "c_custkey": Attribute(scope_name, "c_custkey", DBType("int", ["64"])),
            "c_name": Attribute(scope_name, "c_name", DBType("string", [])),
            "c_address": Attribute(scope_name, "c_address", DBType("string", [])),
            "c_nationkey": Attribute(scope_name, "c_nationkey", DBType("int", ["64"])),
            "c_phone": Attribute(scope_name, "c_phone", DBType("string", [])),
            "c_acctbal": Attribute(scope_name, "c_acctbal", DBType("decimal", ["15", "2"])),
            "c_mktsegment": Attribute(scope_name, "c_mktsegment", DBType("string", [])),
            "c_comment": Attribute(scope_name, "c_comment", DBType("string", [])),
        })

    if name == "orders":
        return Table("orders", scope_name, 150000, ["o_orderkey"],{
            "o_orderkey": Attribute(scope_name, "o_orderkey", DBType("int", ["64"])),
            "o_custkey": Attribute(scope_name, "o_custkey", DBType("int", ["64"])),
            "o_orderstatus": Attribute(scope_name, "o_orderstatus", DBType("string", [])),
            "o_totalprice": Attribute(scope_name, "o_totalprice", DBType("decimal", ["15", "2"])),
            "o_orderdate": Attribute(scope_name, "o_orderdate", DBType("date", ["day"])),
            "o_orderpriority": Attribute(scope_name, "o_orderpriority", DBType("string", [])),
            "o_clerk": Attribute(scope_name, "o_clerk", DBType("string", [])),
            "o_shippriority": Attribute(scope_name, "o_shippriority", DBType("int", ["32"])),
            "o_comment": Attribute(scope_name, "o_comment", DBType("string", [])),
        })

    if name == "lineitem":
        return Table("lineitem", scope_name,600572,["l_orderkey","l_linenumber"], {
            "l_orderkey": Attribute(scope_name, "l_orderkey", DBType("int", ["64"])),
            "l_partkey": Attribute(scope_name, "l_partkey", DBType("int", ["64"])),
            "l_suppkey": Attribute(scope_name, "l_suppkey", DBType("int", ["64"])),
            "l_linenumber": Attribute(scope_name, "l_linenumber", DBType("int", ["32"])),
            "l_quantity": Attribute(scope_name, "l_quantity", DBType("decimal", ["15", "2"])),
            "l_extendedprice": Attribute(scope_name, "l_extendedprice", DBType("decimal", ["15", "2"])),
            "l_discount": Attribute(scope_name, "l_discount", DBType("decimal", ["15", "2"])),
            "l_tax": Attribute(scope_name, "l_tax", DBType("decimal", ["15", "2"])),
            "l_returnflag": Attribute(scope_name, "l_returnflag", DBType("string", [])),
            "l_linestatus": Attribute(scope_name, "l_linestatus", DBType("string", [])),
            "l_shipdate": Attribute(scope_name, "l_shipdate", DBType("date", ["day"])),
            "l_commitdate": Attribute(scope_name, "l_commitdate", DBType("date", ["day"])),
            "l_receiptdate": Attribute(scope_name, "l_receiptdate", DBType("date", ["day"])),
            "l_shipinstruct": Attribute(scope_name, "l_shipinstruct", DBType("string", [])),
            "l_shipmode": Attribute(scope_name, "l_shipmode", DBType("string", [])),
            "l_comment": Attribute(scope_name, "l_comment", DBType("string", [])),
        })
    if name == "assistenten":
        return Table("assistenten", scope_name,0,["persnr"], {
            "persnr": Attribute(scope_name, "persnr", DBType("int", ["64"])),
            "name": Attribute(scope_name, "name", DBType("string", [])),
            "fachgebiet": Attribute(scope_name, "fachgebiet", DBType("string", [])),
            "boss": Attribute(scope_name, "boss", DBType("int", ["64"]))
        })
    if name == "hoeren":
        return Table("hoeren", scope_name,0,["matrnr","vorlnr"], {
            "matrnr": Attribute(scope_name, "matrnr", DBType("int", ["64"])),
            "vorlnr": Attribute(scope_name, "vorlnr", DBType("int", ["64"])),
        })
    if name == "studenten":
        return Table("studenten", scope_name,0, ["matrnr"],{
            "matrnr": Attribute(scope_name, "matrnr", DBType("int", ["64"])),
            "name": Attribute(scope_name, "name", DBType("string", [])),
            "semester": Attribute(scope_name, "semester", DBType("int", ["64"])),
        })
    if name == "professoren":
        return Table("professoren", scope_name,0, ["persnr"],{
            "persnr": Attribute(scope_name, "persnr", DBType("int", ["64"])),
            "name": Attribute(scope_name, "name", DBType("string", [])),
            "rang": Attribute(scope_name, "rang", DBType("string", [])),
            "raum": Attribute(scope_name, "raum", DBType("int", ["64"])),
        })
    if name == "vorlesungen":
        return Table("vorlesungen", scope_name,0, ["vorlnr"],{
            "vorlnr": Attribute(scope_name, "vorlnr", DBType("int", ["64"])),
            "titel": Attribute(scope_name, "titel", DBType("string", [])),
            "sws": Attribute(scope_name, "sws", DBType("int", ["64"])),
            "gelesenvon": Attribute(scope_name, "gelesenvon", DBType("int", ["64"])),
        })
    if name == "voraussetzen":
        return Table("voraussetzen", scope_name,0, ["vorgaenger","nachfolger"],{
            "vorgaenger": Attribute(scope_name, "vorgaenger", DBType("int", ["64"])),
            "nachfolger": Attribute(scope_name, "nachfolger", DBType("int", ["64"])),
        })
    if name == "pruefen":
        return Table("pruefen", scope_name,0, ["matrnr","vorlrnr","persnr"],{
            "matrnr": Attribute(scope_name, "matrnr", DBType("int", ["64"])),
            "vorlnr": Attribute(scope_name, "vorlnr", DBType("int", ["64"])),
            "persnr": Attribute(scope_name, "persnr", DBType("int", ["64"])),
            "note": Attribute(scope_name, "note", DBType("decimal", ["2", "1"]))
        })
    else:
        raise Exception("unknown table: " + name)
