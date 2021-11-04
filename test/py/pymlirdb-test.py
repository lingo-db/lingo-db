#//RUN: python3 %s %S| FileCheck %s

#//CHECK:                 s_name  numwait
#//CHECK: 0   Supplier#000000825       16
#//CHECK: 1   Supplier#000000445       16
#//CHECK: 2   Supplier#000000762       15
#//CHECK: 3   Supplier#000000709       15
#//CHECK: 4   Supplier#000000357       14
#//CHECK: 5   Supplier#000000399       14
#//CHECK: 6   Supplier#000000496       14
#//CHECK: 7   Supplier#000000977       13
#//CHECK: 8   Supplier#000000188       12
#//CHECK: 9   Supplier#000000415       12
#//CHECK: 10  Supplier#000000889       12
#//CHECK: 11  Supplier#000000144       12
#//CHECK: 12  Supplier#000000708       12
#//CHECK: 13  Supplier#000000472       12
#//CHECK: 14  Supplier#000000633       12
#//CHECK: 15  Supplier#000000659       11
#//CHECK: 16  Supplier#000000602       11
#//CHECK: 17  Supplier#000000821       11
#//CHECK: 18  Supplier#000000380       11
#//CHECK: 19  Supplier#000000929       11
#//CHECK: 20  Supplier#000000262       10
#//CHECK: 21  Supplier#000000669       10
#//CHECK: 22  Supplier#000000778       10
#//CHECK: 23  Supplier#000000486       10
#//CHECK: 24  Supplier#000000460       10
#//CHECK: 25  Supplier#000000718       10
#//CHECK: 26  Supplier#000000578        9
#//CHECK: 27  Supplier#000000167        9
#//CHECK: 28  Supplier#000000687        9
#//CHECK: 29  Supplier#000000673        9
#//CHECK: 30  Supplier#000000565        8
#//CHECK: 31  Supplier#000000648        8
#//CHECK: 32  Supplier#000000074        8
#//CHECK: 33  Supplier#000000918        8
#//CHECK: 34  Supplier#000000811        7
#//CHECK: 35  Supplier#000000610        7
#//CHECK: 36  Supplier#000000670        7
#//CHECK: 37  Supplier#000000503        7
#//CHECK: 38  Supplier#000000427        7
#//CHECK: 39  Supplier#000000788        6
#//CHECK: 40  Supplier#000000660        6
#//CHECK: 41  Supplier#000000500        6
#//CHECK: 42  Supplier#000000846        6
#//CHECK: 43  Supplier#000000436        6
#//CHECK: 44  Supplier#000000379        6
#//CHECK: 45  Supplier#000000114        6
#//CHECK: 46  Supplier#000000920        4
import os,sys

build_dir_path=os.getcwd()+'/../../'
sys.path.insert(0,build_dir_path)
sys.path.insert(0,sys.argv[1]+'/../../tools/pymlirdb')
sys.path.insert(0,sys.argv[1]+'/../../arrow/python')

import pymlirdb
from sql2mlir.mlir import Function, DBType
from sql2mlir.tables import loadSchema
import pyarrow as pa

loadSchema(None)# load default schema


supplier=pa.ipc.open_file(pa.OSFile(sys.argv[1]+'/../../resources/data/tpch/supplier.arrow')).read_all()
lineitem=pa.ipc.open_file(pa.OSFile(sys.argv[1]+'/../../resources/data/tpch/lineitem.arrow')).read_all()
orders=pa.ipc.open_file(pa.OSFile(sys.argv[1]+'/../../resources/data/tpch/orders.arrow')).read_all()
nation=pa.ipc.open_file(pa.OSFile(sys.argv[1]+'/../../resources/data/tpch/nation.arrow')).read_all()

pymlirdb.load_tables({"supplier":supplier,"lineitem":lineitem,"orders":orders,"nation":nation})

def count_delayed_orders_for_supplier(suppkey):
    items = pymlirdb.read_table("lineitem")
    items = items[items["l_suppkey"]==suppkey]
    orders = pymlirdb.read_table("orders")
    orders = orders[orders["o_orderstatus"] == 'F']
    orders = orders.join(items, on = (orders["o_orderkey"] == items["l_orderkey"]), how="left")
    orders["delayed"] = is_delayed(orders["o_orderkey"], suppkey)
    orders = orders[orders["delayed"] == True]
    return orders.count()

def is_delayed(orderkey, suppkey):
    containsDelayed = False
    containsOther = False
    onlyDelayed = True
    items = pymlirdb.read_table("lineitem")
    items = items[items["l_orderkey"] == orderkey]
    for item in items:
        if item["l_suppkey"] == suppkey:
            if item["l_receiptdate"] > item["l_commitdate"]:
                containsDelayed = True
        else:
            containsOther = True
            if item["l_receiptdate"] > item["l_commitdate"]:
                onlyDelayed = False
    return containsDelayed and containsOther and onlyDelayed

pymlirdb.registerFunction(Function("count_delayed_orders_for_supplier",[DBType("int",["64"])],DBType("int",["64"]),count_delayed_orders_for_supplier))
pymlirdb.registerFunction(Function("is_delayed",[DBType("int",["64"]),DBType("int",["64"])],DBType("bool"),is_delayed))

df = pymlirdb.query("""
                        select s_name, count_delayed_orders_for_supplier(s_suppkey) as numwait
                        from supplier, nation
                        where s_nationkey=n_nationkey
                        and n_name='SAUDI ARABIA'
                        order by numwait desc
                    """)
print(df.to_pandas())
