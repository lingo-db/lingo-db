#//RUN: python3 %s %S| FileCheck %s
#//CHECK: 0     2
import sys,os
sys.setdlopenflags(os.RTLD_NOW|os.RTLD_GLOBAL)

import pyarrow as pa
import pymlirdbext
import pandas as pd
df = pd.DataFrame(data={'col1': [1, 2,3,4]})

pymlirdbext.load({"df":pa.Table.from_pandas(df)})
simpleMLIRQuery="""module {
  func.func @main() -> !dsa.table {
    %0 = relalg.basetable {table_identifier = "df"} columns: { col1 => @df::@col1({type = i64})}
    %1 = relalg.selection %0 (%arg0 : !relalg.tuple){
        %4 = db.constant(2) : i64
        %5 = relalg.getcol %arg0 @df::@col1 : i64
        %6 = db.compare gt %5 : i64, %4: i64
        relalg.return %6 : i1
    }
    %2 = relalg.aggregation %1 [] computes : [@aggr0::@tmp_attr0({type = i64})] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
      %4 = relalg.count %arg0
      relalg.return %4 : i64
    }
    %3 = relalg.materialize %2 [@aggr0::@tmp_attr0] => ["count"] : !dsa.table
    return %3 : !dsa.table
  }
}
"""
res = pymlirdbext.run(simpleMLIRQuery)
print(res.to_pandas())
