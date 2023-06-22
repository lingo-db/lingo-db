import pyarrow as pa

import pylingodb

import pandas as pd
con = pylingodb.connect_to_db("./resources/data/uni")
print(con.sql("""
-- all students who never attended a lecture
select * from studenten s
where not exists(select * from hoeren h where h.matrnr=s.matrnr)
""").to_pandas())
print(con.sql("select 1").to_pandas())

df = pd.DataFrame(data={'col1': [1, 2, 3, 4]})

con2 = pylingodb.create_in_memory()
con2.add_table("df", pylingodb.meta_data_from_arrow(pa.Table.from_pandas(df)), pa.Table.from_pandas(df))
print(con2.mlir("""module {
  func.func @main() {
    %0 = relalg.basetable {table_identifier = "df"} columns: { col1 => @df::@col1({type = i64})}
    %1 = relalg.selection %0 (%arg0 : !tuples.tuple){
        %4 = db.constant(2) : i64
        %5 = tuples.getcol %arg0 @df::@col1 : i64
        %6 = db.compare gt %5 : i64, %4: i64
        tuples.return %6 : i1
    }
    %2 = relalg.aggregation %1 [] computes : [@aggr0::@tmp_attr0({type = i64})] (%arg0: !tuples.tuplestream,%arg1: !tuples.tuple){
      %4 = relalg.count %arg0
      tuples.return %4 : i64
    }
    %3 = relalg.materialize %2 [@aggr0::@tmp_attr0] => ["count"] : !subop.result_table<[count: i64]>
    subop.set_result 0 %3 :  !subop.result_table<[count: i64]>
    return
  }
}
""").to_pandas())
print(con2.sql("select count(*) from df where col1>2").to_pandas())
