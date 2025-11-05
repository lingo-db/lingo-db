//RUN: env LINGODB_EXECUTION_MODE=DEFAULT run-mlir %s %S/../../../resources/data/uni| FileCheck %s
//CHECK: |                        matrnr  |                             c  |
//CHECK: |                         28106  |                             4  |
module {
  func.func @main() {
    %0 = relalg.query (){
      %1 = relalg.basetable  {table_identifier = "studenten"} columns: {matrnr => @s::@matrnr({type = !db.nullable<i64>})}
      %2 = relalg.map %1 computes : [@map_u_3::@tmp_attr_u_1({type = i64})] (%arg0: !tuples.tuple){
        %matrnr = tuples.getcol %arg0 @s::@matrnr : !db.nullable<i64>
        %c = relalg.sql_query "select count(*) from hoeren h where h.matrnr = PARAM(1)", %matrnr : !db.nullable<i64> -> !db.nullable<i64>
        %8 = db.isnull %c : <i64>
        %9 = db.nullable_get_val %c : <i64>
        %10 = db.constant(0 : i64) : i64
        %11 = arith.select %8, %10, %9 : i64
        tuples.return %11 : i64
      }
      %3 = relalg.materialize %2 [@s::@matrnr,@map_u_3::@tmp_attr_u_1] => ["matrnr", "c"] : !subop.local_table<[matrnr$0 : !db.nullable<i64>, tmp_attr_u_1$0 : i64], ["matrnr", "c"]>
      relalg.query_return %3 : !subop.local_table<[matrnr$0 : !db.nullable<i64>, tmp_attr_u_1$0 : i64], ["matrnr", "c"]>
    } -> !subop.local_table<[matrnr$0 : !db.nullable<i64>, tmp_attr_u_1$0 : i64], ["matrnr", "c"]>
    subop.set_result 0 %0 : !subop.local_table<[matrnr$0 : !db.nullable<i64>, tmp_attr_u_1$0 : i64], ["matrnr", "c"]>
    return
  }
}