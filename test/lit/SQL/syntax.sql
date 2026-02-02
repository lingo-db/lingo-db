--// RUN: sql-to-mlir %s %S/../../../resources/data/test | FileCheck %s
--//CHECK: module {
--//CHECK:     func.func @main() {
--//CHECK:         %{{.*}} = relalg.const_relation columns : [@dummyScope::@dummyName({type = i32})] values :
--//CHECK:         %{{.*}} = relalg.map %{{.*}} computes : [@{{.*}}::@tmp_attr({type = i32})] (%arg0: !tuples.tuple){
--//CHECK:             %{{.*}} = db.constant(1 : i32) : i32
--//CHECK:             tuples.return %{{.*}} : i32
--//CHECK:         }
--//CHECK:         %{{.*}} = relalg.materialize %{{.*}} [@{{.*}}::@tmp_attr] => [""] : !subop.local_table<[tmp_attr$0 : i32], [""]>
--//CHECK:         subop.set_result 0 %{{.*}} : !subop.local_table<[tmp_attr$0 : i32], [""]>
--//CHECK:         return
--//CHECK:     }
--//CHECK: }
select 1;
--//CHECK: module
--//CHECK: %{{.*}} = db.add %{{.*}} : i32, %{{.*}} : i32
select 3+2;
--//CHECK: module
--//CHECK: %{{.*}} = db.sub %{{.*}} : i32, %{{.*}} : i32
select 3-2;
--//CHECK: module
--//CHECK: %{{.*}} = db.mul %{{.*}} : i32, %{{.*}} : i32
select 3*2;
--//CHECK: module
--//CHECK: %{{.*}} = db.div %{{.*}} : i32, %{{.*}} : i32
select 3/2;
--//CHECK: module
--//CHECK: %{{.*}} = db.div %{{.*}} : !db.decimal<1, 0>, %{{.*}} : !db.decimal<19, 0>
select 3::decimal(1,0)/2;
--//CHECK: module
--//CHECK: %{{.*}} = db.mod %{{.*}} : i32, %{{.*}} : i32
select 3%2;
--//CHECK: module
--//CHECK: %{{.*}} = db.constant("2023-07-18") : !db.date<day>
--//CHECK: %{{.*}} = db.constant("5 days") : !db.interval<daytime>
--//CHECK: %{{.*}} = db.runtime_call "DateAdd"(%{{.*}}, %{{.*}}) : (!db.date<day>, !db.interval<daytime>) -> !db.date<day>
select date '2023-07-18' + interval '5 days';
--//CHECK: module
--//CHECK: %{{.*}} = db.constant("2023-07-18") : !db.date<day>
--//CHECK: %{{.*}} = db.constant("5 days") : !db.interval<daytime>
--//CHECK: %{{.*}} = db.runtime_call "DateSubtract"(%{{.*}}, %{{.*}}) : (!db.date<day>, !db.interval<daytime>) -> !db.date<day>
select date '2023-07-18' - interval '5 days';
--//CHECK: module
--//CHECK: %{{.*}} = db.constant(2 : i32) : i32
--//CHECK: %{{.*}} = db.constant(3 : i32) : i32
--//CHECK: %{{.*}} = db.between %{{.*}} : i32 between %{{.*}} : i32, %{{.*}} : i32, lowerInclusive : true, upperInclusive : true
--//CHECK: %{{.*}} = db.not %{{.*}} : i1
select 1 not between 2 and 3;
--//CHECK: %{{.*}} = db.constant(2 : i32) : i32
--//CHECK: %{{.*}} = db.between %{{.*}} : i32 between %{{.*}} : i32, %{{.*}} : i32, lowerInclusive : true, upperInclusive : true
select 1 between 1 and 2;
--//CHECK: module
--//CHECK: %{{.*}} = db.constant(1 : i32) : i32
--//CHECK: %{{.*}} = db.constant(2 : i32) : i32
--//CHECK: %{{.*}} = db.oneof %{{.*}} : i32 ? %{{.*}}, %{{.*}} : i32, i32
select 1 in (1,2);
--//CHECK: module
--//CHECK: %{{.*}} = db.constant(false) : i1
select false;
--//CHECK: module
--//CHECK: %{{.*}} = db.null : <none>
--//CHECK: %{{.*}} = db.isnull %{{.*}} : <none>
select null is null;
--//CHECK: module
--//CHECK: %{{.*}} = db.null : <none>
--//CHECK: %{{.*}} = db.isnull %{{.*}} : <none>
--//CHECK: %{{.*}} = db.not %{{.*}} : i1
select null is not null;
--//CHECK: module
--//CHECK: %{{.*}} = db.constant("1.2") : !db.string
select cast(1.2 as string);
--//CHECK: module
--//CHECK: %{{.*}} = db.constant(42 : i32) : i32
--//CHECK: %{{.*}} = db.compare gt %{{.*}} : i32, %{{.*}} : i32
--//CHECK: %{{.*}} = db.derive_truth %{{.*}} : i1
--//CHECK: %{{.*}} = scf.if %{{.*}} -> (i32) {
--//CHECK:   %{{.*}} = tuples.getcol %arg0 @{{.*}}::@const{{.*}} : i32
--//CHECK:   %{{.*}} = db.constant(2 : i32) : i32
--//CHECK:   %{{.*}} = db.div %{{.*}} : i32, %{{.*}} : i32
--//CHECK:   scf.yield %{{.*}} : i32
--//CHECK: } else {
--//CHECK:   %{{.*}} = db.constant(0 : i32) : i32
--//CHECK:   scf.yield %{{.*}} : i32
--//CHECK: }
select x, case when x>42 then x/2 else 0 end from (values (1)) t(x);
--//CHECK: module
--//CHECK: %{{.*}} = db.constant(1 : i32) : i32
--//CHECK: %{{.*}} = db.compare eq %{{.*}} : i32, %{{.*}} : i32
--//CHECK: %{{.*}} = db.derive_truth %{{.*}} : i1
--//CHECK: %{{.*}} = scf.if %{{.*}} -> (i32) {
--//CHECK:   %{{.*}} = db.constant(10 : i32) : i32
--//CHECK:   scf.yield %{{.*}} : i32
--//CHECK: } else {
--//CHECK:   %{{.*}} = tuples.getcol %arg0 @{{.*}}::@const{{.*}} : i32
--//CHECK:   %{{.*}} = db.constant(2 : i32) : i32
--//CHECK:   %{{.*}} = db.compare eq %{{.*}} : i32, %{{.*}} : i32
--//CHECK:   %{{.*}} = db.derive_truth %{{.*}} : i1
--//CHECK:   %{{.*}} = scf.if %{{.*}} -> (i32) {
--//CHECK:     %{{.*}} = db.constant(20 : i32) : i32
--//CHECK:     scf.yield %{{.*}} : i32
--//CHECK:   } else {
--//CHECK:     %{{.*}} = db.constant(0 : i32) : i32
--//CHECK:     scf.yield %{{.*}} : i32
--//CHECK:   }
--//CHECK:   scf.yield %{{.*}} : i32
--//CHECK: }
select x, case when x=1 then 10 when x=2 then 20 else 0 end from (values (1)) t(x);
--//CHECK: module
--//CHECK: %{{.*}} = db.constant(1 : i32) : i32
--//CHECK: %{{.*}} = db.compare eq %{{.*}} : i32, %{{.*}} : i32
--//CHECK: %{{.*}} = db.derive_truth %{{.*}} : i1
--//CHECK: %{{.*}} = scf.if %{{.*}} -> (!db.nullable<i32>) {
--//CHECK:   %{{.*}} = db.constant(10 : i32) : i32
--//CHECK:   %{{.*}} = db.as_nullable %{{.*}} : i32 -> <i32>
--//CHECK:   scf.yield %{{.*}} : !db.nullable<i32>
--//CHECK: } else {
--//CHECK:   %{{.*}} = db.constant(2 : i32) : i32
--//CHECK:   %{{.*}} = db.compare eq %{{.*}} : i32, %{{.*}} : i32
--//CHECK:   %{{.*}} = db.derive_truth %{{.*}} : i1
--//CHECK:   %{{.*}} = scf.if %{{.*}} -> (!db.nullable<i32>) {
--//CHECK:     %{{.*}} = db.constant(20 : i32) : i32
--//CHECK:     %{{.*}} = db.as_nullable %{{.*}} : i32 -> <i32>
--//CHECK:     scf.yield %{{.*}} : !db.nullable<i32>
--//CHECK:   } else {
--//CHECK:     %{{.*}} = db.null : <i32>
--//CHECK:     scf.yield %{{.*}} : !db.nullable<i32>
--//CHECK:   }
--//CHECK:   scf.yield %{{.*}} : !db.nullable<i32>
--//CHECK: }
select x, case x when 1 then 10 when 2 then 20 end from (values (1)) t(x);
--//CHECK: module
--//CHECK: %{{.*}} = db.null
--//CHECK: %{{.*}} = db.isnull %{{.*}}
--//CHECK: %{{.*}} = db.not %{{.*}} : i1
--//CHECK: %{{.*}} = scf.if %{{.*}} -> (i32) {
--//CHECK:   %{{.*}} = db.nullable_get_val %{{.*}}
--//CHECK:   scf.yield %{{.*}} : i32
--//CHECK: } else {
--//CHECK:   %{{.*}} = db.constant(1 : i32) : i32
--//CHECK:   scf.yield %{{.*}} : i32
--//CHECK: }
select coalesce(null,1);
--//CHECK: module
--//CHECK: %{{.*}} = db.constant("a") : !db.string
--//CHECK: %{{.*}} = db.constant("b") : !db.string
--//CHECK: %{{.*}} = db.runtime_call "Concatenate"(%{{.*}}, %{{.*}}) : (!db.string, !db.string) -> !db.string
select 'a' || 'b';
--//CHECK: module
--//CHECK: %{{.*}} = db.constant("hello world") : !db.string
--//CHECK: %{{.*}} = db.constant("hello %") : !db.string
--//CHECK: %{{.*}} = db.runtime_call "Like"(%{{.*}}, %{{.*}}) : (!db.string, !db.string) -> i1
select 'hello world' like 'hello %';
--//CHECK: module
--//CHECK: %{{.*}} = relalg.aggregation %{{.*}} [] computes :
--//CHECK-DAG:   %{{.*}} = relalg.aggrfn min @{{.*}}::@const{{.*}} %arg0 : !db.nullable<i32>
--//CHECK-DAG:   %{{.*}} = relalg.aggrfn max @{{.*}}::@const{{.*}} %arg0 : !db.nullable<i32>
--//CHECK-DAG:   %{{.*}} = relalg.aggrfn sum @{{.*}}::@const{{.*}} %arg0 : !db.nullable<i32>
--//CHECK-DAG:   %{{.*}} = relalg.aggrfn count @{{.*}}::@const{{.*}} %arg0 : i64
--//CHECK-DAG:   %{{.*}} = relalg.count %arg0
--//CHECK:   tuples.return
--//CHECK: }
select min(x),max(x),sum(x),count(x), count(*)  from (values (1)) t(x);
--//CHECK: module
--//CHECK: %{{.*}} = relalg.aggregation %{{.*}} [@{{.*}}{{.*}}::@const{{.*}}] computes :
--//CHECK-DAG:       %{{.*}} = relalg.count %arg0
--//CHECK-DAG:       %{{.*}} = relalg.aggrfn count @{{.*}}{{.*}}::@const{{.*}} %arg0 : i64
--//CHECK-DAG:       %{{.*}} = relalg.aggrfn sum @{{.*}}{{.*}}::@const{{.*}} %arg0 : i32
--//CHECK-DAG:       %{{.*}} = relalg.aggrfn max @{{.*}}{{.*}}::@const{{.*}} %arg0 : i32
--//CHECK-DAG:       %{{.*}} = relalg.aggrfn min @{{.*}}{{.*}}::@const{{.*}} %arg0 : i32
--//CHECK-NOT:       %{{.*}} = relalg.aggrfn count @{{.*}}{{.*}}::@const{{.*}} %arg0 : i64
--//CHECK:       tuples.return
--//CHECK: }
select y, min(x),max(x),sum(x),count(x), count(*) from (values (1,2)) t(x,y) group by y having count(x)>0;
--//CHECK: module
--//CHECK: call @{{.*}}RelationHelper{{.*}}createTable{{.*}}(%{{.*}}) : (!util.varlen32) -> ()
create table test_tmp(
                     str varchar(20),
                     float32 float(2),
                     float64 float(4),
                     decimal decimal(5, 2),
                     int32 int,
                     int64 bigint,
                     bool bool,
                     date32 date,
                     date64 timestamp,
                     primary key(float64)
);
--//CHECK: module
--//CHECK: %{{.*}} = relalg.const_relation
--//CHECK: %{{.*}} = relalg.map
--//CHECK: %{{.*}} = relalg.materialize
--//CHECK: subop.set_result 0
--//CHECK: call @{{.*}}RelationHelper{{.*}}appendTableFromResult{{.*}}(%{{.*}}, %{{.*}}) : (!util.varlen32, i64) -> ()
INSERT into test(str, float32, float64, decimal, int32, int64, bool, date32, date64, char1, char20) values ('str', 1.1, 1.1, 1.10, 1, 1, 1, '1996-01-02', '1996-01-02 13:37','a','abcdefghijklmnopqrst'), (null, null, null, null, null, null, null, null, null, null, null);
--//CHECK: module
--//CHECK: %{{.*}} = util.varlen32_create_const "test"
--//CHECK: %{{.*}} = util.varlen32_create_const "t.csv"
--//CHECK: %{{.*}} = util.varlen32_create_const "|"
--//CHECK: %{{.*}} = util.varlen32_create_const "\\"
--//CHECK: call @{{.*}}RelationHelper{{.*}}copyFromIntoTable{{.*}}(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!util.varlen32, !util.varlen32, !util.varlen32, !util.varlen32) -> ()
copy test from 't.csv' csv escape '\' delimiter '|' null '';
--//CHECK: %{{.*}} = relalg.aggregation %{{.*}} [@{{.*}}{{.*}}::@const{{.*}}] computes : [@{{.*}}::@{{.*}}({type = i32})] (%arg0: !tuples.tuplestream,%arg1: !tuples.tuple){
--//CHECK:       %{{.*}} = relalg.projection distinct [@{{.*}}::@const{{.*}}] %arg0
--//CHECK:       %{{.*}} = relalg.aggrfn sum @{{.*}}::@const{{.*}} %{{.*}} : i32
--//CHECK-NOT:       %{{.*}} = relalg.aggrfn sum @{{.*}}::@const{{.*}} %{{.*}} : i32
--//CHECK:       tuples.return %{{.*}} : i32
--//CHECK: }
select x,sum(distinct y) from (values (1,2)) t(x,y) group by x;
--//CHECK: %{{.*}} = relalg.map %1 computes : [@{{.*}}::@{{.*}}({type = i32})] (%arg0: !tuples.tuple){
--//CHECK:       %{{.*}} = db.add %{{.*}} : i32, %{{.*}} : i32
--//CHECK: %{{.*}} = relalg.aggregation %{{.*}} [@{{.*}}{{.*}}::@const{{.*}}] computes : [@{{.*}}::@{{.*}}({type = i32})] (%arg0: !tuples.tuplestream,%arg1: !tuples.tuple){
--//CHECK:       %{{.*}} = relalg.aggrfn sum @{{.*}}::@{{.*}} %{{.*}} : i32
--//CHECK-NOT:       %{{.*}} = relalg.aggrfn sum @{{.*}}::@{{.*}} %{{.*}} : i32
--//CHECK:       tuples.return %{{.*}} : i32
--//CHECK: }
select x,sum(y+1) from (values (1,2)) t(x,y) group by x having sum(y+1)<x;
--//CHECK: %{{.*}} = relalg.sort %{{.*}} [(@{{.*}}::@const{{.*}},asc),(@{{.*}}::@const{{.*}},desc),(@{{.*}}::@const{{.*}},asc)]
select * from (values (1,2,3)) t(x,y,z) order by x, y desc, z asc;
--//CHECK: %{{.*}} = relalg.sort %{{.*}} [(@{{.*}}::@const{{.*}},asc),(@{{.*}}::@const{{.*}},desc),(@{{.*}}::@const{{.*}},asc)]
select * from (values (1,2,3)) t(x,y,z) order by 1, 2 desc, 3 asc;
--//CHECK: %{{.*}} = relalg.sort %{{.*}} [(@{{.*}}::@const{{.*}},asc),(@{{.*}}::@const{{.*}},desc),(@{{.*}}::@const{{.*}},asc)]
with t (x,y,z) as (select * from (values (1,2,3)) t(x,y,z) order by x, y desc, z asc) select * from t where x=1;
--//CHECK: %[[AGGR1:.*]] = relalg.aggregation
--//CHECK: %{{.*}} = relalg.aggrfn sum @{{.*}}::@{{.*}} %{{.*}} : i32
--//CHECK-NOT: %{{.*}} = relalg.aggrfn sum @{{.*}}::@{{.*}} %{{.*}} : i32
--//CHECK: %[[AGGR2:.*]] = relalg.aggregation %[[AGGR1]]
--//CHECK: %{{.*}} = relalg.aggrfn sum @{{.*}}::@{{.*}} %{{.*}} : i32
--//CHECK-NOT: %{{.*}} = relalg.aggrfn sum @{{.*}}::@{{.*}} %{{.*}} : i32
--//CHECK: %{{.*}} = relalg.aggregation %[[AGGR2]]
--//CHECK: %{{.*}} = relalg.aggrfn sum @{{.*}}::@{{.*}} %{{.*}} : i32
--//CHECK-NOT: %{{.*}} = relalg.aggrfn sum @{{.*}}::@{{.*}} %{{.*}} : i32
--//CHECK: %{{.*}} = relalg.union all
--//CHECK: %{{.*}} = relalg.union all
select x,y,sum(z) from (values (1,2,3)) t(x,y,z) group by rollup(x,y) having sum(z)<1 order by sum(z);
--//CHECK: %{{.*}} = relalg.window %{{.*}} partitionBy : [@{{.*}}::@const{{.*}}] orderBy : [(@{{.*}}::@const{{.*}},asc)] rows_between : -9223372036854775808 and 0 computes : [@tmp_attr::@sum({type = !db.nullable<i32>})] (%arg0: !tuples.tuplestream,%arg1: !tuples.tuple){
--//CHECK:       %{{.*}} = relalg.aggrfn sum @{{.*}}::@const{{.*}} %arg0 : !db.nullable<i32>
--//CHECK:       tuples.return %{{.*}} : !db.nullable<i32>
--//CHECK: }
select sum(x) over (partition by y order by z) from (values (1,2,3)) t(x,y,z);
--//CHECK: %{{.*}} = relalg.window %{{.*}} partitionBy : [@{{.*}}::@const{{.*}}] orderBy : [(@{{.*}}::@const{{.*}},asc)] rows_between : -9223372036854775808 and 0 computes : [@tmp_attr::@sum({type = !db.nullable<i32>})] (%arg0: !tuples.tuplestream,%arg1: !tuples.tuple){
--//CHECK:       %{{.*}} = relalg.aggrfn sum @{{.*}}::@const{{.*}} %arg0 : !db.nullable<i32>
--//CHECK:       tuples.return %{{.*}} : !db.nullable<i32>
--//CHECK: }
select sum(x) over (partition by y order by z rows between unbounded preceding and current row) from (values (1,2,3)) t(x,y,z);
--//CHECK: %{{.*}} = relalg.window %{{.*}} partitionBy : [@{{.*}}::@const{{.*}}{{.*}}] orderBy : [(@{{.*}}::@const{{.*}}{{.*}},asc)] rows_between : -100 and 100 computes : [@{{.*}}::@{{.*}}({type = !db.nullable<i32>})] (%arg0: !tuples.tuplestream,%arg1: !tuples.tuple){
--//CHECK:       %{{.*}} = relalg.aggrfn sum @{{.*}}::@const{{.*}} %arg0 : !db.nullable<i32>
--//CHECK:       tuples.return %{{.*}} : !db.nullable<i32>
--//CHECK: }
select sum(x) over (partition by y order by z rows between 100 preceding and 100 following) from (values (1,2,3)) t(x,y,z);
--//CHECK: %{{.*}}  = relalg.union distinct
select * from test union select * from test;
--//CHECK: %{{.*}}  = relalg.union all
select * from test union all select * from test;
--//CHECK: %{{.*}}  = relalg.union distinct
--//CHECK: %{{.*}}  = relalg.limit 1 {{.*}}
select * from test union select * from test LIMIT 1;
--//CHECK: %{{.*}}  = relalg.outerjoin
select * from test t1 left outer join test t2 on t1.bool=t2.bool;
--//CHECK: %{{.*}}  = relalg.outerjoin
select * from test t1 right outer join test t2 on t1.bool=t2.bool;
--//CHECK: %{{.*}}  = relalg.fullouterjoin
select * from test t1 full outer join test t2 on t1.bool=t2.bool;
--//CHECK: %{{.*}}  = relalg.join
select * from test t1 join test t2 on t1.bool=t2.bool;
--//CHECK: module
--//CHECK: %{{.*}} = relalg.map %{{.*}} computes :
--//CHECK:       %{{.*}} = relalg.exists %{{.*}}
--//CHECK:       tuples.return %{{.*}} : i1
--//CHECK: }
select exists(select 1);
--//CHECK: module
--//CHECK: %{{.*}} = relalg.map %{{.*}} computes :
--//CHECK:       %{{.*}} = relalg.getscalar
--//CHECK:       tuples.return %{{.*}}
--//CHECK: }
select (select 1);
--//CHECK-DAG: %{{.*}} = relalg.selection %{{.*}} (%arg1: !tuples.tuple){
--//CHECK-DAG:         %{{.*}} = db.constant(1 : i32) : i32
--//CHECK:         %{{.*}} = tuples.getcol %arg1 @map{{.*}}
--//CHECK:         %{{.*}} = db.compare eq %{{.*}} : i32, %{{.*}} : i32
select 1=all(select 1);
--//CHECK-DAG: %{{.*}} = relalg.selection %{{.*}} (%arg1: !tuples.tuple){
--//CHECK-DAG:   %{{.*}} = db.constant(1 : i32) : i32
--//CHECK:   %{{.*}} = tuples.getcol %arg1 @map{{.*}}
--//CHECK:   %{{.*}} = db.compare eq %{{.*}} : i32, %{{.*}} : i32
--//CHECK:   tuples.return %{{.*}} : i1
--//CHECK: }
--//CHECK: %{{.*}} = relalg.exists %{{.*}}
select 1=any(select 1);
--//CHECK: call @{{.*}}RelationHelper{{.*}}setPersist{{.*}}(%true) : (i1) -> ()
set persist=1;
--//CHECK: module
--//CHECK:  %{{.*}} = relalg.aggregation
select case when x=1 then 10 when x=2 then 20 else 0 end from (values (1)) t(x) group by case when x=1 then 10 when x=2 then 20 else 0 end;
--//CHECK: module
--//CHECK:  %{{.*}} = db.runtime_call "AbsInt"({{.*}}) : (i32) -> i32
--//CHECK: }
--//CHECK:  %{{.*}} = relalg.aggregation {{.*}} [@{{.*}}::@{{.*}}] computes : [] (%arg0: !tuples.tuplestream,%arg1: !tuples.tuple){
select abs(x+1) from (values (1)) t(x) group by abs(x+1);
--//CHECK: module
--//CHECK:  %{{.*}} = db.compare eq %{{.*}} : i32, %{{.*}} : i32
--//CHECK: }
--//CHECK:  %{{.*}} = relalg.aggregation {{.*}} [@{{.*}}::@{{.*}}] computes : [] (%arg0: !tuples.tuplestream,%arg1: !tuples.tuple){
select x=1 and y=2 from (values (1,2)) t(x,y) group by x=1 and y=2 ;
--//CHECK: module
--//CHECK:  %{{.*}} = db.between %{{.*}} : i32 between %{{.*}} : i32, %{{.*}} : i32, lowerInclusive : true, upperInclusive : true
--//CHECK: }
--//CHECK:  {{.*}} = relalg.aggregation {{.*}} [@{{.*}}::@{{.*}}] computes : [] (%arg0: !tuples.tuplestream,%arg1: !tuples.tuple){
select x between 0 and 1 from (values (1)) t(x) group by x between 0 and 1;
--//CHECK: module
--//CHECK:  %{{.*}} = relalg.map %{{.*}} computes : [@{{.*}}::@{{.*}}({type = !db.nullable<i32>})] (%arg0: !tuples.tuple){
--//CHECK:      %{{.*}} = relalg.const_relation columns :
--//CHECK:      %{{.*}} = relalg.const_relation columns :
--//CHECK:      %{{.*}} = relalg.crossproduct %{{.*}}, %{{.*}}
--//CHECK: }
--//CHECK:  {{.*}} = relalg.aggregation {{.*}} [@{{.*}}::@{{.*}}] computes : [] (%arg0: !tuples.tuplestream,%arg1: !tuples.tuple){
--//CHECK: }
--//CHECK:  %{{.*}} = relalg.materialize %{{.*}} [@{{.*}}::@{{.*}}] => [""] : !subop.local_table<[{{.*}}$0 : !db.nullable<i32>], [""]>
select (select x from (values (1)) t(x), (values (2)) z(y)) from (values (1)) t(x) group by (select x from (values (1)) t(x), (values (2)) z(y));
--//CHECK: module
--//CHECK:  %{{.*}} = db.cast {{.*}} : i32 -> f64
--//CHECK: }
--//CHECK:  %{{.*}} = relalg.aggregation {{.*}} [@{{.*}}::@{{.*}}] computes : [] (%arg0: !tuples.tuplestream,%arg1: !tuples.tuple){
select x::float from (values (1)) t(x) group by x::float;
--//CHECK: module
--//CHECK:  %{{.*}} = db.runtime_call "Concatenate"({{.*}}, {{.*}}) : (!db.string, !db.string) -> !db.string
--//CHECK:  %{{.*}} = db.runtime_call "ToUpper"({{.*}}) : (!db.string) -> !db.string
--//CHECK: }
--//CHECK:  %{{.*}} = relalg.aggregation %2 [@{{.*}}::@{{.*}}] computes : [@{{.*}}::@{{.*}}({type = i32})] (%arg0: !tuples.tuplestream,%arg1: !tuples.tuple){
--//CHECK-NOT:  %{{.*}} = db.runtime_call "ToUpper"({{.*}}) : (!db.string) -> !db.string
--//CHECK-NOT:  %{{.*}} = db.runtime_call "Concatenate"({{.*}}) : (!db.string) -> !db.string
--//CHECK:      %{{.*}} = relalg.aggrfn min @{{.*}}::@{{.*}} %arg0 : i32
--//CHECK-NOT:  %{{.*}} = relalg.aggrfn min @{{.*}}::@{{.*}} %arg0 : i32
select UPPER(y || 'extra'), min(y) from (values ('Value1', 1), ('VALUE2', 2), ('VALUE3', 3) ) t(x,y) group by UPPER(y || 'extra');
--//CHECK: %[[AGGR1:.*]] = relalg.aggregation
--//CHECK: {{.*}} = relalg.aggrfn sum @{{.*}}::@{{.*}} %arg0 : i32
--//CHECK-NOT: %{{.*}} = relalg.aggrfn sum @{{.*}}::@{{.*}} %arg0 : i32
--//CHECK: %[[AGGR2:.*]] = relalg.aggregation %[[AGGR1]]
--//CHECK: {{.*}} = relalg.aggrfn sum @{{.*}}::@{{.*}} %arg0 : i32
--//CHECK-NOT: %{{.*}} = relalg.aggrfn sum @{{.*}}::@{{.*}} %arg0 : i32
--//CHECK: %[[AGGR3:.*]] = relalg.aggregation %[[AGGR2]]
--//CHECK: {{.*}} = relalg.aggrfn sum @{{.*}}::@{{.*}} %arg0 : i32
--//CHECK-NOT: %{{.*}} = relalg.aggrfn sum @{{.*}}::@{{.*}} %arg0 : i32
--//CHECK: %{{.*}} = relalg.union all
--//CHECK: %{{.*}} = relalg.union all
--//CHECK:  %{{.*}} = relalg.map %{{.*}} computes : [@{{.*}}::@{{.*}}({type = i64})] (%arg0: !tuples.tuple)
--//CHECK:      %{{.*}} = arith.constant 1 : i64
--//CHECK:      %{{.*}} = arith.shrui %{{.*}}, %{{.*}} : i64
--//CHECK:      %{{.*}} = arith.constant 1 : i64
--//CHECK:      %{{.*}} = arith.andi %{{.*}}, %{{.*}} : i64
--//CHECK:      tuples.return %{{.*}} : i64
--//CHECK:  %{{.*}} = relalg.map %{{.*}} computes : [@{{.*}}::@{{.*}}({type = i64})] (%arg0: !tuples.tuple)
--//CHECK:      %{{.*}} = arith.constant 0 : i64
--//CHECK:      %{{.*}} = arith.shrui %{{.*}}, %{{.*}} : i64
--//CHECK:      %{{.*}} = arith.constant 1 : i64
--//CHECK:      %{{.*}} = arith.andi %{{.*}}, %{{.*}} : i64
--//CHECK:      tuples.return %{{.*}} : i64
select x,y, sum(z), grouping(x), grouping(y) from (values (1,2,3)) t(x,y,z) group by rollup(x,y) having  sum(z) > 1 order by x;
--//CHECK: %{{.*}} = db.compare eq {{.*}} : i32, %5 : i32
select x from (values (1), (2), (3)) t(x) where x=1;
--//CHECK: %{{.*}} = db.compare lt {{.*}} : i32, %5 : i32
select x from (values (1), (2), (3)) t(x) where x<1;
--//CHECK: %{{.*}} = db.compare gt {{.*}} : i32, %5 : i32
select x from (values (1), (2), (3)) t(x) where x>1;
--//CHECK: %{{.*}} = db.compare lte {{.*}} : i32, %5 : i32
select x from (values (1), (2), (3)) t(x) where x<=1;
--//CHECK: %{{.*}} = db.compare gte {{.*}} : i32, %5 : i32
select x from (values (1), (2), (3)) t(x) where x>=1;
--//CHECK: %{{.*}} = db.compare neq {{.*}} : i32, %5 : i32
select x from (values (1), (2), (3)) t(x) where x<>1;
--//CHECK: %{{.*}} = relalg.limit 2 {{.*}}
select x from (values (1), (2), (3)) t(x) LIMIT 2;
--//CHECK:  %{{.*}} = relalg.materialize %1 [@{{.*}}::@{{.*}}] => ["y"] : !subop.local_table<[const_u_1$0 : i32], ["y"]>
from (values (1,1), (2,2), (3,3)) t(x,y)
|> select *
|> DROP x;
--//CHECK:   %{{.*}} = relalg.materialize %1 [@{{.*}}::@const,@{{.*}}::@{{.*}},@{{.*}}::@const] => ["x", "y", "x"] : !subop.local_table<[const$0 : i32, const_u_1$0 : i32, const$1 : i32], ["x", "y", "x"]>
from (values (1,1), (2,2), (3,3)) t(x,y)
|> select *
|> EXTEND x;
--//CHECK:  %{{.*}} = relalg.map %1 computes : [@{{.*}}::@{{.*}}({type = i32})] (%arg0: !tuples.tuple){
--//CHECK:  %{{.*}} = db.add %{{.*}} : i32, %{{.*}} : i32
from (values (1,1), (2,2), (3,3)) t(x,y)
|> select *
|> SET y=y+1;
--//CHECK: %{{.*}}  = relalg.limit 1 {{.*}}
--//CHECK  %{{.*}} = relalg.union all
from test
|> LIMIT 1
|> union all (select * from test);
--//CHECK: %[[AGGR1:.*]] = relalg.aggregation
--//CHECK: %[[AGGR2:.*]] = relalg.aggregation %[[AGGR1]]
--//CHECK: %{{.*}} = relalg.aggregation %[[AGGR2]]
--//CHECK: %{{.*}} = relalg.union all
--//CHECK: %{{.*}} = relalg.union all
from (values (1,2,3)) t(x,y,z)
 |> AGGREGATE sum(z) group by rollup(x,y);
--//CHECK:  %{{.*}} = relalg.basetable  {table_identifier = "test"} columns: {}
select 1 from test;
--//CHECK:  %{{.*}} = relalg.basetable  {table_identifier = "test"} columns: {bool => @test::@bool({type = !db.nullable<i1>})}
select bool from test;
--//CHECK:  %{{.*}} = relalg.basetable  {table_identifier = "test"} columns: {int32 => @test::@int32({type = !db.nullable<i32>}), int64 => @test::@int64({type = !db.nullable<i64>})}
select int32 from test where int64 > 100;
--//CHECK-DAG:  %{{.*}} = relalg.basetable  {table_identifier = "test"} columns: {int32 => @test::@int32({type = !db.nullable<i32>})}
--//CHECK-DAG:  %{{.*}} = relalg.basetable  {table_identifier = "test"} columns: {}
select int32 from test union select 1 from test;
--//CHECK-DAG:  %{{.*}} = relalg.basetable  {table_identifier = "test"} columns: {int64 => @t2::@int64({type = !db.nullable<i64>}), str => @t2::@str({type = !db.nullable<!db.string>})}
--//CHECK-DAG:  %{{.*}} = relalg.basetable  {table_identifier = "test"} columns: {decimal => @t1::@decimal({type = !db.nullable<!db.decimal<5, 2>>}), int32 => @t1::@int32({type = !db.nullable<i32>}), int64 => @t1::@int64({type = !db.nullable<i64>})}
select t1.int32, t2.str from test t1 join test t2 on t1.int64 = t2.int64 where t1.decimal > 1.0;
--//CHECK-DAG:  %{{.*}} = relalg.basetable  {table_identifier = "test"} columns: {int32 => @{{.*}}::@int32({type = !db.nullable<i32>}), int64 => @{{.*}}::@int64({type = !db.nullable<i64>})}
--//CHECK-DAG:  %{{.*}} = relalg.basetable  {table_identifier = "test"} columns: {decimal => @{{.*}}::@decimal({type = !db.nullable<!db.decimal<5, 2>>}), int64 => @{{.*}}::@int64({type = !db.nullable<i64>})}
select int32 from test where int32 IN (select int64 from test t where t.decimal=42) and int64 > 10;
--//CHECK:      %{{.*}} = relalg.basetable  {table_identifier = "test"} columns: {bool => @{{.*}}::@bool({type = !db.nullable<i1>})}
--//CHECK:      %{{.*}} = relalg.basetable  {table_identifier = "test"} columns: {}
select t1.bool from test t1 where not exists( select * from test t2);
--//CHECK:      %{{.*}} = relalg.basetable  {table_identifier = "test"} columns: {bool => @{{.*}}::@bool({type = !db.nullable<i1>})}
--//CHECK:      %{{.*}} = relalg.basetable  {table_identifier = "test"} columns: {}
select t1.bool from test t1 where not exists( select t2.bool from test t2);
--//CHECK:      %{{.*}} = relalg.basetable  {table_identifier = "test"} columns: {bool => @{{.*}}::@bool({type = !db.nullable<i1>}), int32 => @{{.*}}::@int32({type = !db.nullable<i32>})}
--//CHECK:      %{{.*}} = relalg.basetable  {table_identifier = "test"} columns: {int32 => @{{.*}}::@int32({type = !db.nullable<i32>})}
select t1.bool from test t1 where not exists( select t2.bool from test t2 where t2.int32=t1.int32);