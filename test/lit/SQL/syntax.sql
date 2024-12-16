--// RUN: sql-to-mlir %s %S/../../../resources/data/test | FileCheck %s
--//CHECK: module {
--//CHECK:     func.func @main() {
--//CHECK:         %{{.*}} = relalg.const_relation columns : [@dummyScope::@dummyName({type = i32})] values :
--//CHECK:         %{{.*}} = relalg.map %{{.*}} computes : [@map0::@tmp_attr0({type = i32})] (%arg0: !tuples.tuple){
--//CHECK:             %{{.*}} = db.constant(1 : i32) : i32
--//CHECK:             tuples.return %{{.*}} : i32
--//CHECK:         }
--//CHECK:         %{{.*}} = relalg.materialize %{{.*}} [@map0::@tmp_attr0] => [""] : !subop.local_table<[unnamed$0 : i32], [""]>
--//CHECK:         subop.set_result 0 %{{.*}} : !subop.local_table<[unnamed$0 : i32], [""]>
--//CHECK:         return
--//CHECK:     }
--//CHECK: }

select 1;
--//CHECK: module {
--//CHECK:     func.func @main() {
--//CHECK:         %{{.*}} = relalg.basetable  {table_identifier = "test"} columns: {bool => @test::@bool({type = !db.nullable<i1>}), date32 => @test::@date32({type = !db.nullable<!db.date<day>>}), date64 => @test::@date64({type = !db.nullable<!db.date<millisecond>>}), decimal => @test::@decimal({type = !db.nullable<!db.decimal<5, 2>>}), float32 => @test::@float32({type = !db.nullable<f32>}), float64 => @test::@float64({type = !db.nullable<f64>}), int32 => @test::@int32({type = !db.nullable<i32>}), int64 => @test::@int64({type = !db.nullable<i64>}), str => @test::@str({type = !db.nullable<!db.string>})}
--//CHECK:         %{{.*}} = relalg.materialize %{{.*}} [@test::@str,@test::@float32,@test::@float64,@test::@decimal,@test::@int32,@test::@int64,@test::@bool,@test::@date32,@test::@date64] => ["str", "float32", "float64", "decimal", "int32", "int64", "bool", "date32", "date64"] : !subop.local_table<[str$0 : !db.nullable<!db.string>, float32$0 : !db.nullable<f32>, float64$0 : !db.nullable<f64>, decimal$0 : !db.nullable<!db.decimal<5, 2>>, int32$0 : !db.nullable<i32>, int64$0 : !db.nullable<i64>, bool$0 : !db.nullable<i1>, date32$0 : !db.nullable<!db.date<day>>, date64$0 : !db.nullable<!db.date<millisecond>>], ["str", "float32", "float64", "decimal", "int32", "int64", "bool", "date32", "date64"]>
--//CHECK:         subop.set_result 0 %{{.*}} : !subop.local_table<[str$0 : !db.nullable<!db.string>, float32$0 : !db.nullable<f32>, float64$0 : !db.nullable<f64>, decimal$0 : !db.nullable<!db.decimal<5, 2>>, int32$0 : !db.nullable<i32>, int64$0 : !db.nullable<i64>, bool$0 : !db.nullable<i1>, date32$0 : !db.nullable<!db.date<day>>, date64$0 : !db.nullable<!db.date<millisecond>>], ["str", "float32", "float64", "decimal", "int32", "int64", "bool", "date32", "date64"]>
--//CHECK:         return
--//CHECK:     }
--//CHECK: }

select * from test;
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
--//CHECK: %{{.*}} = db.constant(2 : i32) : i32
--//CHECK: %{{.*}} = db.constant(1 : i32) : i32
--//CHECK: %{{.*}} = db.oneof %{{.*}} : i32 ? %{{.*}}, %{{.*}} : i32, i32
select 1 in (1,2);
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
--//TODO: %{{.*}} = db.constant("1.2") : f64
--select '1.2'::float(32);
--//CHECK: module
--//CHECK: %{{.*}} = db.constant(42 : i32) : i32
--//CHECK: %{{.*}} = db.compare gt %{{.*}} : i32, %{{.*}} : i32
--//CHECK: %{{.*}} = db.derive_truth %{{.*}} : i1
--//CHECK: %{{.*}} = scf.if %{{.*}} -> (i32) {
--//CHECK:   %{{.*}} = tuples.getcol %arg0 @constrel{{.*}}::@const0 : i32
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
--//CHECK:   %{{.*}} = tuples.getcol %arg0 @constrel{{.*}}::@const0 : i32
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
--//CHECK: %{{.*}} = db.null : <i32>
--//CHECK: %{{.*}} = db.isnull %{{.*}} : <i32>
--//CHECK: %{{.*}} = db.not %{{.*}} : i1
--//CHECK: %{{.*}} = scf.if %{{.*}} -> (!db.nullable<i32>) {
--//CHECK:   scf.yield %{{.*}} : !db.nullable<i32>
--//CHECK: } else {
--//CHECK:   %{{.*}} = db.constant(1 : i32) : i32
--//CHECK:   %false = arith.constant false
--//CHECK:   %{{.*}} = db.not %false : i1
--//CHECK:   %{{.*}} = scf.if %{{.*}} -> (!db.nullable<i32>) {
--//CHECK:     %{{.*}} = db.as_nullable %{{.*}} : i32 -> <i32>
--//CHECK:     scf.yield %{{.*}} : !db.nullable<i32>
--//CHECK:   } else {
--//CHECK:     %{{.*}} = db.null : <i32>
--//CHECK:     scf.yield %{{.*}} : !db.nullable<i32>
--//CHECK:   }
--//CHECK:   scf.yield %{{.*}} : !db.nullable<i32>
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
--//CHECK-DAG:   %{{.*}} = relalg.count %arg0
--//CHECK-DAG:   %{{.*}} = relalg.aggrfn count @constrel{{.*}}::@const0 %arg0 : i64
--//CHECK-DAG:   %{{.*}} = relalg.aggrfn sum @constrel{{.*}}::@const0 %arg0 : !db.nullable<i32>
--//CHECK-DAG:   %{{.*}} = relalg.aggrfn max @constrel{{.*}}::@const0 %arg0 : !db.nullable<i32>
--//CHECK-DAG:   %{{.*}} = relalg.aggrfn min @constrel{{.*}}::@const0 %arg0 : !db.nullable<i32>
--//CHECK:   tuples.return 
--//CHECK: }
select min(x),max(x),sum(x),count(x), count(*)  from (values (1)) t(x);
--//CHECK: module
--//CHECK: %{{.*}} = relalg.aggregation %{{.*}} [@constrel{{.*}}::@const0] computes : 
--//CHECK-DAG:       %{{.*}} = relalg.count %arg0
--//CHECK-DAG:       %{{.*}} = relalg.aggrfn count @constrel{{.*}}::@const0 %arg0 : i64
--//CHECK-DAG:       %{{.*}} = relalg.aggrfn sum @constrel{{.*}}::@const0 %arg0 : i32
--//CHECK-DAG:       %{{.*}} = relalg.aggrfn max @constrel{{.*}}::@const0 %arg0 : i32
--//CHECK-DAG:       %{{.*}} = relalg.aggrfn min @constrel{{.*}}::@const0 %arg0 : i32
--//CHECK:       tuples.return 
--//CHECK: }
select y, min(x),max(x),sum(x),count(x), count(*) from (values (1,2)) t(x,y) group by x;
--//CHECK: module
--//CHECK: call @_ZN7lingodb7runtime14RelationHelper11createTableEPNS0_16ExecutionContextENS0_8VarLen32ES4_(%{{.*}}, %{{.*}}, %{{.*}}) : (!util.ref<i8>, !util.varlen32, !util.varlen32) -> ()
create table test(
                     str varchar(20),
                     float32 float(2),
                     float64 float(40),
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
--//CHECK: call @_ZN7lingodb7runtime14RelationHelper21appendTableFromResultENS0_8VarLen32EPNS0_16ExecutionContextEm(%{{.*}}, %{{.*}}, %{{.*}}) : (!util.varlen32, !util.ref<i8>, i64) -> ()
INSERT into test(str, float32, float64, decimal, int32, int64, bool, date32, date64) values ('str', 1.1, 1.1, 1.10, 1, 1, 1, '1966-01-02', '1996-01-02'), (null, null, null, null, null, null, null, null, null);
--//CHECK: module
--//CHECK: %{{.*}} = util.varlen32_create_const "test"
--//CHECK: %{{.*}} = util.varlen32_create_const "t.csv"
--//CHECK: %{{.*}} = util.varlen32_create_const "|"
--//CHECK: %{{.*}} = util.varlen32_create_const "\\"
--//CHECK: %{{.*}} = call @rt_get_execution_context() : () -> !util.ref<i8>
--//CHECK: call @_ZN7lingodb7runtime14RelationHelper17copyFromIntoTableEPNS0_16ExecutionContextENS0_8VarLen32ES4_S4_S4_(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!util.ref<i8>, !util.varlen32, !util.varlen32, !util.varlen32, !util.varlen32) -> ()
copy test from 't.csv' csv escape '\' delimiter '|' null '';
--//CHECK: %{{.*}} = relalg.aggregation %{{.*}} [@constrel{{.*}}::@const0] computes : [@aggr2::@tmp_attr31({type = i32})] (%arg0: !tuples.tuplestream,%arg1: !tuples.tuple){
--//CHECK:       %{{.*}} = relalg.projection distinct [@constrel{{.*}}::@const1] %arg0
--//CHECK:       %{{.*}} = relalg.aggrfn sum @constrel{{.*}}::@const1 %{{.*}} : i32
--//CHECK:       tuples.return %{{.*}} : i32
--//CHECK: }
select x,sum(distinct y) from (values (1,2)) t(x,y) group by x;
--//CHECK: %{{.*}} = relalg.sort %{{.*}} [(@constrel{{.*}}::@const0,asc),(@constrel{{.*}}::@const1,desc),(@constrel{{.*}}::@const2,asc)]
select * from (values (1,2,3)) t(x,y,z) order by x, y desc, z asc;
--//CHECK: %{{.*}} = relalg.aggregation
--//CHECK: %{{.*}} = relalg.aggregation
--//CHECK: %{{.*}} = relalg.aggregation
--//CHECK: %{{.*}} = relalg.union all
--//CHECK: %{{.*}} = relalg.union all
select x,y,sum(z) from (values (1,2,3)) t(x,y,z) group by rollup(x,y);
--//CHECK: %{{.*}} = relalg.window %{{.*}} partitionBy : [@constrel{{.*}}::@const1] orderBy : [(@constrel{{.*}}::@const2,asc)] rows_between : -9223372036854775808 and 0 computes : [@window0::@tmp_attr33({type = !db.nullable<i32>})] (%arg0: !tuples.tuplestream,%arg1: !tuples.tuple){
--//CHECK:       %{{.*}} = relalg.aggrfn sum @constrel{{.*}}::@const0 %arg0 : !db.nullable<i32>
--//CHECK:       tuples.return %{{.*}} : !db.nullable<i32>
--//CHECK: }
select sum(x) over (partition by y order by z) from (values (1,2,3)) t(x,y,z);
--//CHECK: %{{.*}} = relalg.window %{{.*}} partitionBy : [@constrel{{.*}}::@const1] orderBy : [(@constrel{{.*}}::@const2,asc)] rows_between : -9223372036854775808 and 0 computes : [@window1::@tmp_attr34({type = !db.nullable<i32>})] (%arg0: !tuples.tuplestream,%arg1: !tuples.tuple){
--//CHECK:       %{{.*}} = relalg.aggrfn sum @constrel{{.*}}::@const0 %arg0 : !db.nullable<i32>
--//CHECK:       tuples.return %{{.*}} : !db.nullable<i32>
--//CHECK: }
select sum(x) over (partition by y order by z rows between unbounded preceding and current row) from (values (1,2,3)) t(x,y,z);
--//CHECK: %{{.*}} = relalg.window %{{.*}} partitionBy : [@constrel{{.*}}::@const1] orderBy : [(@constrel{{.*}}::@const2,asc)] rows_between : -100 and 100 computes : [@window2::@tmp_attr35({type = !db.nullable<i32>})] (%arg0: !tuples.tuplestream,%arg1: !tuples.tuple){
--//CHECK:       %{{.*}} = relalg.aggrfn sum @constrel{{.*}}::@const0 %arg0 : !db.nullable<i32>
--//CHECK:       tuples.return %{{.*}} : !db.nullable<i32>
--//CHECK: }
select sum(x) over (partition by y order by z rows between 100 preceding and 100 following) from (values (1,2,3)) t(x,y,z);
--//CHECK: %{{.*}}  = relalg.union distinct
select * from test union select * from test;
--//CHECK: %{{.*}}  = relalg.union all
select * from test union all select * from test;
--//CHECK: %{{.*}}  = relalg.outerjoin
select * from test t1 left outer join test t2 on t1.bool=t2.bool;
--//CHECK: %{{.*}}  = relalg.outerjoin
select * from test t1 right outer join test t2 on t1.bool=t2.bool;
--//CHECK: %{{.*}}  = relalg.fullouterjoin
select * from test t1 full outer join test t2 on t1.bool=t2.bool;
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
--//CHECK: %{{.*}} = relalg.selection %{{.*}} (%arg1: !tuples.tuple){
--//CHECK:         %{{.*}} = db.constant(1 : i32) : i32
--//CHECK:         %{{.*}} = tuples.getcol %arg1 @map
--//CHECK:         %{{.*}} = db.compare eq %{{.*}} : i32, %{{.*}} : i32
--//CHECK:         %{{.*}} = db.not %{{.*}} : i1
--//CHECK:         tuples.return %{{.*}} : i1
--//CHECK:       }
--//CHECK: %{{.*}} = relalg.exists %{{.*}}
--//CHECK: %{{.*}} = db.not %{{.*}} : i1
select 1=all(select 1);
--//CHECK: %{{.*}} = relalg.selection %{{.*}} (%arg1: !tuples.tuple){
--//CHECK:   %{{.*}} = db.constant(1 : i32) : i32
--//CHECK:   %{{.*}} = tuples.getcol %arg1 @map
--//CHECK:   %{{.*}} = db.compare eq %{{.*}} : i32, %{{.*}} : i32
--//CHECK:   tuples.return %{{.*}} : i1
--//CHECK: }
--//CHECK: %{{.*}} = relalg.exists %{{.*}}
select 1=any(select 1);
--//CHECK: call @_ZN7lingodb7runtime14RelationHelper10setPersistEPNS0_16ExecutionContextEb(%{{.*}}, %true) : (!util.ref<i8>, i1) -> ()
set persist=1;