// RUN: mlir-db-opt -allow-unregistered-dialect %s -split-input-file -mlir-print-debuginfo -mlir-print-local-scope  | FileCheck %s
!c_v = !subop.continuous_view<!subop.buffer<[val : index]>>
!c_v_e_r = !subop.continous_entry_ref<!c_v>
//CHECK: %{{.*}} = subop.create !subop.buffer<[val : index]>
%vals = subop.create!subop.buffer<[val : index]>
//CHECK: %{{.*}} =  subop.create_continuous_view %{{.*}} : !subop.buffer<[val : index]> -> <!subop.buffer<[val : index]>>
%view = subop.create_continuous_view %vals : !subop.buffer<[val : index]> -> !c_v
//CHECK: %{{.*}} = subop.create_segment_tree_view %{{.*}} : !subop.continuous_view<!subop.buffer<[val : index]>> -> !subop.segment_tree_view<[from : !subop.continous_entry_ref<!subop.continuous_view<!subop.buffer<[val : index]>>>, to : !subop.continous_entry_ref<!subop.continuous_view<!subop.buffer<[val : index]>>>], [sum : index]> initial["val"]:(%arg0){
//CHECK:              tuples.return %arg0 : index
//CHECK:            }combine: ([%arg0],[%arg1]){
//CHECK:              %{{.*}} = arith.addi %arg0, %arg1 : index
//CHECK:             tuples.return %{{.*}} : index
//CHECK:           }

%segment_tree_view = subop.create_segment_tree_view %view :  !c_v -> !subop.segment_tree_view<[from : !c_v_e_r, to : !c_v_e_r],[sum : index]>
						initial["val"]:(%val){
							tuples.return %val : index
						}
						combine: ([%left],[%right]){
							%added = arith.addi %left, %right : index
							tuples.return %added : index
						}

// -----
%c0 = arith.constant 0 : i64
//CHECK: %{{.*}} = subop.create_simple_state <[sum : i64]> initial : {
//CHECK:                      %{{.*}} = arith.constant 0 : i64
//CHECK:                      tuples.return %{{.*}} : i64
//CHECK:                    }

%state = subop.create_simple_state !subop.simple_state<[sum:i64]> initial: {
 %c00 = arith.constant 0 : i64
tuples.return %c00 : i64
}
%initial_stream=subop.in_flight %c0 : i64 => [@t::@col({type=i64})]
//CHECK: %{{.*}} = subop.lookup %{{.*}}%{{.*}} [] : !subop.simple_state<[sum : i64]> @state::@ref({type = !subop.lookup_entry_ref<!subop.simple_state<[sum : i64]>>})
%stream = subop.lookup %initial_stream %state[] : !subop.simple_state<[sum:i64]> @state::@ref({type=!subop.lookup_entry_ref<!subop.simple_state<[sum:i64]> >})
//CHECK:   subop.reduce %{{.*}}  @state::@ref[@t::@col] ["sum"] ([%arg0],[%arg1]){
//CHECK:     %{{.*}}  = arith.addi %arg0, %arg1 : i64
//CHECK:     tuples.return %{{.*}}  : i64
//CHECK:   }combine: ([%arg0],[%arg1]){
//CHECK:     %{{.*}}  = arith.addi %arg0, %arg1 : i64
//CHECK:     tuples.return %{{.*}}  : i64
//CHECK:   }

subop.reduce %stream @state::@ref [@t::@col] ["sum"] ([%curr],[%val]){
	%next = arith.addi %curr, %val : i64
	tuples.return %next : i64
} combine: ([%left],[%right]){
	%added = arith.addi %left, %right : i64
	tuples.return %added : i64
}

// -----
//CHECK: %{{.*}} = subop.create_heap["ih"] -> !subop.heap<4, [ih : index]> ([%arg0],[%arg1]){
//CHECK:   %{{.*}} = arith.cmpi ult, %arg0, %arg1 : index
//CHECK:   tuples.return %{{.*}} : i1
//CHECK: }

%heap = subop.create_heap ["ih"] -> !subop.heap<4,[ih : index]> ([%left],[%right]){
	%lt = arith.cmpi ult, %left, %right : index
	tuples.return %lt : i1
}
// -----
%vals = subop.create !subop.buffer<[val : index]>
//CHECK:  %{{.*}} = subop.create_sorted_view %{{.*}} : !subop.buffer<[val : index]> ["val"] ([%arg0],[%arg1]){
//CHECK:    %{{.*}} = arith.cmpi ult, %arg0, %arg1 : index
//CHECK:    tuples.return %{{.*}} : i1
//CHECK:  }

%sorted_view = subop.create_sorted_view %vals : !subop.buffer<[val : index]> ["val"] ([%left],[%right]){
	%lt = arith.cmpi ult, %left, %right : index
	tuples.return %lt : i1
}
// -----
//CHECK:  %{{.*}} = subop.generate[@t::@c1({type = index})]{
//CHECK:    %{{.*}} = arith.constant 0 : index
//CHECK:    subop.generate_emit %{{.*}} : index
//CHECK:    tuples.return
//CHECK:  }

%generated, %streams = subop.generate [@t::@c1({type=index})] {
	%c0 = arith.constant 0 : index
	subop.generate_emit %c0 : index
	tuples.return
}
// -----
%c0 = arith.constant 0 : i64
%initial_stream=subop.in_flight %c0 : i64 => [@t::@col({type=i64})]
%map = subop.create !subop.map<[key : i64],[val : i64] >
//CHECK: %{{.*}} = subop.lookup_or_insert %{{.*}}%{{.*}} [@t::@col] : !subop.map<[key : i64], [val : i64] > @state::@ref({type = !subop.lookup_entry_ref<!subop.map<[key : i64], [val : i64] >>}) eq: ([%arg0],[%arg1]) {
//CHECK:     %{{.*}} = arith.cmpi eq, %arg0, %arg1 : i64
//CHECK:     tuples.return %{{.*}} : i1
//CHECK:   }initial: {
//CHECK:     %{{.*}} = arith.constant 0 : i64
//CHECK:     tuples.return %{{.*}} : i64
//CHECK:   }
//CHECK: }

%stream =subop.lookup_or_insert %initial_stream %map[@t::@col] : !subop.map<[key : i64],[val : i64] > @state::@ref({type=!subop.lookup_entry_ref<!subop.map<[key : i64],[val : i64] >>})
eq: ([%l], [%r]){
	%eq = arith.cmpi eq, %l, %r :i64
	tuples.return %eq : i1
}
initial: {
	%c00 = arith.constant 0 : i64
	tuples.return %c00 : i64
}
// -----
%c0 = arith.constant 0 : i64
%initial_stream=subop.in_flight %c0 : i64 => [@t::@col({type=i64})]
%map = subop.create !subop.map<[key : i64],[val : i64] >
//CHECK:   subop.insert %{{.*}}%{{.*}}  : !subop.map<[key : i64], [val : i64] > {@t::@col => key, @t::@col => val}
subop.insert %initial_stream %map : !subop.map<[key : i64],[val : i64] > {@t::@col => key,@t::@col => val}
//CHECK:  subop.insert %{{.*}}%{{.*}}  : !subop.map<[key : i64], [val : i64] > {@t::@col => key, @t::@col => val} eq: ([%arg0],[%arg1]) {
//CHECK:    %{{.*}} = arith.cmpi eq, %arg0, %arg1 : i64
//CHECK:    tuples.return %{{.*}} : i1
//CHECK:  }

subop.insert %initial_stream %map : !subop.map<[key : i64],[val : i64] > {@t::@col => key,@t::@col => val}
eq: ([%l], [%r]){
	%eq = arith.cmpi eq, %l, %r :i64
	tuples.return %eq : i1
}