// RUN: mlir-db-opt %s -subop-organize-execution-steps -mlir-print-local-scope | FileCheck %s
//
// Two parallel pipelines joined by a `subop.union`, with a downstream
// `subop.materialize` consuming the unioned stream. The materialize is
// multi-rooted (downstream of the union), so OrganizeExecutionStepsPass:
//   - clones it into the earlier pipeline,
//   - moves the original into the later pipeline,
//   - rewires each clone's tuple-stream operand past the union (via
//     resolveStreamForPipeline),
//   - erases the now-unused union.

!buf = !subop.buffer<[v : i32]>
!entry_ref = !subop.entry_ref<!buf>

func.func @union_split() {
  %dst  = subop.create !subop.buffer<[v : i32]>
  %src1 = subop.create !subop.buffer<[v : i32]>
  %src2 = subop.create !subop.buffer<[v : i32]>

  %s1 = subop.scan %src1 : !buf {v => @c1::@v({type = i32})}
  %s2 = subop.scan %src2 : !buf {v => @c2::@v({type = i32})}
  // Project both branches' columns into a common name so the union is
  // type-compatible:
  %m1 = subop.map %s1 computes : [@u::@v({type = i32})] input : [@c1::@v] (%a: i32){
    tuples.return %a : i32
  }
  %m2 = subop.map %s2 computes : [@u::@v({type = i32})] input : [@c2::@v] (%a: i32){
    tuples.return %a : i32
  }
  %u  = subop.union %m1, %m2
  subop.materialize %u {@u::@v => v}, %dst : !subop.buffer<[v : i32]>

  return
}

// Each parallel pipeline ends up with its own materialize (one cloned, one
// moved). The union is gone.

// CHECK-LABEL: func.func @union_split
// CHECK:       subop.execution_group
// CHECK-NOT:   subop.union

// Materialize must appear twice — once per branch of the eliminated union.
// CHECK-COUNT-2: subop.materialize
