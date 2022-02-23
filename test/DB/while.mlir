// RUN: db-run %s | FileCheck %s

module  {
  func @main() {
    %c0 = db.constant ( 0 ) : !db.int<32>
    %c0n = db.cast %c0 : !db.int<32> -> !db.nullable<!db.int<32>>

    %c1 = db.constant ( 1 ) : !db.int<32>
    %c10 = db.constant ( 10 ) : !db.int<32>
    %2:2 = db.while (%arg1 = %c0n, %arg2 = %c0) : (!db.nullable<!db.int<32>>, !db.int<32>) -> (!db.nullable<!db.int<32>>, !db.int<32>) {
      %4 = db.compare lt %arg1 : !db.nullable<!db.int<32>> , %c10 : !db.int<32>
      db.condition(%4 : !db.nullable<!db.bool>) %arg1, %arg2 : !db.nullable<!db.int<32>>, !db.int<32>
    } do {
    ^bb0(%arg1: !db.nullable<!db.int<32>>, %arg2: !db.int<32>):  // no predecessors
      %4 = db.add %arg1 : !db.nullable<!db.int<32>>, %c1 : !db.int<32>
      %5 = db.add %arg2 : !db.int<32>, %c1 : !db.int<32>
      db.yield %4, %5 : !db.nullable<!db.int<32>>, !db.int<32>
    }
    //CHECK: int(10)
    db.dump %2#1 : !db.int<32>
    return
  }
}