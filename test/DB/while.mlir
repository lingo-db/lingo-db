// RUN: db-run %s | FileCheck %s

module  {
  func @main() {
    %c0 = db.constant ( 0 ) : i32
    %c0n = db.cast %c0 : i32 -> !db.nullable<i32>

    %c1 = db.constant ( 1 ) : i32
    %c10 = db.constant ( 10 ) : i32
    %2:2 = db.while (%arg1 = %c0n, %arg2 = %c0) : (!db.nullable<i32>, i32) -> (!db.nullable<i32>, i32) {
      %4 = db.compare lt %arg1 : !db.nullable<i32> , %c10 : i32
      db.condition(%4 : !db.nullable<i1>) %arg1, %arg2 : !db.nullable<i32>, i32
    } do {
    ^bb0(%arg1: !db.nullable<i32>, %arg2: i32):  // no predecessors
      %4 = db.add %arg1 : !db.nullable<i32>, %c1 : i32
      %5 = db.add %arg2 : i32, %c1 : i32
      db.yield %4, %5 : !db.nullable<i32>, i32
    }
    //CHECK: int(10)
    db.dump %2#1 : i32
    return
  }
}