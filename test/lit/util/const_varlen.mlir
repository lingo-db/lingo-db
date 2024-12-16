// RUN: run-mlir %s | FileCheck %s

//CHECK: string("constant string!!!!!")
//CHECK: string("short str")
module  {
  func.func private @dumpString(!util.varlen32)

  func.func @main() {
    %varlen_1 = util.varlen32_create_const "constant string!!!!!"

    call @dumpString(%varlen_1) : (!util.varlen32) -> ()
    %varlen_2 = util.varlen32_create_const "short str"

    call @dumpString(%varlen_2) : (!util.varlen32) -> ()
    return
  }
}

