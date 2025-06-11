// RUN: env LINGODB_EXECUTION_MODE=DEFAULT run-mlir %s | FileCheck %s
// RUN: if [ "$(uname)" = "Linux" ]; then env LINGODB_EXECUTION_MODE=BASELINE run-mlir %s | FileCheck %s; fi

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
