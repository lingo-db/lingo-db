// RUN: LINGODB_EXECUTION_MODE=DEBUGGING LINGODB_BACKEND_ONLY=ON run-mlir %s | FileCheck %s
// RUN: LINGODB_EXECUTION_MODE=DEFAULT LINGODB_BACKEND_ONLY=ON run-mlir %s | FileCheck %s
// RUN: LINGODB_EXECUTION_MODE=C LINGODB_BACKEND_ONLY=ON run-mlir %s | FileCheck %s

module  {
    func.func private @dumpString(!util.varlen32)
    func.func private @dumpI64(i64)
    func.func private @dumpF64(f64)
    func.func private @dumpBool(i1)

    func.func @testFloatArithAdvanced(){
        //CHECK: float(0)
    	%c2 = arith.constant 2.0 : f64
    	%c42 = arith.constant 42.0 : f64
        %mod = arith.remf %c42, %c2 : f64
        call @dumpF64(%mod) : (f64) -> ()
        return
    }
    func.func @getTwoResults() -> (i64,i64) {
        %c7 = arith.constant 7 : i64
        %c42 = arith.constant 42 : i64
        return %c42,%c7 : i64,i64
    }
    func.func @testMultiReturn(){
        //CHECK: int(42)
        //CHECK: int(7)
        %res1,%res2 = call @getTwoResults() : () -> (i64,i64)
        call @dumpI64(%res1) : (i64) -> ()
        call @dumpI64(%res2) : (i64) -> ()
        return
    }
    func.func @testTuple(){
        //CHECK: bool(true)
        //CHECK: bool(false)
        %false = arith.constant false
        %true = arith.constant true
        %5 = util.pack %true, %false : i1, i1 -> tuple<i1, i1>
        //%6,%7 = util.unpack %5 : tuple<i1, i1> -> i1, i1
        %6 = util.get_tuple %5[0] : (tuple<i1, i1>) -> i1
        %7 = util.get_tuple %5[1] : (tuple<i1, i1>) -> i1
        call @dumpBool(%6) : (i1) ->()
        call @dumpBool(%7) : (i1) ->()
        return
    }
    func.func @testRef(){
        //CHECK: int(7)
        //CHECK: int(42)
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        %ref=util.alloc(%c2) : !util.ref<i64>
        %c7 = arith.constant 7 : i64
        %c42 = arith.constant 42 : i64
        util.store %c7:i64, %ref[] : !util.ref<i64>
        util.store %c42:i64, %ref[%c1] : !util.ref<i64>
        %l0 = util.load %ref[] : !util.ref<i64> -> i64
        %l1 = util.load %ref[%c1] : !util.ref<i64> -> i64
        call @dumpI64(%l0) : (i64) -> ()
        call @dumpI64(%l1) : (i64) -> ()
        //convert to memref
        //CHECK: int(7)
        //CHECK: int(14)
        %memref = util.to_memref %ref : !util.ref<i64> -> memref<i64>
        %before = memref.atomic_rmw "addi" %c7, %memref[] : (i64, memref<i64>) -> i64
        %after = util.load %ref[] : !util.ref<i64> -> i64
        call @dumpI64(%before) : (i64) -> ()
        call @dumpI64(%after) : (i64) -> ()
        util.dealloc %ref : !util.ref<i64>
        return
    }
    func.func @main() {
        call @testFloatArithAdvanced() : () -> ()
        call @testMultiReturn() : () -> ()
        call @testTuple() : () -> ()
        call @testRef() : () -> ()
        return
    }
}
