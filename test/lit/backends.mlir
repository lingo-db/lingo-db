// RUN: env LINGODB_EXECUTION_MODE=C env LINGODB_BACKEND_ONLY=ON run-mlir %s | FileCheck %s
// RUN: env LINGODB_EXECUTION_MODE=DEBUGGING env LINGODB_BACKEND_ONLY=ON run-mlir %s | FileCheck %s
// RUN: env LINGODB_EXECUTION_MODE=DEFAULT env LINGODB_BACKEND_ONLY=ON run-mlir %s | FileCheck %s


module  {
  func.func private @dumpString(!util.varlen32)
  func.func private @dumpI64(i64)
  func.func private @dumpF64(f64)
  func.func private @dumpBool(i1)

  func.func @testVarLen(){
	  //CHECK: string("constant string!!!!!")
	  //CHECK: string("short str")
      %varlen_1 = util.varlen32_create_const "constant string!!!!!"
      %varlen_2 = util.varlen32_create_const "short str"
      call @dumpString(%varlen_1) : (!util.varlen32) -> ()
      call @dumpString(%varlen_2) : (!util.varlen32) -> ()
      return
  }
  func.func @testIntArith(){
   //CHECK: int(43)
   //CHECK: int(-41)
   //CHECK: int(84)
   //CHECK: int(21)
   //CHECK: int(0)
  	%c1 = arith.constant 1 : i64
  	%c2 = arith.constant 2 : i64
  	%c42 = arith.constant 42 : i64
  	%add = arith.addi %c1, %c42 : i64
  	%sub = arith.subi %c1, %c42 : i64
  	%mul = arith.muli %c2, %c42 : i64
  	%div = arith.divui %c42, %c2 : i64
  	%mod = arith.remui %c42, %c2 : i64
  	call @dumpI64(%add) : (i64) -> ()
  	call @dumpI64(%sub) : (i64) -> ()
  	call @dumpI64(%mul) : (i64) -> ()
  	call @dumpI64(%div) : (i64) -> ()
  	call @dumpI64(%mod) : (i64) -> ()
  	return
  }
  func.func @testIntCmp(){
    //CHECK: bool(true)
    //CHECK: bool(true)
    //CHECK: bool(false)
    //CHECK: bool(false)
    //CHECK: bool(true)
    //CHECK: bool(false)
    //CHECK: bool(true)
    //CHECK: bool(false)
    //CHECK: bool(false)
    //CHECK: bool(true)
  	%c1 = arith.constant 1 : i64
  	%c2 = arith.constant 2 : i64
  	%ult = arith.cmpi ult, %c1, %c2 : i64
  	%slt = arith.cmpi slt, %c1, %c2 : i64
  	%ugt = arith.cmpi ugt, %c1, %c2 : i64
  	%sgt = arith.cmpi sgt, %c1, %c2 : i64
  	%ule = arith.cmpi ule, %c1, %c2 : i64
  	%uge = arith.cmpi uge, %c1, %c2 : i64
  	%sle = arith.cmpi sle, %c1, %c2 : i64
  	%sge = arith.cmpi sge, %c1, %c2 : i64
  	%eq = arith.cmpi eq, %c1, %c2 : i64
  	%ne = arith.cmpi ne, %c1, %c2 : i64
  	call @dumpBool(%ult) : (i1) -> ()
  	call @dumpBool(%slt) : (i1) -> ()
  	call @dumpBool(%ugt) : (i1) -> ()
  	call @dumpBool(%sgt) : (i1) -> ()
  	call @dumpBool(%ule) : (i1) -> ()
  	call @dumpBool(%uge) : (i1) -> ()
  	call @dumpBool(%sle) : (i1) -> ()
  	call @dumpBool(%sge) : (i1) -> ()
  	call @dumpBool(%eq) : (i1) -> ()
  	call @dumpBool(%ne) : (i1) -> ()
  	return
  }
    func.func @testFloatArith(){
       //CHECK: float(43)
       //CHECK: float(-41)
       //CHECK: float(84)
       //CHECK: float(21)
       //CHECK: float(0)
    	%c1 = arith.constant 1.0 : f64
    	%c2 = arith.constant 2.0 : f64
    	%c42 = arith.constant 42.0 : f64
    	%add = arith.addf %c1, %c42 : f64
    	%sub = arith.subf %c1, %c42 : f64
    	%mul = arith.mulf %c2, %c42 : f64
    	%div = arith.divf %c42, %c2 : f64
    	%mod = arith.remf %c42, %c2 : f64
    	call @dumpF64(%add) : (f64) -> ()
    	call @dumpF64(%sub) : (f64) -> ()
    	call @dumpF64(%mul) : (f64) -> ()
    	call @dumpF64(%div) : (f64) -> ()
    	call @dumpF64(%mod) : (f64) -> ()
    	return
    }
  func.func @testFloatCmp(){
    //CHECK: bool(true)
    //CHECK: bool(false)
    //CHECK: bool(true)
    //CHECK: bool(false)
    //CHECK: bool(false)
    //CHECK: bool(true)
  	%c1 = arith.constant 1.0 : f64
  	%c2 = arith.constant 2.0 : f64
  	%olt = arith.cmpf olt, %c1, %c2 : f64
  	%ogt = arith.cmpf ogt, %c1, %c2 : f64
  	%ole = arith.cmpf ole, %c1, %c2 : f64
  	%oge = arith.cmpf oge, %c1, %c2 : f64
  	%oeq = arith.cmpf oeq, %c1, %c2 : f64
  	%one = arith.cmpf one, %c1, %c2 : f64
  	call @dumpBool(%olt) : (i1) -> ()
  	call @dumpBool(%ogt) : (i1) -> ()
  	call @dumpBool(%ole) : (i1) -> ()
  	call @dumpBool(%oge) : (i1) -> ()
  	call @dumpBool(%oeq) : (i1) -> ()
  	call @dumpBool(%one) : (i1) -> ()
  	return
  }
  func.func @testIf(){
  	%false = arith.constant false
  	%true = arith.constant true
  	%res1, %res2 = scf.if %true -> (i1,i1) {
  		scf.yield %true,%true : i1,i1
  	} else{
  	  	scf.yield %false,%false : i1,i1
  	}
  	//CHECK: bool(true)
    //CHECK: bool(true)
  	call @dumpBool(%res1) : (i1) -> ()
  	call @dumpBool(%res2) : (i1) -> ()
  	%res1_2, %res2_2 = scf.if %false -> (i1,i1) {
  		scf.yield %true,%true : i1,i1
  	} else{
  	  	scf.yield %false,%false : i1,i1
  	}
  	//CHECK: bool(false)
    //CHECK: bool(false)
  	 call @dumpBool(%res1_2) : (i1) -> ()
  	 call @dumpBool(%res2_2) : (i1) -> ()
  	 return
  }
  func.func @testFor(){
  	//CHECK: int(10)
    //CHECK: int(120)
  	%c0 = arith.constant 0 : index
  	%c1 = arith.constant 1 : index
  	%c5 = arith.constant 5 : index
  	%res1, %res2 = scf.for %i=%c0 to %c5 step %c1 iter_args(%curr1=%c0, %curr2=%c1) -> (index,index) {
  		%add= arith.addi %curr1, %i :index
  		%inc1 = arith.addi %i, %c1 : index
  		%mul= arith.muli %curr2,%inc1 :index
  		scf.yield %add, %mul : index,index
  	}
  	%intres1=arith.index_cast %res1 : index to i64
  	%intres2=arith.index_cast %res2 : index to i64
 	call @dumpI64(%intres1) : (i64) -> ()
  	call @dumpI64(%intres2) : (i64) -> ()
	return
  }
  func.func @testWhile(){
	//CHECK: int(10)
    //CHECK: int(120)
	%c0 = arith.constant 0 : index
	%c1 = arith.constant 1 : index
	%c5 = arith.constant 5 : index

	%lastI, %res1, %res2 = scf.while(%i1=%c0,%curr1_i=%c0,%curr2_i=%c1) : (index,index,index) -> (index,index,index){
	 	%lt = arith.cmpi ult, %i1, %c5 :index
	 	scf.condition(%lt) %i1, %curr1_i,%curr2_i : index,index,index
	 }do {
	 ^bb0(%i :index,%curr1:index, %curr2:index):
		%add= arith.addi %curr1, %i :index
		%inc1 = arith.addi %i, %c1 : index
		%mul= arith.muli %curr2,%inc1 :index
		scf.yield %inc1, %add, %mul : index,index,index
	}
	%intres1=arith.index_cast %res1 : index to i64
	%intres2=arith.index_cast %res2 : index to i64
	call @dumpI64(%intres1) : (i64) -> ()
	call @dumpI64(%intres2) : (i64) -> ()
	return
  }
  func.func @get42() -> i64{
  	%c42 = arith.constant 42 : i64
  	return %c42 : i64
  }
  func.func @testReturn(){
  	//CHECK: int(42)
  	%res = call @get42() : () -> i64
    call @dumpI64(%res) : (i64) -> ()
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
//  func.func @testBranch(){
//      %true = arith.constant true
//      %c1 = arith.constant 1 : i64
//      %c2 = arith.constant 2 : i64
//      cf.cond_br %true, ^bb1(%c1 : i64), ^bb1(%c2 : i64)
//    ^bb1(%x : i64) :
//      cf.br ^bb2(%x : i64)
//    ^bb2(%x2 : i64) :
//       call @dumpI64(%x2) : (i64) -> ()
//       return
//  }
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

		return
	}

  func.func @main() {
	call @testVarLen() : () -> ()
	call @testIntArith() : () -> ()
	call @testIntCmp() : () -> ()
	call @testFloatArith() : () -> ()
	call @testFloatCmp() : () -> ()
	call @testIf() : () -> ()
	call @testFor() : () -> ()
	call @testWhile() : () -> ()
	//call @testBranch() : () -> ()
	call @testReturn() : () -> ()
	call @testMultiReturn() : () -> ()
	call @testTuple() : () -> ()
	call @testRef() : () -> ()
    return
  }
}