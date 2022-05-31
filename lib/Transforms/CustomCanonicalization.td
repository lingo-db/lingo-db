include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.td"
//include "mlir/Dialect/DB/IR/DBOps.td"
include "mlir/IR/PatternBase.td"

def ConstAddCmpFold :
    Pat<(Arith_CmpFOp $pred,
          (Arith_AddFOp $a, (Arith_ConstantOp $c2 )),
          (Arith_ConstantOp $c1)),
        (Arith_CmpFOp $pred, $a, (Arith_SubFOp (Arith_ConstantOp $c1), (Arith_ConstantOp $c2)))>;
def ConstMulCmpFold :
    Pat<(Arith_CmpFOp $pred,
          (Arith_MulFOp $a, (Arith_ConstantOp $c2)),
          (Arith_ConstantOp $c1)),
        (Arith_CmpFOp $pred, $a, (Arith_DivFOp (Arith_ConstantOp $c1), (Arith_ConstantOp $c2)))>;