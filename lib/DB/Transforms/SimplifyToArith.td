include "mlir/Dialect/DB/IR/DBOps.td"
include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.td"
include "mlir/IR/PatternBase.td"

def ToCmpIP : NativeCodeCall<"convertToCmpIPred($_builder,$0)">;
def ToCmpFP : NativeCodeCall<"convertToCmpFPred($_builder,$0)">;
def ConvertConst : NativeCodeCall<"convertConst($0,$1)">;

def DBCmpToCmpI :
    Pat<(DB_CmpOp $pred,$a,$b), (Arith_CmpIOp (ToCmpIP $pred), $a, $b),
        [(Constraint<CPred<"$0.getType().isIntOrIndex()&&$1.getType().isIntOrIndex()">> $a, $b)]>;

def DBCmpToCmpF :
    Pat<(DB_CmpOp $pred,$a,$b), (Arith_CmpFOp (ToCmpFP $pred), $a, $b),
        [(Constraint<CPred<"$0.getType().isa<mlir::FloatType>()&&$1.getType().isa<mlir::FloatType>()">> $a, $b)]>;

def DBAddToAddI :
    Pat<(DB_AddOp $a,$b), (Arith_AddIOp $a, $b),
        [(Constraint<CPred<"$0.getType().isIntOrIndex()&&$1.getType().isIntOrIndex()">> $a, $b)]>;

def DBAddToAddF :
    Pat<(DB_AddOp $a,$b), (Arith_AddFOp $a, $b),
        [(Constraint<CPred<"$0.getType().isa<mlir::FloatType>()&&$1.getType().isa<mlir::FloatType>()">> $a, $b)]>;
def DBConstToConst:
    Pat<(DB_ConstantOp:$r $attr), (Arith_ConstantOp (ConvertConst $attr, $r)), [(Constraint<CPred<"$0.getType().isIntOrIndexOrFloat()">> $r)]>;