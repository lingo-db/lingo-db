include "mlir/Dialect/DB/IR/DBOps.td"
include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.td"

def EliminateNullCmp :
    Pat<(DB_CmpOp $pred,$a,$b),
        (DB_CombineNullOp (DB_CmpOp $pred, (DB_NullableGetVal $a), (DB_NullableGetVal $b)), (Arith_OrIOp (DB_IsNullOp $a), (DB_IsNullOp $b) )),
        [(Constraint<CPred<"($0.getType().isa<mlir::db::NullableType>() || $1.getType().isa<mlir::db::NullableType>()) && getBaseType($0.getType()).isIntOrIndexOrFloat()">> $a, $b)]>;

def EliminateNullAdd :
    Pat<(DB_AddOp $a,$b),
        (DB_CombineNullOp  (DB_AddOp (DB_NullableGetVal $a), (DB_NullableGetVal $b)), (Arith_OrIOp (DB_IsNullOp $a), (DB_IsNullOp $b) )),
        [(Constraint<CPred<"($0.getType().isa<mlir::db::NullableType>() || $1.getType().isa<mlir::db::NullableType>()) && getBaseType($0.getType()).isIntOrIndexOrFloat()">> $a, $b)]>;
def EliminateIsNull :
    Pat<(DB_IsNullOp (DB_CombineNullOp $val, $null) ), (replaceWithValue $null)>;
