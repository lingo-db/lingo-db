#ifndef LINGODB_DIALECT_TupleStream_IR_OPS
#define LINGODB_DIALECT_TupleStream_IR_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "lingodb/compiler/Dialect/TupleStream/TupleStreamBase.td"


class TupleStream_Op<string mnemonic, list<Trait> traits = []> : Op<TupleStream_Dialect, mnemonic, traits> {
  let hasCustomAssemblyFormat = 1;

}
////////////////////////////////////////////////////////////////////////////////////////////////////
// Terminators
////////////////////////////////////////////////////////////////////////////////////////////////////

def TupleStream_ReturnOp : TupleStream_Op<"return", [Pure,ReturnLike,Terminator]> {
    let summary = "loop yield and termination operation";

    let arguments = (ins Variadic<AnyType>:$results);
    let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
    let assemblyFormat = "attr-dict ($results^ `:` type($results))?";
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility
////////////////////////////////////////////////////////////////////////////////////////////////////

def GetColumnOp : TupleStream_Op<"getcol", [Pure]> {
    let summary = "get column operation";

    let arguments = (ins ColumnRefAttr:$attr,TupleStream_Tuple:$tuple);
    let results = (outs AnyType : $res);
    let assemblyFormat = [{  $tuple custom<CustRef>($attr) `:` type($res) attr-dict }];
}

def GetParamVal: TupleStream_Op<"get_param_val", [Pure]> {
    let summary = "get param value operation";
    let arguments = (ins AnyType: $param);
    let results = (outs AnyType: $res);
    let assemblyFormat = [{  $param `:` type($param) `->` type($res) attr-dict }];
}
#endif // LINGODB_DIALECT_TupleStream_IR_OPS
