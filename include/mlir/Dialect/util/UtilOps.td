#ifndef OPS
#define OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"

//===----------------------------------------------------------------------===//
// Util dialect definition.
//===----------------------------------------------------------------------===//

def Util_Dialect : Dialect {
let name = "util";
let summary = "A util out-of-tree MLIR dialect.";
let description = [{
This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
}];
let cppNamespace = "::mlir::util";

}

class Util_Op<string mnemonic, list<OpTrait> traits = []> :
Op<Util_Dialect, mnemonic, traits>{
let printer = [{ return ::print(p, *this); }];
let parser = [{ return ::parse$cppClass(parser, result); }];
}
def CombineOp : Util_Op<"combine"> {
let summary = "string constant";
let arguments = (ins Variadic<AnyType>:$vals);
let results =(outs TupleOf<[AnyType]>:$tuple);
let assemblyFormat = "$vals `:` type($vals) attr-dict `:` type($tuple) ";
}
def SplitOp : Util_Op<"split_tuple"> {
let summary = "split tuple";
let arguments = (ins TupleOf<[AnyType]>:$tuple);
let results =(outs Variadic<AnyType>:$vals);
let assemblyFormat = "$tuple `:` type($tuple) attr-dict `:` type($vals)";
}
def UndefTupleOp : Util_Op<"undef_tuple"> {
let summary = "undef Tuple";
let arguments = (ins);
let results =(outs TupleOf<[AnyType]>:$tuple);
let assemblyFormat = " attr-dict `:` type($tuple) ";
}
def SetTupleOp : Util_Op<"set_tuple"> {
let summary = "set tuple";
let arguments = (ins TupleOf<[AnyType]>:$tuple, AnyType: $val, I32Attr:$offset);
let results =(outs TupleOf<[AnyType]>:$tuple_out);
let assemblyFormat = "$tuple `:` `[` $offset `]` `=` $val `:` `(` type($tuple)`,` type($val)`)` `->` type($tuple_out) attr-dict";
}
def GetTupleOp : Util_Op<"get_tuple"> {
let summary = "get tuple";
let arguments = (ins TupleOf<[AnyType]>:$tuple, I32Attr:$offset);
let results =(outs AnyType: $val);
let assemblyFormat = "$tuple `[` $offset `]` `:` `(`type($tuple)`)` `->` type($val) attr-dict";
}
#endif// OPS
