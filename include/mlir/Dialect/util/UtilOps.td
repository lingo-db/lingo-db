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
def StringConstantOp : Util_Op<"stringconst"> {
let summary = "string constant";
let arguments = (ins StrAttr:$val);
let results =(outs AnyMemRef:$ptr,I32:$len);
let assemblyFormat = "$val attr-dict `:` type($ptr) ";
}
def CombineOp : Util_Op<"combine"> {
let summary = "string constant";
let arguments = (ins Variadic<AnyType>:$vals);
let results =(outs TupleOf<[AnyType]>:$tuple);
let assemblyFormat = "$vals `:` type($vals) attr-dict `:` type($tuple) ";
}
#endif// OPS
