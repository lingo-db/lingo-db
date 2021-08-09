#ifndef OPS
#define OPS

include "mlir/IR/OpBase.td"
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
let extraClassDeclaration = [{
private:
void registerTypes();

public:
}];
}

class Util_Op<string mnemonic, list<OpTrait> traits = []> :
Op<Util_Dialect, mnemonic, traits>{
let printer = [{ return ::print(p, *this); }];
let parser = [{ return ::parse$cppClass(parser, result); }];
}

class Util_Type<string name, string typeMnemonic> : TypeDef<Util_Dialect, name> {
    let mnemonic = typeMnemonic;
}


def GenericMemrefType : Util_Type<"GenericMemref","generic_memref"> {
let summary = "generic memref type";
let parameters = (ins "Type":$elementType,"llvm::Optional<int64_t>": $size);
}

def AllocOp  : Util_Op<"alloc"> {
let arguments = (ins Optional<Index>:$size);
let results=(outs GenericMemrefType:$generic_memref);
let assemblyFormat = " `(` $size `)` `:` type($generic_memref) attr-dict";
}
def AllocaOp  : Util_Op<"alloca"> {
let arguments = (ins Optional<Index>:$size);
let results=(outs GenericMemrefType:$generic_memref);
let assemblyFormat = " `(` $size `)` `:` type($generic_memref) attr-dict";
}
def DeAllocOp  : Util_Op<"dealloc"> {
let arguments = (ins GenericMemrefType:$generic_memref);
let assemblyFormat = "$generic_memref `:` type($generic_memref) attr-dict";
}
def ToGenericMemrefOp  : Util_Op<"to_generic_memref",[NoSideEffect]> {
let arguments = (ins I8MemRef:$memref);
let results=(outs GenericMemrefType:$generic_memref);
let assemblyFormat = "$memref `:` type($memref) `->` type($generic_memref) attr-dict";
}
def ToMemrefOp  : Util_Op<"to_memref",[NoSideEffect]> {
let arguments = (ins GenericMemrefType:$generic_memref);
let results=(outs I8MemRef:$memref);
let assemblyFormat = "$generic_memref `:` type($generic_memref) `->` type($memref) attr-dict";
}
def GenericMemrefCastOp : Util_Op<"generic_memref_cast",[NoSideEffect]> {
let summary = "generic memref cast function";
let arguments = (ins GenericMemrefType:$val);
let results=(outs GenericMemrefType:$res);
let assemblyFormat = "$val `:` type($val) `->` type($res) attr-dict";
}
def StoreOp  : Util_Op<"store"> {
let arguments = (ins AnyType:$val,GenericMemrefType:$generic_memref,Optional<Index>:$idx);
let assemblyFormat = "$val `:` type($val) `,` $generic_memref `[` $idx `]` `:`type($generic_memref) attr-dict";
}
def DimOp  : Util_Op<"dim",[NoSideEffect]> {
let arguments = (ins GenericMemrefType:$generic_memref);
let results=(outs Index:$size);
let assemblyFormat = "$generic_memref `:`type($generic_memref) attr-dict";
}
def LoadOp  : Util_Op<"load",[NoSideEffect]> {
let arguments = (ins GenericMemrefType:$generic_memref,Optional<Index>:$idx);
let results=(outs AnyType:$val);
let assemblyFormat = "$generic_memref `[` $idx `]` `:`type($generic_memref) `->` type($val) attr-dict";
}
def SizeOfOp  : Util_Op<"sizeof",[NoSideEffect]> {
let arguments = (ins TypeAttr:$type);
let results=(outs Index:$res);
let assemblyFormat = "$type attr-dict";
}
def PackOp : Util_Op<"pack",[NoSideEffect]> {
let summary = "string constant";
let arguments = (ins Variadic<AnyType>:$vals);
let results =(outs TupleOf<[AnyType]>:$tuple);
let assemblyFormat = "$vals `:` type($vals) attr-dict `->` type($tuple) ";
let verifier = [{ return ::verify(*this); }];
}
def UnPackOp : Util_Op<"unpack",[NoSideEffect]> {
let summary = "unpack tuple";
let arguments = (ins TupleOf<[AnyType]>:$tuple);
let results =(outs Variadic<AnyType>:$vals);
let assemblyFormat = "$tuple `:` type($tuple) attr-dict `->` type($vals)";
let verifier = [{ return ::verify(*this); }];
let hasCanonicalizeMethod = 1;
}
def UndefTupleOp : Util_Op<"undef_tuple",[NoSideEffect]> {
let summary = "undef Tuple";
let arguments = (ins);
let results =(outs TupleOf<[AnyType]>:$tuple);
let assemblyFormat = " attr-dict `:` type($tuple) ";
}
def SetTupleOp : Util_Op<"set_tuple",[NoSideEffect]> {
let summary = "set tuple";
let arguments = (ins TupleOf<[AnyType]>:$tuple, AnyType: $val, I32Attr:$offset);
let results =(outs TupleOf<[AnyType]>:$tuple_out);
let assemblyFormat = "$tuple `[` $offset `]` `=` $val `:` `(` type($tuple)`,` type($val)`)` `->` type($tuple_out) attr-dict";
}
def GetTupleOp : Util_Op<"get_tuple",[NoSideEffect]> {
let summary = "get tuple";
let arguments = (ins TupleOf<[AnyType]>:$tuple, I32Attr:$offset);
let results =(outs AnyType: $val);
let assemblyFormat = "$tuple `[` $offset `]` `:` `(`type($tuple)`)` `->` type($val) attr-dict";
}
#endif// OPS
