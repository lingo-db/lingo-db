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
def ToGenericMemrefOp  : Util_Op<"to_generic_memref"> {
let arguments = (ins I8MemRef:$memref);
let results=(outs GenericMemrefType:$generic_memref);
let assemblyFormat = "$memref `:` type($memref) `->` type($generic_memref) attr-dict";
}
def ToMemrefOp  : Util_Op<"to_memref"> {
let arguments = (ins GenericMemrefType:$generic_memref);
let results=(outs I8MemRef:$memref);
let assemblyFormat = "$generic_memref `:` type($generic_memref) `->` type($memref) attr-dict";
}
def StoreOp  : Util_Op<"store"> {
let arguments = (ins AnyType:$val,GenericMemrefType:$generic_memref,Optional<Index>:$idx);
let assemblyFormat = "$val `:` type($val) `,` $generic_memref `[` $idx `]` `:`type($generic_memref) attr-dict";
}
def DimOp  : Util_Op<"dim"> {
let arguments = (ins GenericMemrefType:$generic_memref);
let results=(outs Index:$size);
let assemblyFormat = "$generic_memref `:`type($generic_memref) attr-dict";
}
def LoadOp  : Util_Op<"load"> {
let arguments = (ins GenericMemrefType:$generic_memref,Optional<Index>:$idx);
let results=(outs AnyType:$val);
let assemblyFormat = "$generic_memref `[` $idx `]` `:`type($generic_memref) `->` type($val) attr-dict";
}
def MemberRefOp  : Util_Op<"member_ref"> {
let arguments = (ins GenericMemrefType:$source_ref,Optional<Index>:$memref_idx,IndexAttr:$tuple_idx);
let results=(outs GenericMemrefType:$result_ref);
let assemblyFormat = "$source_ref `[` $memref_idx `]` `[` $tuple_idx `]` `:`type($source_ref) `->` type($result_ref) attr-dict";
}
def ToRawPointerOp  : Util_Op<"to_raw_ptr"> {
let arguments = (ins GenericMemrefType:$ref);
let results=(outs Index:$ptr);
let assemblyFormat = "$ref `:`type($ref) attr-dict";
}
def FromRawPointerOp  : Util_Op<"from_raw_ptr"> {
let arguments = (ins Index:$ptr);
let results=(outs GenericMemrefType:$ref);
let assemblyFormat = "$ptr `->` type($ref) attr-dict";
}
def SizeOfOp  : Util_Op<"sizeof"> {
let arguments = (ins TypeAttr:$type);
let results=(outs Index:$res);
let assemblyFormat = "$type attr-dict";
}
def PackOp : Util_Op<"pack"> {
let summary = "string constant";
let arguments = (ins Variadic<AnyType>:$vals);
let results =(outs TupleOf<[AnyType]>:$tuple);
let assemblyFormat = "$vals `:` type($vals) attr-dict `->` type($tuple) ";
}
def UnPackOp : Util_Op<"unpack"> {
let summary = "unpack tuple";
let arguments = (ins TupleOf<[AnyType]>:$tuple);
let results =(outs Variadic<AnyType>:$vals);
let assemblyFormat = "$tuple `:` type($tuple) attr-dict `->` type($vals)";
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
let assemblyFormat = "$tuple `[` $offset `]` `=` $val `:` `(` type($tuple)`,` type($val)`)` `->` type($tuple_out) attr-dict";
}
def GetTupleOp : Util_Op<"get_tuple"> {
let summary = "get tuple";
let arguments = (ins TupleOf<[AnyType]>:$tuple, I32Attr:$offset);
let results =(outs AnyType: $val);
let assemblyFormat = "$tuple `[` $offset `]` `:` `(`type($tuple)`)` `->` type($val) attr-dict";
}
#endif// OPS
