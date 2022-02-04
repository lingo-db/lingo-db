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


def RefType : Util_Type<"Ref","ref"> {
let summary = "ref type";
let parameters = (ins "Type":$elementType,"llvm::Optional<int64_t>": $size);
    let builders = [
    TypeBuilder<(ins "Type":$elementType), [{
    return $_get($_ctxt, elementType,llvm::Optional<int64_t>());
    }]>,
    TypeBuilder<(ins ), [{
    return $_get($_ctxt, mlir::IntegerType::get($_ctxt,8),llvm::Optional<int64_t>());
    }]>
    ];
let builderCall = "$_builder.getType<" # dialect.cppNamespace #
"::" # cppClassName # ">()";
}


def VarLen32Type : Util_Type<"VarLen32","varlen32"> {
let summary = "type representing variable-length data up to 2^32 bytes";
}

def CreateVarLen  : Util_Op<"varlen32_create"> {
let arguments = (ins RefType:$ref, I32: $len);
let results=(outs VarLen32Type:$varlen);
let assemblyFormat = " $ref `,` $len attr-dict";
}
def VarLenGetRef : Util_Op<"varlen32_getref"> {
    let arguments = (ins VarLen32Type:$varlen);
    let results = (outs RefType: $ref);
    let assemblyFormat = " $varlen `->` type($ref) attr-dict";
}
def VarLenGetLen : Util_Op<"varlen32_getlen"> {
    let arguments = (ins VarLen32Type:$varlen);
    let results = (outs Index: $len);
    let assemblyFormat = " $varlen attr-dict";
}

def Hash64 : Util_Op<"hash_64"> {
    let arguments = (ins Index:$val);
    let results = (outs Index:$hash);
    let assemblyFormat = "$val attr-dict";
}
def HashVarLen : Util_Op<"hash_varlen"> {
    let arguments = (ins VarLen32Type:$val);
    let results = (outs Index:$hash);
    let assemblyFormat = "$val attr-dict";
}
def HashCombine : Util_Op<"hash_combine"> {
    let arguments = (ins Index:$h1, Index:$h2);
    let results = (outs Index:$hash);
    let assemblyFormat = "$h1 `,` $h2 attr-dict";
}
def AllocOp  : Util_Op<"alloc"> {
let arguments = (ins Optional<Index>:$size);
let results=(outs RefType:$ref);
let assemblyFormat = " `(` $size `)` `:` type($ref) attr-dict";
}
def AllocaOp  : Util_Op<"alloca"> {
let arguments = (ins Optional<Index>:$size);
let results=(outs RefType:$ref);
let assemblyFormat = " `(` $size `)` `:` type($ref) attr-dict";
}
def DeAllocOp  : Util_Op<"dealloc"> {
let arguments = (ins RefType:$ref);
let assemblyFormat = "$ref `:` type($ref) attr-dict";
}
def ToGenericMemrefOp  : Util_Op<"to_generic_memref",[NoSideEffect]> {
let arguments = (ins I8MemRef:$memref);
let results=(outs RefType:$ref);
let assemblyFormat = "$memref `:` type($memref) `->` type($ref) attr-dict";
}
def ToMemrefOp  : Util_Op<"to_memref",[NoSideEffect]> {
let arguments = (ins RefType:$ref);
let results=(outs AnyMemRef:$memref);
let assemblyFormat = "$ref `:` type($ref) `->` type($memref) attr-dict";
}
def GenericMemrefCastOp : Util_Op<"generic_memref_cast",[NoSideEffect]> {
let summary = "generic memref cast function";
let arguments = (ins RefType:$val);
let results=(outs RefType:$res);
let assemblyFormat = "$val `:` type($val) `->` type($res) attr-dict";
}
def StoreOp  : Util_Op<"store"> {
let arguments = (ins AnyType:$val,RefType:$ref,Optional<Index>:$idx);
let assemblyFormat = "$val `:` type($val) `,` $ref `[` $idx `]` `:`type($ref) attr-dict";
}
def DimOp  : Util_Op<"dim",[NoSideEffect]> {
let arguments = (ins RefType:$ref);
let results=(outs Index:$size);
let assemblyFormat = "$ref `:`type($ref) attr-dict";
}
def LoadOp  : Util_Op<"load",[NoSideEffect]> {
let arguments = (ins RefType:$ref,Optional<Index>:$idx);
let results=(outs AnyType:$val);
let assemblyFormat = "$ref `[` $idx `]` `:`type($ref) `->` type($val) attr-dict";
}
def ElementPtrOp  : Util_Op<"elementptr",[NoSideEffect]> {
    let arguments = (ins RefType:$ref,Index:$idx);
    let results=(outs RefType:$res);
    let assemblyFormat = "$ref `[` $idx `]` `:`type($ref) `->` type($res) attr-dict";
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
let builders = [
        OpBuilder<(ins "ValueRange":$vals), [{
    std::vector<Type> types;
    for(auto val:vals){
        types.push_back(val.getType());
        }
    build($_builder, $_state, TupleType::get($_builder.getContext(),types), vals);
}]>];
}
def UnPackOp : Util_Op<"unpack",[NoSideEffect]> {
let summary = "unpack tuple";
let arguments = (ins TupleOf<[AnyType]>:$tuple);
let results =(outs Variadic<AnyType>:$vals);
let assemblyFormat = "$tuple `:` type($tuple) attr-dict `->` type($vals)";
let verifier = [{ return ::verify(*this); }];
let hasCanonicalizeMethod = 1;
let builders = [
        OpBuilder<(ins "Value":$tuple), [{
    build($_builder, $_state, tuple.getType().cast<TupleType>().getTypes(), tuple);
}]>
];
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
