#ifndef OPS
#define OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "UtilBase.td"


class Util_Op<string mnemonic, list<Trait> traits = []> :
Op<Util_Dialect, mnemonic, traits>{

}



def CreateVarLen  : Util_Op<"varlen32_create",[Pure]> {//todo: technically not currect
    let arguments = (ins RefType:$ref, I32: $len);
    let results=(outs VarLen32Type:$varlen);
    let assemblyFormat = " $ref `,` $len attr-dict";
}
def CreateConstVarLen  : Util_Op<"varlen32_create_const",[Pure]> {
    let arguments = (ins StrAttr: $str);
    let results = (outs VarLen32Type:$varlen);
    let assemblyFormat = " $str attr-dict";
}
def BufferGetLen : Util_Op<"buffer_getlen"> {
    let arguments = (ins BufferType:$buffer);
    let results = (outs Index: $len);
    let assemblyFormat = " $buffer `:` type($buffer)  attr-dict";
}
def BufferGetRef : Util_Op<"buffer_getref"> {
    let arguments = (ins BufferType:$buffer);
    let results = (outs RefType: $ref);
    let assemblyFormat = " $buffer `:` type($buffer) `->` type($ref) attr-dict";
}
def BufferGetElementRef : Util_Op<"buffer_getelementref"> {
    let arguments = (ins BufferType:$buffer, Index:$idx);
    let results = (outs RefType: $ref);
    let assemblyFormat = " $buffer `[` $idx `]` `:` type($buffer) `->` type($ref) attr-dict";
}
def VarLenGetLen : Util_Op<"varlen32_getlen"> {
    let arguments = (ins VarLen32Type:$varlen);
    let results = (outs Index: $len);
    let assemblyFormat = " $varlen attr-dict";
}
def VarLenCmp : Util_Op<"varlen32_cmp"> {
    let arguments = (ins VarLen32Type:$left, VarLen32Type:$right);
    let results = (outs I1: $eq, I1: $needs_detailed_eval);
    let assemblyFormat = " $left `,` $right attr-dict";
}
def VarLenTryCheapHash : Util_Op<"varlen32_try_cheap_hash"> {
    let arguments = (ins VarLen32Type:$varlen);
    let results = (outs I1: $complete, Index:$hash);
    let assemblyFormat = " $varlen attr-dict";
}

def Hash64 : Util_Op<"hash_64",[Pure]> {
    let arguments = (ins Index:$val);
    let results = (outs Index:$hash);
    let assemblyFormat = "$val attr-dict";
}
def HashVarLen : Util_Op<"hash_varlen",[Pure]> {
    let arguments = (ins VarLen32Type:$val);
    let results = (outs Index:$hash);
    let assemblyFormat = "$val attr-dict";
}
def HashCombine : Util_Op<"hash_combine",[Pure]> {
    let arguments = (ins Index:$h1, Index:$h2);
    let results = (outs Index:$hash);
    let assemblyFormat = "$h1 `,` $h2 attr-dict";
}
def TagPtr : Util_Op<"tag_ptr",[Pure]> {
    let arguments = (ins RefType:$ref,Index:$hash);
    let results = (outs RefType:$res);
    let assemblyFormat = "$hash `,` $ref `:` type($ref) `->` type($res) attr-dict";
}
def UnTagPtr : Util_Op<"untag_ptr",[Pure]> {
    let arguments = (ins RefType:$ref);
    let results = (outs RefType:$res);
    let assemblyFormat = "$ref `:` type($ref) `->` type($res) attr-dict";
}
def FilterTaggedPtr : Util_Op<"filter_tagged_ptr",[Pure]> {
    let arguments = (ins RefType:$ref,Index:$hash);
    let results = (outs RefType:$res);
    let assemblyFormat = "$hash `,` $ref `:` type($ref) `->` type($res) attr-dict";
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
def ToGenericMemrefOp  : Util_Op<"to_generic_memref",[Pure]> {
let arguments = (ins I8MemRef:$memref);
let results=(outs RefType:$ref);
let assemblyFormat = "$memref `:` type($memref) `->` type($ref) attr-dict";
}
def ToMemrefOp  : Util_Op<"to_memref",[Pure]> {
let arguments = (ins RefType:$ref);
let results=(outs AnyMemRef:$memref);
let assemblyFormat = "$ref `:` type($ref) `->` type($memref) attr-dict";
}
def IsRefValidOp  : Util_Op<"is_ref_valid",[Pure]> {
let arguments = (ins RefType:$ref);
let results=(outs I1:$valid);
let assemblyFormat = "$ref `:` type($ref) attr-dict";
}
def InvalidRefOp  : Util_Op<"invalid_ref",[Pure]> {
let arguments = (ins );
let results=(outs RefType:$ref);
let assemblyFormat = "type($ref) attr-dict";
}
def GenericMemrefCastOp : Util_Op<"generic_memref_cast",[Pure]> {
let summary = "generic memref cast function";
let arguments = (ins RefType:$val);
let results=(outs RefType:$res);
let assemblyFormat = "$val `:` type($val) `->` type($res) attr-dict";
}
def BufferCastOp : Util_Op<"buffer_cast",[Pure]> {
let summary = "generic memref cast function";
let arguments = (ins BufferType:$val);
let results=(outs BufferType:$res);
let assemblyFormat = "$val `:` type($val) `->` type($res) attr-dict";
}
def BufferCreateOp : Util_Op<"buffer_create",[Pure]> {
let summary = "create buffer from ptr and len";
let arguments = (ins RefType:$ptr, Index:$len);
let results=(outs BufferType:$res);
let assemblyFormat = "$ptr `:` type($ptr) `,` $len `->` type($res) attr-dict";
}
def BufferGetMemRefOp : Util_Op<"buffer_get_memref",[Pure]> {
    let summary = "create a memref<?xi8> with dimension info to the buffer";
    let arguments = (ins BufferType:$buffer);
    let results = (outs Arg<MemRefRankOf<[I8], [1]>>:$memref);
    let assemblyFormat = "$buffer `:`  type($buffer) `->` type($memref) attr-dict";
}

def StoreOp  : Util_Op<"store"> {
let arguments = (ins AnyType:$val,RefType:$ref,Optional<Index>:$idx);
let assemblyFormat = "$val `:` type($val) `,` $ref `[` $idx `]` `:`type($ref) attr-dict";
let hasCanonicalizeMethod = 1;
}
def LoadOp  : Util_Op<"load",[DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
let arguments = (ins RefType:$ref,Optional<Index>:$idx);
let results=(outs AnyType:$val);
let assemblyFormat = "$ref `[` $idx `]` `:`type($ref) `->` type($val) attr-dict";
let builders = [
        OpBuilder<(ins "mlir::Value":$ref, "mlir::Value":$idx), [{
    build($_builder, $_state, ref.getType().cast<lingodb::compiler::dialect::util::RefType>().getElementType(), ref, idx);
}]>,OpBuilder<(ins "mlir::Value":$ref), [{
        build($_builder, $_state, ref.getType().cast<lingodb::compiler::dialect::util::RefType>().getElementType(), ref, mlir::Value());
    }]>
];
}
def ArrayElementPtrOp  : Util_Op<"arrayelementptr",[Pure]> {
    let arguments = (ins RefType:$ref,Index:$idx);
    let results=(outs RefType:$res);
    let assemblyFormat = "$ref `[` $idx `]` `:`type($ref) `->` type($res) attr-dict";
}
def TupleElementPtrOp  : Util_Op<"tupleelementptr",[Pure]> {
    let arguments = (ins RefType:$ref,I32Attr:$idx);
    let results=(outs RefType:$res);
    let assemblyFormat = "$ref `[` $idx `]` `:`type($ref) `->` type($res) attr-dict";
    let hasVerifier=1;
}
def SizeOfOp  : Util_Op<"sizeof",[Pure]> {
let arguments = (ins TypeAttr:$type);
let results=(outs Index:$res);
let assemblyFormat = "$type attr-dict";
}
def PackOp : Util_Op<"pack",[Pure]> {
let summary = "string constant";
let arguments = (ins Variadic<AnyType>:$vals);
let results =(outs TupleOf<[AnyType]>:$tuple);
let assemblyFormat = "$vals `:` type($vals) attr-dict `->` type($tuple) ";
    let hasVerifier=1;
let builders = [
        OpBuilder<(ins "mlir::ValueRange":$vals), [{
    std::vector<Type> types;
    for(auto val:vals){
        types.push_back(val.getType());
        }
    build($_builder, $_state, TupleType::get($_builder.getContext(),types), vals);
}]>];
}
def UnPackOp : Util_Op<"unpack",[Pure]> {
let summary = "unpack tuple";
let arguments = (ins TupleOf<[AnyType]>:$tuple);
let results =(outs Variadic<AnyType>:$vals);
let assemblyFormat = "$tuple `:` type($tuple) attr-dict `->` type($vals)";
    let hasVerifier=1;
let hasCanonicalizeMethod = 1;
let builders = [
        OpBuilder<(ins "mlir::Value":$tuple), [{
    build($_builder, $_state, tuple.getType().cast<TupleType>().getTypes(), tuple);
}]>
];
}
def UndefOp : Util_Op<"undef",[Pure]> {
    let summary = "undefined value";
    let arguments = (ins);
    let results =(outs AnyType:$res);
    let hasCanonicalizeMethod = 1;
    let assemblyFormat = " attr-dict `:` type($res) ";
}
def GetTupleOp : Util_Op<"get_tuple",[Pure]> {
let summary = "get tuple";
let arguments = (ins TupleOf<[AnyType]>:$tuple, I32Attr:$offset);
let results =(outs AnyType: $val);
let assemblyFormat = "$tuple `[` $offset `]` `:` `(`type($tuple)`)` `->` type($val) attr-dict";
let hasCanonicalizeMethod = 1;
}
#endif// OPS
