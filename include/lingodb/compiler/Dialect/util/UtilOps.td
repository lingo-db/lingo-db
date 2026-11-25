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
    let hasVerifier=1;
}
def VarLenGetLen : Util_Op<"varlen32_getlen", [Pure]> {
    let arguments = (ins VarLen32Type:$varlen);
    let results = (outs Index: $len);
    let assemblyFormat = " $varlen attr-dict";
    let hasCanonicalizeMethod = 1;
}
def VarLenGetRef : Util_Op<"varlen32_get_ref", [Pure]> {
    let arguments = (ins VarLen32Type:$varlen);
    let results = (outs RefType:$ref);
    let assemblyFormat = "$varlen `:` type($ref) attr-dict";
    let hasFolder = 1;
}
def VarLenGetInlinedString : Util_Op<"varlen32_get_inlined_str", [Pure]> {
    let arguments = (ins VarLen32Type:$varlen);
    let results = (outs I128:$str);
    let assemblyFormat = " $varlen attr-dict";
}

def VarLenIsInvalid : Util_Op<"varlen32_is_invalid"> {
    let arguments = (ins VarLen32Type:$varlen);
    let results = (outs I1: $is_invalid);
    let assemblyFormat = " $varlen attr-dict";
}
def VarLenInvalid : Util_Op<"varlen32_invalid"> {
    let arguments = (ins );
    let results = (outs VarLen32Type:$varlen);
    let assemblyFormat = " attr-dict";
}
def VarLenCmp : Util_Op<"varlen32_cmp"> {
    let arguments = (ins VarLen32Type:$left, VarLen32Type:$right);
    let results = (outs I1: $eq, I1: $needs_detailed_eval);
    let assemblyFormat = " $left `,` $right attr-dict";
}
def VarLenCmpSimple : Util_Op<"varlen32_cmp_simple"> {
    let arguments = (ins VarLen32Type:$left, VarLen32Type:$right);
    let results = (outs I1: $eq);
    let assemblyFormat = " $left `,` $right attr-dict";
}
def VarLenTryCheapHash : Util_Op<"varlen32_try_cheap_hash"> {
    let arguments = (ins VarLen32Type:$varlen);
    let results = (outs I1: $complete, Index:$hash);
    let assemblyFormat = " $varlen attr-dict";
}
def LikeOp : Util_Op<"like"> {
    let arguments = (ins VarLen32Type:$str, OptionalAttr<DictionaryAttr>:$prefix, OptionalAttr<DictionaryAttr>:$suffix, DictArrayAttr:$subpatterns, I1Attr:$forceExact);
    let results = (outs I1:$matches);
    let assemblyFormat = "$str `like` $subpatterns (`prefix` $prefix^)? (`suffix` $suffix^)? `forceExact` $forceExact attr-dict `:` functional-type(operands, results)";
}
def BytesStartsWith : Util_Op<"bytes_starts_with", [Pure]> {
    let arguments = (ins I128:$str, StrAttr:$prefix, Index:$startIndex, Index:$endIndex);
    let results = (outs I1:$starts_with);
    let assemblyFormat = " $str `[` $startIndex `to` $endIndex `]` `,` $prefix attr-dict";
}
def StringStartsWith : Util_Op<"string_starts_with", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let arguments = (ins RefType:$str, StrAttr:$prefix, Index:$startIndex, Index:$endIndex);
    let results = (outs I1:$starts_with);
    let assemblyFormat = " $str `[` $startIndex `to` $endIndex `]` `,` $prefix attr-dict";
}
def BytesEndsWith : Util_Op<"bytes_ends_with", [Pure]> {
    let arguments = (ins I128:$str, StrAttr:$suffix, Index:$startIndex, Index:$endIndex);
    let results = (outs I1:$ends_with);
    let assemblyFormat = " $str `[` $startIndex `to` $endIndex `]` `,` $suffix attr-dict";
}
def StringEndsWith : Util_Op<"string_ends_with", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let arguments = (ins RefType:$str, StrAttr:$suffix, Index:$startIndex, Index:$endIndex);
    let results = (outs I1:$ends_with);
    let assemblyFormat = " $str `[` $startIndex `to` $endIndex `]` `,` $suffix attr-dict";
}
def BytesContains : Util_Op<"bytes_contains", [Pure]> {
    let arguments = (ins I128:$str, StrAttr:$subpattern, Index:$startIndex, Index:$endIndex);
    let results = (outs I1:$contains, Index:$matchStart);
    let assemblyFormat = " $str `[` $startIndex `to` $endIndex `]` `,` $subpattern attr-dict";
}
def StringContains : Util_Op<"string_contains", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let arguments = (ins RefType:$str, StrAttr:$subpattern, Index:$startIndex, Index:$endIndex);
    let results = (outs I1:$contains, Index:$matchStart);
    let assemblyFormat = " $str `[` $startIndex `to` $endIndex `]` `,` $subpattern attr-dict";
}
def BytesNonMatchIndex : Util_Op<"bytes_non_match_index", [Pure]> {
    let arguments = (ins I128:$str, StrAttr:$subpattern, Index:$startIndex, Index:$endIndex);
    let results = (outs Index:$pos);
    let assemblyFormat = " $str `[` $startIndex `to` $endIndex `]` `,` $subpattern attr-dict";
}

def StringNonMatchIndex : Util_Op<"string_non_match_index", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let arguments = (ins RefType:$str, StrAttr:$subpattern, Index:$startIndex, Index:$endIndex);
    let results = (outs Index:$pos);
    let assemblyFormat = " $str `[` $startIndex `to` $endIndex `]` `,` $subpattern attr-dict";
}

def RefMemchr : Util_Op<"memchr", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let arguments = (ins RefType:$ref, Index:$len, I8:$byte);
    let results = (outs I1:$found, Index:$pos);
    let assemblyFormat = "$ref `:` type($ref) `,` $len `,` $byte attr-dict";
}

def InlineMemchr : Util_Op<"swar_memchr", [Pure]> {
    let arguments = (ins I128:$data, Index:$len, I8:$byte);
    let results = (outs I1:$found, Index:$pos);
    let assemblyFormat = "$data `,` $len `,` $byte attr-dict";
}
def GetUTF8CodeLenOp : Util_Op<"get_utf8codelen", [Pure]> {
    let arguments = (ins I8:$byte);
    let results = (outs Index:$len);
    let assemblyFormat = "$byte attr-dict `:` type($len)";
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
def UnTagPtr : Util_Op<"untag_ptr",[Pure]> {
    let arguments = (ins RefType:$ref);
    let results = (outs RefType:$res);
    let assemblyFormat = "$ref `:` type($ref) `->` type($res) attr-dict";
}
def PtrTagMatches : Util_Op<"ptr_tag_matches",[Pure]> {
    let arguments = (ins RefType:$ref,Index:$hash);
    let results = (outs I1:$matches);
    let assemblyFormat = "$hash `,` $ref `:` type($ref) attr-dict";
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
    let summary = "create a `memref<?xi8>` with dimension info to the buffer";
    let arguments = (ins BufferType:$buffer);
    let results = (outs Arg<MemRefRankOf<[I8], [1]>>:$memref);
    let assemblyFormat = "$buffer `:`  type($buffer) `->` type($memref) attr-dict";
}

def StoreOp  : Util_Op<"store"> {
let arguments = (ins AnyType:$val,RefType:$ref,Optional<Index>:$idx);
let assemblyFormat = "$val `:` type($val) `,` $ref `[` $idx `]` `:`type($ref) attr-dict";
let hasCanonicalizeMethod = 1;
}
def LoadElementOp : Util_Op<"loadelement",[DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let arguments = (ins RefType:$ref,I32Attr:$idx);
    let results=(outs AnyType:$val);
    let assemblyFormat = "$ref `[` $idx `]` `:`type($ref) `->` type($val) attr-dict";
}
def StoreElementOp : Util_Op<"storeelement"> {
    let arguments = (ins AnyType:$val,RefType:$ref,I32Attr:$idx);
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
def UnalignedLoadOp  : Util_Op<"unaligned_load",[DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let arguments = (ins RefType:$ref);
    let results=(outs AnyType:$val);
    let assemblyFormat = "$ref `:`type($ref) `->` type($val) attr-dict";
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
let assemblyFormat = "($vals^ `:` type($vals))? attr-dict `->` type($tuple) ";
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
  let hasFolder = 1;
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
 let hasFolder = 1;
}
def IsBitSetConstOp : Util_Op<"is_bit_set_const",[Pure]> {
    let arguments = (ins AnyInteger: $bitset, I32Attr:$bitpos);
    let results =(outs I1: $is_set);
    let assemblyFormat = " $bitset  `:` type($bitset)  `,` $bitpos attr-dict";
}
def SetBitConstOp : Util_Op<"set_bit_const",[Pure]> {
    let arguments = (ins AnyInteger: $bitset, I1: $val,  I32Attr:$bitpos);
    let results =(outs AnyInteger: $res);
    let assemblyFormat = " $bitset `:` type($bitset) `,` $val `,` $bitpos `->` type($res) attr-dict";
}
def CreateConstArrayOp : Util_Op<"const_array", [Pure]> {
    let arguments = (ins ElementsAttr:$data, I64Attr:$alignment);
    let results = (outs RefType:$res);
    let assemblyFormat = "$data `:` type($res) attr-dict";
    let hasVerifier = 1;
}
def GetConstArrayAtOp : Util_Op<"get_const_array_at", [Pure]> {
    let arguments = (ins RefType:$array, Index:$idx);
    let results = (outs AnyType:$res);
    let assemblyFormat = "$array `[` $idx `]` `:` type($array) `->` type($res) attr-dict";
    let hasVerifier = 1;
}
def LoadVectorOp : Util_Op<"load_vector", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let arguments = (ins RefType:$ref, I64Attr:$alignment);
    let results = (outs VectorOfLengthAndType<[16], [I8]>:$res);
    let assemblyFormat = "$ref `align` $alignment `:` type($ref) `->` type($res) attr-dict";
}
def CmpistriOp : Util_Op<"pcmpistri", [Pure]> {
    let arguments = (ins VectorOfLengthAndType<[16], [I8]>:$a, VectorOfLengthAndType<[16], [I8]>:$b, I8Attr:$flags);
    let results = (outs I32:$index);
    let assemblyFormat = "$a `,` $b `,` $flags `:` type($a) `,` type($b) attr-dict";
}
#endif// OPS
