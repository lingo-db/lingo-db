#ifndef OPS
#define OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"

def Util_Dialect : Dialect {
let name = "util";
let cppNamespace = "::mlir::util";
let extraClassDeclaration = [{
private:
void registerTypes();
std::shared_ptr<FunctionHelper> functionHelper;
public:
FunctionHelper& getFunctionHelper(){
    return *functionHelper;
}
}];
}

class Util_Op<string mnemonic, list<Trait> traits = []> :
Op<Util_Dialect, mnemonic, traits>{

}

class Util_Type<string name, string typeMnemonic> : TypeDef<Util_Dialect, name> {
    let mnemonic = typeMnemonic;
}


def RefType : Util_Type<"Ref","ref"> {
    let summary = "ref type";
    let parameters = (ins "Type":$elementType);
    let builders = [
        TypeBuilder<(ins), [{
            return $_get($_ctxt, mlir::IntegerType::get($_ctxt,8));
        }]>,
        TypeBuilderWithInferredContext<(ins "Type":$type), [{
            return $_get(type.getContext(), type);
        }]>,
    ];
    let builderCall = "$_builder.getType<" # dialect.cppNamespace # "::" # cppClassName # ">()";
    let hasCustomAssemblyFormat = 1;
}


def VarLen32Type : Util_Type<"VarLen32","varlen32"> {
    let summary = "type representing variable-length data up to 2^31 bytes";
}

def CreateVarLen  : Util_Op<"varlen32_create",[NoSideEffect]> {//todo: technically not currect
    let arguments = (ins RefType:$ref, I32: $len);
    let results=(outs VarLen32Type:$varlen);
    let assemblyFormat = " $ref `,` $len attr-dict";
}
def CreateConstVarLen  : Util_Op<"varlen32_create_const",[NoSideEffect]> {
    let arguments = (ins StrAttr: $str);
    let results = (outs VarLen32Type:$varlen);
    let assemblyFormat = " $str attr-dict";
}
def VarLenGetLen : Util_Op<"varlen32_getlen"> {
    let arguments = (ins VarLen32Type:$varlen);
    let results = (outs Index: $len);
    let assemblyFormat = " $varlen attr-dict";
}

def Hash64 : Util_Op<"hash_64",[NoSideEffect]> {
    let arguments = (ins Index:$val);
    let results = (outs Index:$hash);
    let assemblyFormat = "$val attr-dict";
}
def HashVarLen : Util_Op<"hash_varlen",[NoSideEffect]> {
    let arguments = (ins VarLen32Type:$val);
    let results = (outs Index:$hash);
    let assemblyFormat = "$val attr-dict";
}
def HashCombine : Util_Op<"hash_combine",[NoSideEffect]> {
    let arguments = (ins Index:$h1, Index:$h2);
    let results = (outs Index:$hash);
    let assemblyFormat = "$h1 `,` $h2 attr-dict";
}
def FilterTaggedPtr : Util_Op<"filter_tagged_ptr",[NoSideEffect]> {
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
def IsRefValidOp  : Util_Op<"is_ref_valid",[NoSideEffect]> {
let arguments = (ins RefType:$ref);
let results=(outs I1:$valid);
let assemblyFormat = "$ref `:` type($ref) attr-dict";
}
def InvalidRefOp  : Util_Op<"invalid_ref",[NoSideEffect]> {
let arguments = (ins );
let results=(outs RefType:$ref);
let assemblyFormat = "type($ref) attr-dict";
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
let hasCanonicalizeMethod = 1;
}
def LoadOp  : Util_Op<"load",[DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
let arguments = (ins RefType:$ref,Optional<Index>:$idx);
let results=(outs AnyType:$val);
let assemblyFormat = "$ref `[` $idx `]` `:`type($ref) `->` type($val) attr-dict";
let builders = [
        OpBuilder<(ins "Value":$ref, "Value":$idx), [{
    build($_builder, $_state, ref.getType().cast<mlir::util::RefType>().getElementType(), ref, idx);
}]>,OpBuilder<(ins "Value":$ref), [{
        build($_builder, $_state, ref.getType().cast<mlir::util::RefType>().getElementType(), ref, mlir::Value());
    }]>
];
}
def ArrayElementPtrOp  : Util_Op<"arrayelementptr",[NoSideEffect]> {
    let arguments = (ins RefType:$ref,Index:$idx);
    let results=(outs RefType:$res);
    let assemblyFormat = "$ref `[` $idx `]` `:`type($ref) `->` type($res) attr-dict";
}
def TupleElementPtrOp  : Util_Op<"tupleelementptr",[NoSideEffect]> {
    let arguments = (ins RefType:$ref,I32Attr:$idx);
    let results=(outs RefType:$res);
    let assemblyFormat = "$ref `[` $idx `]` `:`type($ref) `->` type($res) attr-dict";
    let hasVerifier=1;
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
    let hasVerifier=1;
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
    let hasVerifier=1;
let hasCanonicalizeMethod = 1;
let builders = [
        OpBuilder<(ins "Value":$tuple), [{
    build($_builder, $_state, tuple.getType().cast<TupleType>().getTypes(), tuple);
}]>
];
}
def UndefOp : Util_Op<"undef",[NoSideEffect]> {
    let summary = "undefined value";
    let arguments = (ins);
    let results =(outs AnyType:$res);
    let assemblyFormat = " attr-dict `:` type($res) ";
}
def GetTupleOp : Util_Op<"get_tuple",[NoSideEffect]> {
let summary = "get tuple";
let arguments = (ins TupleOf<[AnyType]>:$tuple, I32Attr:$offset);
let results =(outs AnyType: $val);
let assemblyFormat = "$tuple `[` $offset `]` `:` `(`type($tuple)`)` `->` type($val) attr-dict";
let hasCanonicalizeMethod = 1;
}
#endif// OPS
