#ifndef OPS
#define OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgInterfaces.td"
include "lingodb/compiler/Dialect/Arrow/IR/ArrowBase.td"
include "lingodb/compiler/Dialect/util/UtilBase.td"


class Arrow_Op<string mnemonic, list<Trait> traits = []> : Op<Arrow_Dialect, mnemonic, traits> {
  let hasCustomAssemblyFormat = 1;

}

def Arrow_IsValidOp : Arrow_Op<"array.is_valid"> {
    let summary = "";
    let arguments = (ins Arrow_Array:$array,Index:$offset);
    let results = (outs I1:$valid);
    let assemblyFormat = " $array `,` $offset attr-dict";
}
def Arrow_LoadBoolOp: Arrow_Op<"array.load_bool"> {
    let summary = "";
    let arguments = (ins Arrow_Array:$array,Index:$offset);
    let results = (outs I1:$value);
    let assemblyFormat = " $array `,` $offset attr-dict";
}
def Arrow_LoadFixedSizedOp : Arrow_Op<"array.load_fixed_sized"> {
    let summary = "";
    let arguments = (ins Arrow_Array:$array,Index:$offset);
    let results = (outs AnyType:$value);
    let assemblyFormat = " $array `,` $offset `->` type($value) attr-dict";
}
def Arrow_LoadVariableSizeBinaryOp : Arrow_Op<"array.load_variable_size_binary"> {
    let summary = "";
    let arguments = (ins Arrow_Array:$array,Index:$offset);
    let results = (outs I32:$length, RefType : $ptr);
    let assemblyFormat = " $array `,` $offset `->` type($ptr) attr-dict";
}
def Arrow_AppendBoolOp : Arrow_Op<"array_builder.append_bool"> {
    let summary = "";
    let arguments = (ins Arrow_ArrayBuilder:$builder, I1: $value, Optional<I1>: $valid);
    let assemblyFormat = " $builder `,` $value ( `,` $valid^ )? attr-dict";
}
def Arrow_AppendFixedSizedOp : Arrow_Op<"array_builder.append_fixed_sized"> {
    let summary = "";
    let arguments = (ins Arrow_ArrayBuilder:$builder, AnyType: $value, Optional<I1>: $valid);
    let assemblyFormat = " $builder `,` $value `:` type($value) ( `,` $valid^ )? attr-dict";
}
def Arrow_AppendVariableSizeBinaryOp : Arrow_Op<"array_builder.append_variable_sized_binary"> {
    let summary = "";
    let arguments = (ins Arrow_ArrayBuilder:$builder, VarLen32Type: $value,  Optional<I1>: $valid);
    let assemblyFormat = " $builder `,` $value   ( `,` $valid^ )? attr-dict";
}

//todo: maybe better integration/abstraction
def Arrow_BuilderFromPtr : Arrow_Op<"array_builder.from_ptr"> {
    let summary = "";
    let arguments = (ins RefType:$ptr);
    let results = (outs Arrow_ArrayBuilder : $builder);

    let assemblyFormat = " $ptr `->` type($builder) attr-dict";
}

#endif// OPS
