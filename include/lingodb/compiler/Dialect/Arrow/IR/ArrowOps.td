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

def Arrow_IsValidOp : Arrow_Op<"array.is_valid", [Pure] > {
    let summary = "returns if a array element at a given offset is valid or not.";
    let arguments = (ins Arrow_Array:$array,Index:$offset);
    let results = (outs I1:$valid);
    let assemblyFormat = " $array `,` $offset attr-dict";
}
def Arrow_LoadBoolOp: Arrow_Op<"array.load_bool", [Pure] > {
    let summary = "loads a boolean value from an array at a given offset.";
    let description = "This special operation is necessary, since Arrow stores boolean values as bitset, and not individual bytes.";
    let arguments = (ins Arrow_Array:$array,Index:$offset);
    let results = (outs I1:$value);
    let assemblyFormat = " $array `,` $offset attr-dict";
}
def Arrow_LoadFixedSizedOp : Arrow_Op<"array.load_fixed_sized", [Pure] > {
    let summary = "Loads an arbitrary, fixed sized value from an array at a given offset";
    let description = [{
        Used for loading types that are of fixed size from an arrow array (e.g., integers, floats, decimals, dates, timestamp).
        There are now runtime checks to ensure that the type of the value matches the type of the array, so this operation can be used for any fixed sized type.
    }];
    let arguments = (ins Arrow_Array:$array,Index:$offset);
    let results = (outs AnyType:$value);
    let assemblyFormat = " $array `,` $offset `->` type($value) attr-dict";
}
def Arrow_LoadVariableSizeBinaryOp : Arrow_Op<"array.load_variable_size_binary", [Pure] > {
    let summary = "Loads a variable sized binary value from an array at a given offset";
    let description = [{
        Used for loading variable sized binary values from an arrow array (e.g., strings, binary data).
        It returns both a pointer to the data and the length of the data.
    }];
    let arguments = (ins Arrow_Array:$array,Index:$offset);
    let results = (outs I32:$length, RefType : $ptr);
    let assemblyFormat = " $array `,` $offset `->` type($ptr) attr-dict";
}
def Arrow_AppendBoolOp : Arrow_Op<"array_builder.append_bool"> {
    let summary = "Appends a boolean value to an Arrow array builder.";
    let description = [{
        This operation appends a boolean value to an Arrow array builder, optionally with a validity flag.
        This operation is necessary because Arrow stores boolean values as a bitset, not as individual bytes.
    }];
    let arguments = (ins Arrow_ArrayBuilder:$builder, I1: $value, Optional<I1>: $valid);
    let assemblyFormat = " $builder `,` $value ( `,` $valid^ )? attr-dict";
}
def Arrow_AppendFixedSizedOp : Arrow_Op<"array_builder.append_fixed_sized"> {
    let summary = "Appends a fixed sized value to an Arrow array builder.";
    let description = [{
        This operation appends a fixed sized value to an Arrow array builder, optionally with a validity flag.
        It can be used for any fixed sized type, such as integers, floats, decimals, dates, and timestamps.
    }];
    let arguments = (ins Arrow_ArrayBuilder:$builder, AnyType: $value, Optional<I1>: $valid);
    let assemblyFormat = " $builder `,` $value `:` type($value) ( `,` $valid^ )? attr-dict";
}
def Arrow_AppendVariableSizeBinaryOp : Arrow_Op<"array_builder.append_variable_sized_binary"> {
    let summary = "Appends a variable sized binary value to an Arrow array builder.";
    let description = [{
        This operation appends a variable sized binary value to an Arrow array builder, optionally with a validity flag
        The binary data is (at the moment) expected to be a util.varlen32 type, which contains pointer and length information.
        In the future, we should return a pointer and length directly.
    }];
    let arguments = (ins Arrow_ArrayBuilder:$builder, VarLen32Type: $value,  Optional<I1>: $valid);
    let assemblyFormat = " $builder `,` $value   ( `,` $valid^ )? attr-dict";
}

def Arrow_BuilderFromPtr : Arrow_Op<"array_builder.from_ptr", [Pure] > {
    let summary = "Creates a builder value from a pointer to an ArrowColumn builder that is managed in the runtime";
    let arguments = (ins RefType:$ptr);
    let results = (outs Arrow_ArrayBuilder : $builder);
    let assemblyFormat = " $ptr `->` type($builder) attr-dict";
}

#endif// OPS
