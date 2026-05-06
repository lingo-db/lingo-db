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

def SubOp_LocalTable : Type<CPred<"mlir::isa<::lingodb::compiler::dialect::subop::LocalTableType>($_self)">,"sub-op local table type">;

def Arrow_TableFromLocalTableOp : Arrow_Op<"table.from_local_table", [Pure]> {
    let summary = "View a sub-op local table as an opaque arrow.table handle";
    let description = [{
        Both types are runtime-equivalent (a `lingodb::runtime::ArrowTable*`),
        but at the MLIR-type level the conversion is explicit so we can hand the
        opaque handle to operations that don't carry a per-column schema
        (e.g. py_interp casts).

        The operand is typed `AnyType` rather than the strict `subop.local_table`
        because the bridge op survives past `lower-subop-to-cf`, which lowers
        the local_table into a plain `i8*` while the bridge op is still in
        place; using `AnyType` keeps the IR verifiable across that window.
    }];
    let arguments = (ins AnyType:$local_table);
    let results = (outs Arrow_Table:$arrow_table);
    let assemblyFormat = " $local_table `:` type($local_table) attr-dict";
}

def Arrow_TableToLocalTableOp : Arrow_Op<"table.to_local_table", [Pure]> {
    let summary = "Annotate an opaque arrow.table with a schema, producing a sub-op local table";
    let description = [{
        The target `local_table` type carries the column names + member types
        the surrounding pipeline (sub-op scans) expects. Verified at runtime
        that the underlying arrow::Table actually has that schema.

        Result is `AnyType` for the same reason as `from_local_table` above.
    }];
    let arguments = (ins Arrow_Table:$arrow_table);
    let results = (outs AnyType:$local_table);
    let assemblyFormat = " $arrow_table `->` type($local_table) attr-dict";
}

#endif// OPS
