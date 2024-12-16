#ifndef OPS
#define OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "lingodb/compiler/Dialect/DSA/IR/DSAInterfaces.td"
include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgInterfaces.td"
include "lingodb/compiler/Dialect/util/UtilBase.td"

//===----------------------------------------------------------------------===//
// DSA dialect definition.
//===----------------------------------------------------------------------===//

def DSA_Dialect : Dialect {
    let name = "dsa";
    let cppNamespace = "::lingodb::compiler::dialect::dsa";
    let extraClassDeclaration = [{
        private:
        void registerTypes();

        public:

    }];
        let useDefaultTypePrinterParser=1;
}

class DSA_Op<string mnemonic, list<Trait> traits = []> : Op<DSA_Dialect, mnemonic, traits> {
  let hasCustomAssemblyFormat = 1;

}

class DSA_Type<string name, string typeMnemonic, list<Trait> traits = []> : TypeDef<DSA_Dialect, name,traits> {
    let mnemonic = typeMnemonic;
}
def DSA_ArrowString : DSA_Type<"ArrowString","arrow_string"> {
	let summary = "arrow string type";
}
def DSA_ArrowDate32 : DSA_Type<"ArrowDate32","arrow_date32"> {
	let summary = "arrow date32 type";
}
def DSA_ArrowDate64 : DSA_Type<"ArrowDate64","arrow_date64"> {
	let summary = "arrow date64 type";
}
def DSA_ArrowMonthInterval : DSA_Type<"ArrowMonthInterval","arrow_month_interval"> {
	let summary = "arrow month interval type";
}
def DSA_ArrowDayTimeInterval : DSA_Type<"ArrowDayTimeInterval","arrow_day_time_interval"> {
	let summary = "arrow day time interval type";
}
def Arrow_TimeUnit_S : I64EnumAttrCase<"second", 0>;
def Arrow_TimeUnit_MS : I64EnumAttrCase<"millisecond", 1>;
def Arrow_TimeUnit_MUS : I64EnumAttrCase<"microsecond", 2>;
def Arrow_TimeUnit_NS : I64EnumAttrCase<"nanosecond", 3>;
def Arrow_TimeUnitAttr : I64EnumAttr<"TimeUnitAttr", "",
        [Arrow_TimeUnit_S,Arrow_TimeUnit_MS,Arrow_TimeUnit_MUS,Arrow_TimeUnit_NS]> {
    let cppNamespace = "::lingodb::compiler::dialect::dsa";
}
def DSA_ArrowTimeStamp : DSA_Type<"ArrowTimeStamp","arrow_timestamp"> {
	let summary = "arrow time stamp type";
	let parameters = (ins "TimeUnitAttr":$unit);
	let assemblyFormat="`<` $unit`>` ";
}
def DSA_ArrowDecimal : DSA_Type<"ArrowDecimal","arrow_decimal"> {
	let summary = "arrow decimal type";
	let parameters = (ins "int64_t":$p,"int64_t":$s);
	let assemblyFormat = "`<` $p`,` $s `>`";
}
def DSA_ArrowFixedSizedBinary : DSA_Type<"ArrowFixedSizedBinary","arrow_fixed_sized_binary"> {
	let summary = "arrow fixed sized binary type";
	let parameters = (ins "int64_t":$byte_width);
	let assemblyFormat = "`<` $byte_width `>`";
}

def DSA_ArrowList : DSA_Type<"ArrowList","arrow_list"> {
    let summary = "arrow list type";
    let description = [{
        A column of a table.
    }];
    let parameters = (ins "mlir::Type":$type);
    let assemblyFormat = "`<` $type `>`";
}
def DSA_Column : DSA_Type<"Column","column"> {
    let summary = "column";
    let description = [{
        A column of a table.
    }];
    let parameters = (ins "mlir::Type":$type);
    let assemblyFormat = "`<` $type `>`";
}
def DSA_ColumnBuilder : DSA_Type<"ColumnBuilder","column_builder"> {
    let summary = "column_builder";
    let description = [{
        A column builder of a table.
    }];
    let parameters = (ins "mlir::Type":$type);
    let assemblyFormat = "`<` $type `>`";
}

def DSA_Table : DSA_Type<"Table", "table"> {
    let summary = "result_table";
    let description = [{
        table with a fixed schema.
        }];

    let parameters = (ins "mlir::TupleType":$rowType);
    let assemblyFormat = "`<` $rowType `>`";
}


def DSA_Record: DSA_Type<"Record","record"> {
    let summary = "record";
    let parameters = (ins "mlir::TupleType":$rowType);
    let assemblyFormat = "`<` $rowType `>`";
}
def DSA_RecordBatch: DSA_Type<"RecordBatch","record_batch",[CollectionType]> {
    let summary = "record_batch";
    let parameters = (ins "mlir::TupleType":$rowType);
    let assemblyFormat = "`<` $rowType `>`";
    let extraClassDeclaration = [{
        mlir::Type getElementType();
    }];
}
def DSA_CreateTable : DSA_Op<"create_table"> {
    let summary = "create table from columns";
    let arguments = (ins ArrayAttr:$column_names,Variadic<DSA_Column>:$columns);
    let results = (outs DSA_Table:$table);
    let assemblyFormat = " $column_names `(` $columns `:` type($columns) `)` `->` type($table) attr-dict";
}
def DSA_FinishColumn : DSA_Op<"finish_column"> {
    let summary = "finish column";
    let arguments = (ins DSA_ColumnBuilder:$builder);
    let results = (outs DSA_Column:$column);
    let assemblyFormat = " $builder `:` type($builder) `->` type($column) attr-dict";
}


def DSA_CreateDS : DSA_Op<"create_ds"> {
    let summary = "create data structure";
    let description = [{
        Create a Vector/JoinHashtable/AggregationHashTable/TableBuilder.
    }];

    let arguments = (ins Optional<AnyType>:$init_val,OptionalAttr<AnyAttr>:$init_attr);
    let results = (outs AnyType:$ds);
    let assemblyFormat = " ( `(` $init_attr^ `)` `->`)? ( $init_val^ `:` type($init_val) `->`)? type($ds) attr-dict";
    let builders = [
        OpBuilder<(ins "mlir::Type":$t), [{
            build($_builder, $_state, t, mlir::Value());
        }]>,
        OpBuilder<(ins "mlir::Type":$t,"mlir::Value":$v), [{
            build($_builder, $_state, t,v, mlir::Attribute());
        }]>,
        OpBuilder<(ins "mlir::Type":$t,"mlir::Attribute":$attr), [{
            build($_builder, $_state, t,mlir::Value(),attr);
        }]>
    ];
}

def DSA_GetRecordBatchLen : DSA_Op<"get_record_batch_len", [Pure]> {
    let summary = "get length of record batch";
    let arguments = (ins DSA_RecordBatch:$batch);
    let results = (outs Index:$len);
    let assemblyFormat = "$batch `:` type($batch) attr-dict";
}


def DSA_GetRecord : DSA_Op<"get_record", [Pure]> {
    let summary = "access record in a batch";
    let arguments = (ins DSA_RecordBatch:$batch,Index:$pos);
    let results = (outs DSA_Record:$record);
    let assemblyFormat = "$batch `[` $pos `]` `:` type($batch) `->` type($record) attr-dict";
}


def DSA_At : DSA_Op<"at", [Pure]> {
    let summary = "access column in a record";

    let arguments = (ins DSA_Record:$collection,I32Attr:$pos);
    let results = (outs AnyType:$val,Optional<I1>:$valid);
    let assemblyFormat = "$collection  `[` $pos `]` `:` type($collection) `->` type($val) (`,` type($valid)^)? attr-dict";
}


def DSA_DownCast : DSA_Op<"down_cast"> {
    let summary = "set result";
    let arguments = (ins AnyType : $state);
    let results = (outs AnyType: $res);
    let assemblyFormat = [{ $state `:` type($state) `->` type($res) attr-dict }];
}
def DSA_SetResultOp : DSA_Op<"set_result"> {
    let summary = "set result";
    let arguments = (ins I32Attr:$result_id, AnyType : $state);
    let assemblyFormat = [{ $result_id $state `:` type($state)  attr-dict }];
}
def DSA_Concat : DSA_Op<"ds_concat"> {
    let summary = "concatenate two data structures";
    let arguments = (ins AnyType:$left, AnyType:$right);
    let results = (outs AnyType:$res);
    let assemblyFormat = "$left  `:` type($left) `,` $right `:` type($right) `->` type($res) attr-dict";
}

def DSA_Append : DSA_Op<"ds_append"> {
    let summary = "append column to Vector or last table row in a TableBuilder";

    let arguments = (ins AnyType:$ds, AnyType:$val,Optional<I1>:$valid);
    let assemblyFormat = "$ds `:` type($ds) `,` $val `:` type($val) ( `,` $valid^ )? attr-dict";
        let builders = [OpBuilder<(ins "mlir::Value":$v1, "mlir::Value":$v2), [{
        build($_builder, $_state, v1,v2, mlir::Value());
    }]>];
}
def DSA_ArrowTypeFrom : DSA_Op<"arrow_type_from"> {
	let summary = "create arrow type from mlir type";
	let arguments = (ins AnyType:$val);
	let results = (outs AnyType:$arrow_val);
	let assemblyFormat = "$val `:` type($val) `->` type($arrow_val) attr-dict";
}
def DSA_ArrowTypeTo : DSA_Op<"arrow_type_to"> {
	let summary = "create mlir type arrow type ";
	let arguments = (ins AnyType:$arrow_val);
	let results = (outs AnyType:$val);
	let assemblyFormat = "$arrow_val `:` type($arrow_val) `->` type($val) attr-dict";
}


#endif// OPS
