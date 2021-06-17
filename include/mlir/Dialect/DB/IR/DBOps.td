#ifndef OPS
#define OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "DBInterfaces.td"

//===----------------------------------------------------------------------===//
// DB dialect definition.
//===----------------------------------------------------------------------===//

def DB_Dialect : Dialect {
let name = "db";
let summary = "A db out-of-tree MLIR dialect.";
let description = [{
This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
}];
let cppNamespace = "::mlir::db";
let extraClassDeclaration = [{
private:
void registerTypes();

public:
}];
}

class DB_Op<string mnemonic, list<OpTrait> traits = []> :
Op<DB_Dialect, mnemonic, traits>{
let printer = [{ return ::print(p, *this); }];
let parser = [{ return ::parse$cppClass(parser, result); }];
}

class DB_Type<string name, string typeMnemonic> : TypeDef<DB_Dialect, name,[],"::mlir::db::DBType"> {
  let mnemonic = typeMnemonic;
}
def DBType : Type<CPred<"$_self.isa<::mlir::db::DBType>()">,"DB dialect type">;

class DB_Collection<string name, string typeMnemonic> : TypeDef<DB_Dialect, name,[],"::mlir::db::CollectionType"> {
    let mnemonic = typeMnemonic;
}
class DB_Builder<string name, string typeMnemonic>  : TypeDef<DB_Dialect, name,[],"::mlir::db::BuilderType"> {
    let mnemonic = typeMnemonic;
}

def DBCollection : Type<CPred<"$_self.isa<::mlir::db::CollectionType>()">,"DB dialect collection">;
def DBBuilder : Type<CPred<"$_self.isa<::mlir::db::BuilderType>()">,"DB dialect builder">;

def DB_GenericIterable : DB_Collection<"GenericIterable","iterable"> {
let summary = "generic iterable";
let parameters = (ins "Type":$elementType,"std::string":$iteratorName);

}

def DB_Range  : DB_Collection<"Range","range"> {
let summary = "range";
let parameters = (ins "Type":$elementType);
}
def DB_Table: DB_Type<"Table","table"> {
let summary = "table";
}
def DB_Flag: DB_Type<"Flag","flag"> {
let summary = "flag";
}
def DB_TableBuilder : DB_Builder<"TableBuilder","table_builder"> {
let summary = "builder for table";
let parameters = (ins "TupleType":$rowType);
}
def DB_Vector : DB_Collection<"Vector","vector"> {
let summary = "vector";
let parameters = (ins "Type":$elementType);
}
def DB_VectorBuilder : DB_Builder<"VectorBuilder","vector_builder"> {
let summary = "builder for vector";
let parameters = (ins "Type":$elementType);
}
def DB_AggregationHashtable : DB_Collection<"AggregationHashtable","aggr_ht"> {
let summary = "aggregation hashtable";
let parameters = (ins "TupleType":$keyType,"TupleType":$valType);
}
def DB_JoinHashtable : DB_Collection<"JoinHashtable","join_ht"> {
let summary = "join hashtable";
let parameters = (ins "TupleType":$keyType,"TupleType":$valType);
}
def DB_AggrHTBuilder : DB_Builder<"AggrHTBuilder","aggr_ht_builder"> {
let summary = "builder for aggregation hashtable";
let parameters = (ins "TupleType":$keyType,"TupleType":$valType);
}
def DB_JoinHTBuilder : DB_Builder<"JoinHTBuilder","join_ht_builder"> {
let summary = "builder for join hashtable";
let parameters = (ins "TupleType":$keyType,"TupleType":$valType);
}
def DB_BoolType : DB_Type<"Bool","bool"> {
  let summary = "database boolean type";
  let parameters = (ins "bool":$nullable);
    let builders = [
    TypeBuilder<(ins ), [{
    return $_get($_ctxt, false);
    }]>
    ];
let builderCall = "$_builder.getType<" # dialect.cppNamespace #
"::" # cppClassName # ">()";
}

def DB_TimeUnit_S  : I64EnumAttrCase<"second", 0>;
def DB_TimeUnit_MS  : I64EnumAttrCase<"millisecond", 1>;
def DB_TimeUnit_MUS : I64EnumAttrCase<"microsecond", 2>;
def DB_TimeUnit_NS : I64EnumAttrCase<"nanosecond", 3>;

def DB_TimeUnitAttr : I64EnumAttr<
        "TimeUnitAttr", "",
        [DB_TimeUnit_S,DB_TimeUnit_MS,DB_TimeUnit_MUS,DB_TimeUnit_NS]> {
let cppNamespace = "::mlir::db";

}
def DB_DateUnit_DAY  : I64EnumAttrCase<"day", 0>;
def DB_DateUnit_MS : I64EnumAttrCase<"millisecond", 1>;

def DB_DateUnitAttr : I64EnumAttr<
        "DateUnitAttr", "",
        [DB_DateUnit_DAY,DB_DateUnit_MS]> {
let cppNamespace = "::mlir::db";

}
def DB_IntervalUnit_MONTH  : I64EnumAttrCase<"months", 0>;
def DB_IntervalUnit_DAYTIME : I64EnumAttrCase<"daytime", 1>;

def DB_IntervalUnitAttr : I64EnumAttr<
        "IntervalUnitAttr", "",
        [DB_IntervalUnit_MONTH,DB_IntervalUnit_DAYTIME]> {
let cppNamespace = "::mlir::db";

}


def DB_UIntType : DB_Type<"UInt","uint"> {
let summary = "database unsigned integer type";
let parameters = (ins "bool":$nullable,"unsigned":$width);
}
def DB_IntType : DB_Type<"Int","int"> {
let summary = "database integer type";
let parameters = (ins "bool":$nullable,"unsigned":$width);
}
def DB_FloatType : DB_Type<"Float","float"> {
let summary = "database float type";
let parameters = (ins "bool":$nullable,"unsigned":$width);
}
def DB_DateType : DB_Type<"Date","date"> {
let summary = "database date type";
let parameters = (ins "bool":$nullable,"DateUnitAttr":$unit);
}
def DB_IntervalType : DB_Type<"Interval","interval"> {
let summary = "database interval type";
let parameters = (ins "bool":$nullable,"IntervalUnitAttr":$unit);
}
def DB_TimestampType : DB_Type<"Timestamp","timestamp"> {
let summary = "database date type";
let parameters = (ins "bool":$nullable,"TimeUnitAttr":$unit,"std::string":$tz);
let builders = [
TypeBuilder<(ins "bool":$nullable,"TimeUnitAttr":$unit), [{
return $_get($_ctxt, nullable,unit,"");
}]>
];
}
def DB_TimeType : DB_Type<"Time","time"> {
let summary = "database time type";
let parameters = (ins "bool":$nullable,"TimeUnitAttr":$unit);
}
def DB_DurationType : DB_Type<"Duration","duration"> {
let summary = "database duration type";
let parameters = (ins "bool":$nullable,"TimeUnitAttr":$unit);
}
def DB_DecimalType : DB_Type<"Decimal","decimal"> {
let summary = "database date type";
let parameters = (ins "bool":$nullable,"unsigned":$p,"unsigned":$s);
}
def DB_StringType : DB_Type<"String","string"> {
let summary = "database date type";
let parameters = (ins "bool":$nullable);
}

def DB_ConstantOp : DB_Op<"constant",
    [NoSideEffect]> {
let summary = "constant";


let arguments = (ins AnyAttr:$value);
let results = (outs DBType);

let extraClassDeclaration = [{
Attribute getValue() { return (*this)->getAttr("value"); }
}];

}
def DB_NullOp : DB_Op<"null",
[NoSideEffect]> {
let summary = "null";
let results = (outs DBType:$res);
let assemblyFormat = "`:` type($res) attr-dict";
}
def DB_CreateFlag : DB_Op<"createflag",
[NoSideEffect]> {
let summary = "create flag";
let results = (outs DB_Flag:$flag);
let assemblyFormat = "attr-dict";
}
def DB_SetFlag : DB_Op<"setflag"> {
let summary = "set flag to true";
let arguments = (ins DB_Flag:$flag,  DB_BoolType:$val);
let assemblyFormat = "$flag `,` $val attr-dict";
}
def DB_GetFlag : DB_Op<"getflag",
[NoSideEffect]> {
let summary = "get flag value";
let arguments = (ins DB_Flag:$flag);
let results= (outs DB_BoolType:$res);
let assemblyFormat = "$flag attr-dict";
}
def DB_Hash : DB_Op<"hash", [NoSideEffect]> {
let arguments = (ins AnyType: $val);
let results = (outs  Index : $hash);
let assemblyFormat = "$val `:` type($val) attr-dict";
}

def DB_CombineNullOp : DB_Op<"combine_null", [NoSideEffect]> {
let arguments = (ins DBType: $val,DB_BoolType: $null);
let results = (outs  DBType : $res);
let assemblyFormat = "$val `:` type($val) `,` $null  `:` type($res) attr-dict";
}
def DB_CreateTableBuilder : DB_Op<"create_table_builder",
[NoSideEffect]> {
let summary = "create table builder";
let arguments = (ins ArrayAttr:$columns);
let results = (outs DB_TableBuilder:$builder);
let assemblyFormat = "$columns `:` type($builder) attr-dict";
}
def DB_CreateVectorBuilder : DB_Op<"create_vector_builder",
[NoSideEffect]> {
let summary = "create vector builder";
let results = (outs DB_VectorBuilder:$builder);
let assemblyFormat = " `:` type($builder) attr-dict";
}
def DB_BuilderMerge : DB_Op<"builder_merge",
[NoSideEffect]> {
let summary = "merge value";
let arguments = (ins DBBuilder:$builder,AnyType:$val);
let results = (outs DBBuilder:$result_builder);
let assemblyFormat = "$builder `:` type($builder) `,` $val `:`  type($val) `->` type($result_builder) attr-dict";
}
def DB_BuilderBuild : DB_Op<"builder_build",
[NoSideEffect]> {
let summary = "build builder";
let arguments = (ins DBBuilder:$builder);
let results = (outs AnyType:$builded);
let assemblyFormat = "$builder `:` type($builder) `->` type($builded) attr-dict";
}
def DB_GetTable : DB_Op<"get_table",
[NoSideEffect]> {
let summary = "get table";
let arguments = (ins StrAttr:$tablename,AnyType:$execution_context);
let results = (outs DB_Table:$table);
let assemblyFormat = "$tablename $execution_context `:` type($execution_context) attr-dict";
}
def DB_TableScan : DB_Op<"tablescan",
[NoSideEffect]> {
let summary = "tablescan";
let arguments = (ins DB_Table:$table,ArrayAttr:$columns);
let results = (outs DB_GenericIterable:$iterable);
let assemblyFormat = "$table $columns `:` type($iterable) attr-dict";
}
def DB_Lookup : DB_Op<"lookup",
[NoSideEffect]> {
let summary = "lookup";
let arguments = (ins DBCollection:$collection,AnyType:$key);
let results = (outs DB_GenericIterable:$iterable);
let assemblyFormat = "$collection `:` type($collection) `,` $key `:` type($key) `->` type($iterable) attr-dict";
}

def DB_CreateRange : DB_Op<"range",
[SameTypeOperands,NoSideEffect]> {
let summary = "create a range of values";
let arguments = (ins DBType:$lower,DBType:$upper,DBType:$step);
let results = (outs DB_Range:$range);
let assemblyFormat = "$lower `,` $upper `,` $step `:` type($lower) `->` type($range) attr-dict";

}
def DB_SortOp : DB_Op<"sort",
[NoSideEffect]> {
let summary = "sort";
let arguments = (ins DB_Vector:$toSort);
let results = (outs DB_Vector:$sorted);
let regions = (region SizedRegion<1>:$region);
}

def DB_CreateAggrHTBuilder : DB_Op<"create_aggr_ht_builder",
[NoSideEffect]> {
let summary = "create aggr hashtable builder";
let arguments = (ins AnyType: $initial);
let results = (outs DB_AggrHTBuilder:$builder);
let regions = (region SizedRegion<1>:$region);
}
def DB_CreateJoinHTBuilder : DB_Op<"create_join_ht_builder",
[NoSideEffect]> {
let summary = "create join hashtable builder";
let results = (outs DB_JoinHTBuilder:$builder);
let assemblyFormat = " `:` type($builder) attr-dict";
}


def ForOp : DB_Op<"for",
[SingleBlockImplicitTerminator<"db::YieldOp">,
RecursiveSideEffects,AttrSizedOperandSegments]> {
let summary = "for operation";

let arguments = (ins DBCollection:$collection,Optional<DB_Flag>:$until,Variadic<AnyType>:$initArgs);
let results = (outs Variadic<AnyType>:$results);
let regions = (region SizedRegion<1>:$region);



let extraClassDeclaration = [{

Value getInductionVar() { return getBody()->getArgument(0); }
Block::BlockArgListType getRegionIterArgs() {
    return getBody()->getArguments().drop_front();
}
Operation::operand_range getIterOperands() {
    return getOperands().drop_front(getNumControlOperands());
}


/// Number of region arguments for loop-carried values
unsigned getNumRegionIterArgs() {
    return getBody()->getNumArguments() - 1;
}
/// Number of operands controlling the loop: lb, ub, step
unsigned getNumControlOperands() { return until()?2:1; }
/// Does the operation hold operands for loop-carried values
bool hasIterOperands() {
    return getOperation()->getNumOperands() > getNumControlOperands();
}
/// Get Number of loop-carried values
unsigned getNumIterOperands() {
    return getOperation()->getNumOperands() - getNumControlOperands();
}


}];

}



def DB_IsNullOp : DB_Op<"isnull",
[NoSideEffect]> {
let summary = "isnull";
let arguments = (ins DBType:$val);
let results = (outs DB_BoolType:$result);
let assemblyFormat = "$val `:` type($val) attr-dict";

}


class DB_BinaryOp<string mnemonic, list<OpTrait> traits = []> :
Op<DB_Dialect, mnemonic,
    !listconcat(traits, [NoSideEffect])>,Arguments<(ins DBType:$left, DBType:$right)> {
let results = (outs AnyType:$result);
let parser = [{
return parseImplicitResultSameOperandBaseTypeOp(parser, result);
}];

let printer = [{
return printImplicitResultSameOperandBaseTypeOp(this->getOperation(), p);
}];
}
class DB_UnaryOp<string mnemonic, list<OpTrait> traits = []> :
Op<DB_Dialect, mnemonic,
    !listconcat(traits, [NoSideEffect,
        SameOperandsAndResultType])>,Arguments<(ins DBType:$val)> {
let results = (outs AnyType:$result);
let parser = [{
return parseImplicitResultSameOperandBaseTypeOp(parser, result);
}];

let printer = [{
return printImplicitResultSameOperandBaseTypeOp(this->getOperation(), p);
}];
}
def DB_AddOp : DB_BinaryOp<"add"> {}
def DB_SubOp : DB_BinaryOp<"sub"> {}
def DB_MulOp : DB_BinaryOp<"mul"> {}
def DB_DivOp : DB_BinaryOp<"div"> {}
def DB_ModOp : DB_BinaryOp<"mod"> {}

def DB_CMP_P_EQ  : I64EnumAttrCase<"eq", 0>;
def DB_CMP_P_NEQ  : I64EnumAttrCase<"neq", 1>;
def DB_CMP_P_LT : I64EnumAttrCase<"lt", 2>;
def DB_CMP_P_LTE : I64EnumAttrCase<"lte", 3>;
def DB_CMP_P_GT : I64EnumAttrCase<"gt", 4>;
def DB_CMP_P_GTE : I64EnumAttrCase<"gte", 5>;
def DB_CMP_P_LIKE : I64EnumAttrCase<"like", 6>;

def DB_CmpPredicateAttr : I64EnumAttr<
    "DBCmpPredicate", "",
    [DB_CMP_P_EQ, DB_CMP_P_NEQ, DB_CMP_P_LT, DB_CMP_P_LTE, DB_CMP_P_GT,
        DB_CMP_P_GTE,DB_CMP_P_LIKE]> {
let cppNamespace = "::mlir::db";
}

def DB_CmpOp : DB_Op<"compare",
    [NoSideEffect]> {
let summary = "integer comparison operation";

let arguments = (ins
DB_CmpPredicateAttr:$predicate,
DBType:$left,
DBType:$right
);
let results = (outs DB_BoolType:$result);

let builders = [
OpBuilder<(ins "mlir::db::DBCmpPredicate":$predicate, "Value":$left,
"Value":$right), [{
::buildDBCmpOp($_builder, $_state, predicate, left, right);
}]>];

let extraClassDeclaration = [{
static StringRef getPredicateAttrName() { return "predicate"; }
static CmpIPredicate getPredicateByName(StringRef name);

CmpIPredicate getPredicate() {
  return (CmpIPredicate)(*this)->getAttrOfType<IntegerAttr>(
      getPredicateAttrName()).getInt();
}
}];
}


def DumpOp : DB_Op<"dump"> {
let summary = "dump function";
let arguments = (ins DBType: $val);
let assemblyFormat = "$val `:` type($val) attr-dict";
}
def DumpIndexOp : DB_Op<"dump_index"> {
let summary = "dump index";
let arguments = (ins Index: $val);
let assemblyFormat = "$val attr-dict";
}


def AndOp : DB_Op<"and", [NoSideEffect]> {
let arguments = (ins Variadic<DB_BoolType>: $vals);
let results = (outs  DB_BoolType : $res);
let parser = [{
return parseImplicitResultSameOperandBaseTypeOp(parser, result);
}];
let printer = [{
return printImplicitResultSameOperandBaseTypeOp(this->getOperation(), p);
}];
}
def OrOp : DB_Op<"or", [NoSideEffect]> {
let arguments = (ins Variadic<DB_BoolType>: $vals);
let results = (outs  DB_BoolType : $res);
let parser = [{
return parseImplicitResultSameOperandBaseTypeOp(parser, result);
}];
let printer = [{
return printImplicitResultSameOperandBaseTypeOp(this->getOperation(), p);
}];
}
def NotOp : DB_Op<"not", [NoSideEffect,SameOperandsAndResultType]> {
let arguments = (ins DB_BoolType: $val);
let results = (outs  DB_BoolType : $res);
let assemblyFormat = "$val `:` type($val) attr-dict";
}

def DateAddOp : DB_Op<"date_add"> {
let summary = "date add function";
let arguments = (ins DB_DateType:$left, DB_IntervalType:$right );
let results = (outs  DB_DateType : $res);
let parser = [{
return parseDateOp(parser, result);
}];
let printer = [{
return printDateOp(this->getOperation(), p);
}];
}

def DB_ExtractableTimeUnit_S  : I64EnumAttrCase<"second", 0>;
def DB_ExtractableTimeUnit_MIN  : I64EnumAttrCase<"minute", 1>;
def DB_ExtractableTimeUnit_H  : I64EnumAttrCase<"hour", 2>;
def DB_ExtractableTimeUnit_D  : I64EnumAttrCase<"day", 3>;
def DB_ExtractableTimeUnit_DOW  : I64EnumAttrCase<"dow", 4>;
def DB_ExtractableTimeUnit_W  : I64EnumAttrCase<"week", 5>;
def DB_ExtractableTimeUnit_MON  : I64EnumAttrCase<"month", 6>;
def DB_ExtractableTimeUnit_Q  : I64EnumAttrCase<"quarter", 7>;
def DB_ExtractableTimeUnit_DOY  : I64EnumAttrCase<"doy", 8>;
def DB_ExtractableTimeUnit_Y  : I64EnumAttrCase<"year", 9>;
def DB_ExtractableTimeUnit_DEC  : I64EnumAttrCase<"decade", 10>;
def DB_ExtractableTimeUnit_CEN  : I64EnumAttrCase<"century", 11>;
def DB_ExtractableTimeUnit_MIL  : I64EnumAttrCase<"millennium", 12>;

def DB_ExtractableTimeUnitAttr : I64EnumAttr<
        "ExtractableTimeUnitAttr", "",
        [DB_ExtractableTimeUnit_S,DB_ExtractableTimeUnit_MIN,DB_ExtractableTimeUnit_H,DB_ExtractableTimeUnit_D,
                DB_ExtractableTimeUnit_DOW,DB_ExtractableTimeUnit_W,DB_ExtractableTimeUnit_MON,DB_ExtractableTimeUnit_Q,
                DB_ExtractableTimeUnit_DOY,DB_ExtractableTimeUnit_Y,DB_ExtractableTimeUnit_DEC,DB_ExtractableTimeUnit_CEN,
                DB_ExtractableTimeUnit_MIL]> {
let cppNamespace = "::mlir::db";
}
def DateSubOp : DB_Op<"date_sub"> {
let summary = "date sub function";
let arguments = (ins DB_DateType:$left, DB_IntervalType:$right );
let results = (outs  DB_DateType : $res);
let parser = [{
return parseDateOp(parser, result);
}];
let printer = [{
return printDateOp(this->getOperation(), p);
}];
}

def DateExtractOp : DB_Op<"date_extract"> {
let summary = "date sub function";
let arguments = (ins DB_ExtractableTimeUnitAttr:$unit, DB_DateType:$val );
let results = (outs  DB_IntType : $res);
}
def CastOp : DB_Op<"cast"> {
let summary = "cast function";
let arguments = (ins DBType:$val);
let results=(outs DBType:$res);
let assemblyFormat = "$val `:` type($val) `->` type($res) attr-dict";
}
def TypeCastOp : DB_Op<"type_cast"> {
let summary = "type cast function";
let arguments = (ins AnyType:$val);
let results=(outs AnyType:$res);
let assemblyFormat = "$val `:` type($val) `->` type($res) attr-dict";
}

def IfOp : DB_Op<"if",
[SingleBlockImplicitTerminator<"db::YieldOp">, RecursiveSideEffects,
NoRegionArguments]> {
let summary = "if-then-else operation";

let arguments = (ins AnyType:$condition);
let results = (outs Variadic<AnyType>:$results);
let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);
}

def ConditionOp : DB_Op<"condition",
[HasParent<"WhileOp">, NoSideEffect, Terminator]> {
let summary = "loop continuation condition";


let arguments = (ins AnyType:$condition, Variadic<AnyType>:$args);

let assemblyFormat =
[{ `(` $condition `:` type($condition) `)` attr-dict ($args^ `:` type($args))? }];
}

def SelectOp : DB_Op<"select", [NoSideEffect,
        AllTypesMatch<["true_value", "false_value", "result"]>] #
ElementwiseMappable.traits> {


let arguments = (ins DB_BoolType:$condition,
AnyType:$true_value,
AnyType:$false_value);
let results = (outs AnyType:$result);

let builders = [
OpBuilder<(ins "Value":$condition, "Value":$trueValue,
"Value":$falseValue), [{
$_state.addOperands({condition, trueValue, falseValue});
$_state.addTypes(trueValue.getType());
}]>];

let extraClassDeclaration = [{
Value getCondition() { return condition(); }
Value getTrueValue() { return true_value(); }
Value getFalseValue() { return false_value(); }
}];

}
def WhileOp : DB_Op<"while",
[RecursiveSideEffects]> {
let summary = "a generic 'while' loop";


let arguments = (ins Variadic<AnyType>:$inits);
let results = (outs Variadic<AnyType>:$results);
let regions = (region SizedRegion<1>:$before, SizedRegion<1>:$after);

let extraClassDeclaration = [{
OperandRange getSuccessorEntryOperands(unsigned index);
}];
}

def YieldOp : DB_Op<"yield", [NoSideEffect, Terminator,
        ParentOneOf<["IfOp","ForOp","WhileOp","SortOp","CreateAggrHTBuilder"]>]> {
let summary = "loop yield and termination operation";

let arguments = (ins Variadic<AnyType>:$results);
let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

let assemblyFormat =
[{  attr-dict ($results^ `:` type($results))? }];

// Override default verifier (defined in SCF_Op), no custom verification
// needed.
let verifier = ?;
}
#endif// OPS
