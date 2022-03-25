#ifndef OPS
#define OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Dialect/DB/IR/DBInterfaces.td"
include "mlir/Dialect/RelAlg/IR/RelAlgInterfaces.td"

//===----------------------------------------------------------------------===//
// DB dialect definition.
//===----------------------------------------------------------------------===//

def DB_Dialect : Dialect {
let name = "db";
let cppNamespace = "::mlir::db";
let extraClassDeclaration = [{
private:
void registerTypes();
std::shared_ptr<mlir::db::RuntimeFunctionRegistry> runtimeFunctionRegistry;

public:
    std::shared_ptr<mlir::db::RuntimeFunctionRegistry> getRuntimeFunctionRegistry(){
        return runtimeFunctionRegistry;
    }
}];
}

class DB_Op<string mnemonic, list<Trait> traits = []> :
Op<DB_Dialect, mnemonic, traits>{
let printer = [{ return ::print(p, *this); }];
let parser = [{ return ::parse$cppClass(parser, result); }];
}



class DB_Type<string name, string typeMnemonic> : TypeDef<DB_Dialect, name> {
  let mnemonic = typeMnemonic;
}
def NullableType : DB_Type<"Nullable","nullable"> {
    let parameters = (ins "Type":$type);
    let assemblyFormat = " `<` $type `>`";
}

class DB_Collection<string name, string typeMnemonic> : TypeDef<DB_Dialect, name,[],"::mlir::db::CollectionType"> {
    let mnemonic = typeMnemonic;
}

def DBCollection : Type<CPred<"$_self.isa<::mlir::db::CollectionType>()">,"DB dialect collection">;

def DB_GenericIterable : DB_Collection<"GenericIterable","iterable"> {
let summary = "generic iterable";
let parameters = (ins "Type":$elementType,"std::string":$iteratorName);

}


def DB_Table: DB_Type<"Table","table"> {
let summary = "table";
let assemblyFormat = "";
}
def DB_Flag: DB_Type<"Flag","flag"> {
let summary = "flag";
let assemblyFormat = "";
}
def DB_TableBuilder : DB_Type<"TableBuilder","table_builder"> {
let summary = "builder for table";
let parameters = (ins "TupleType":$rowType);
let assemblyFormat = "`<` $rowType `>`";
}
def DB_Vector : DB_Collection<"Vector","vector"> {
let summary = "vector";
let parameters = (ins "Type":$elementType);
let assemblyFormat = "`<` $elementType `>`";
}

def DB_AggregationHashtable : DB_Collection<"AggregationHashtable","aggr_ht"> {
let summary = "aggregation hashtable";
let parameters = (ins "TupleType":$keyType,"TupleType":$valType);
let assemblyFormat = "`<` $keyType `,` $valType `>`";
}
def DB_JoinHashtable : DB_Collection<"JoinHashtable","join_ht"> {
let summary = "join hashtable";
let parameters = (ins "TupleType":$keyType,"TupleType":$valType);
let assemblyFormat = "`<` $keyType `,` $valType `>`";
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

def DB_CharType : DB_Type<"Char","char"> {
let parameters = (ins "unsigned":$bytes);
let assemblyFormat="`<` $bytes `>` ";
}
def DB_DateType : DB_Type<"Date","date"> {
let parameters = (ins "DateUnitAttr":$unit);
let assemblyFormat="`<` $unit  `>` ";
}
def DB_IntervalType : DB_Type<"Interval","interval"> {
let parameters = (ins "IntervalUnitAttr":$unit);
let assemblyFormat="`<` $unit  `>` ";
}
def DB_TimestampType : DB_Type<"Timestamp","timestamp"> {
let summary = "database date type";
let parameters = (ins "TimeUnitAttr":$unit);
let assemblyFormat="`<` $unit`>` ";
}

def DB_DecimalType : DB_Type<"Decimal","decimal"> {
let parameters = (ins "unsigned":$p,"unsigned":$s);
let assemblyFormat="`<` $p `,` $s  `>` ";
}
def DB_StringType : DB_Type<"String","string"> {}
def AnyBool: AnyTypeOf<[I1,NullableType]>;
def NullableDBType: AnyTypeOf<[AnyType,NullableType]>;

def DB_ConstantOp : DB_Op<"constant",
    [NoSideEffect]> {
let arguments = (ins AnyAttr:$value);
let results = (outs AnyType:$result);

let extraClassDeclaration = [{
Attribute getValue() { return (*this)->getAttr("value"); }
}];
let assemblyFormat = "`(` $value `)` `:` type($result) attr-dict";


}
def DB_NullOp : DB_Op<"null",
[NoSideEffect]> {
let summary = "null";
let results = (outs NullableType:$res);
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
let arguments = (ins DB_Flag:$flag, I1:$val);
let assemblyFormat = "$flag `,` $val attr-dict";
}
def DB_GetFlag : DB_Op<"getflag",
[NoSideEffect]> {
let summary = "get flag value";
let arguments = (ins DB_Flag:$flag);
let results= (outs I1:$res);
let assemblyFormat = "$flag attr-dict";
}

def DB_DeriveTruth : DB_Op<"derive_truth",
[NoSideEffect]> {
let arguments = (ins AnyType:$val);
let results= (outs I1:$res);
let assemblyFormat = "$val `:` type($val) attr-dict";
}
def DB_Hash : DB_Op<"hash", [NoSideEffect]> {
let arguments = (ins AnyType: $val);
let results = (outs  Index : $hash);
let assemblyFormat = "$val `:` type($val) attr-dict";
}

def DB_AsNullableOp : DB_Op<"as_nullable", [NoSideEffect]> {
let arguments = (ins AnyType: $val,Optional<I1>: $null);
let results = (outs  NullableType : $res);
let assemblyFormat = "$val `:` type($val) ( `,` $null^ )?  `->` type($res) attr-dict";
}
def DB_NullableGetVal : DB_Op<"nullable_get_val", [NoSideEffect,DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
let arguments = (ins NullableType: $val);
let results = (outs  AnyType : $res);
let assemblyFormat = "$val `:` type($val)  attr-dict";
  let extraClassDefinition = [{
    LogicalResult $cppClass::inferReturnTypes(
   MLIRContext* context, Optional<Location> location, ValueRange operands,
   DictionaryAttr attributes, RegionRange regions,
   SmallVectorImpl<Type>& inferredReturnTypes) {
     inferredReturnTypes.assign({operands[0].getType().cast<mlir::db::NullableType>().getType()});
     return success();
    }
  }];
}
def DB_CreateTableBuilder : DB_Op<"create_table_builder",
[NoSideEffect]> {
let arguments = (ins ArrayAttr:$columns);
let results = (outs DB_TableBuilder:$builder);
let assemblyFormat = "$columns `:` type($builder) attr-dict";
}

def DB_AddTableRow : DB_Op<"add_row"> {
let arguments = (ins AnyType:$builder,AnyType:$row);
let assemblyFormat = "$builder `:` type($builder) `,` $row `:` type($row) attr-dict";

}
def DB_CreateDS : DB_Op<"create_ds"> {
    let arguments =(ins Optional<AnyType>:$init_val);
    let results =(outs AnyType:$ds);
    let assemblyFormat = "( $init_val^ `:` type($init_val) `->`)? type($ds) attr-dict";
    let builders = [OpBuilder<(ins "mlir::Type":$t), [{
    build($_builder, $_state, t, mlir::Value());
    }]>];
}
def DB_HashtableInsert: DB_Op<"ht_insert"> {
    let arguments = (ins AnyType:$ht, AnyType:$key, Optional<AnyType>: $val);
    let assemblyFormat = "$ht `:` type($ht) `,` $key `:` type($key) ( `,` $val^ `:` type($val) )? attr-dict";
}
def DB_HashtableInsertReduce: DB_Op<"ht_insert_reduce"> {
    let arguments = (ins AnyType:$ht, AnyType:$key, Optional<AnyType>: $val);
    let regions = (region AnyRegion:$equal, AnyRegion:$reduce);
}

def DB_HashtableFinalize: DB_Op<"ht_finalize"> {
    let arguments = (ins AnyType:$ht );
    let assemblyFormat = "$ht `:` type($ht) attr-dict";
}
def DB_Append : DB_Op<"ds_append"> {
    let arguments = (ins AnyType:$ds, AnyType:$val);
    let assemblyFormat = "$ds `:` type($ds) `,` $val `:` type($val)  attr-dict";
}
def DB_FinalizeTable : DB_Op<"finalize_table",
[NoSideEffect]> {
let arguments = (ins AnyType:$builder);
let results = (outs DB_Table:$builded);
let assemblyFormat = "$builder `:` type($builder) attr-dict";
}

def DB_ScanSource : DB_Op<"scan_source",
[NoSideEffect]> {
let arguments = (ins StrAttr:$descr);
let results = (outs DB_GenericIterable:$iterable);
let assemblyFormat = "$descr `:` type($iterable) attr-dict";
}
def DB_Lookup : DB_Op<"lookup",
[NoSideEffect]> {
let arguments = (ins DBCollection:$collection,AnyType:$key);
let results = (outs DB_GenericIterable:$iterable);
let assemblyFormat = "$collection `:` type($collection) `,` $key `:` type($key) `->` type($iterable) attr-dict";
}
def DB_RuntimeCall : DB_Op<"runtime_call",[DB_HandleNullInterface]> {
let arguments = (ins StrAttr:$fn,Variadic<AnyType>: $args);
let results = (outs Optional<AnyType>:$res);
let assemblyFormat = "$fn  `(` $args `)` `:`  functional-type($args,$res)  attr-dict";
let verifier = [{ return ::verify(*this); }];
let extraClassDeclaration = [{
public:
    bool canHandleInvalidValues();
    bool canHandleNulls();
}];
}

def DB_SortOp : DB_Op<"sort",
[]> {
let arguments = (ins DB_Vector:$toSort);
let regions = (region SizedRegion<1>:$region);
}





def ForOp : DB_Op<"for",
[SingleBlockImplicitTerminator<"db::YieldOp">,
RecursiveSideEffects,AttrSizedOperandSegments]> {
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
let arguments = (ins NullableType:$val);
let results = (outs I1:$result);
let assemblyFormat = "$val `:` type($val) attr-dict";

}


class DB_BinaryOp<string mnemonic, list<Trait> traits = []> :
Op<DB_Dialect, mnemonic,
    !listconcat(traits, [NoSideEffect,DeclareOpInterfaceMethods<InferTypeOpInterface>,DeclareOpInterfaceMethods<DB_HandleNullInterface>])>,Arguments<(ins NullableDBType:$left, NullableDBType:$right)> {
let results = (outs AnyType:$result);
  let extraClassDefinition = [{
    LogicalResult $cppClass::inferReturnTypes(
   MLIRContext* context, Optional<Location> location, ValueRange operands,
   DictionaryAttr attributes, RegionRange regions,
   SmallVectorImpl<Type>& inferredReturnTypes) {
         return inferReturnType(context,location,operands,inferredReturnTypes);
    }
    bool $cppClass::canHandleInvalidValues(){
        return true;
    }
  }];
let assemblyFormat = "$left `:` type($left) `,` $right `:` type($right) attr-dict";
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
    [DB_CMP_P_EQ, DB_CMP_P_NEQ, DB_CMP_P_LT, DB_CMP_P_LTE, DB_CMP_P_GT,DB_CMP_P_GTE,DB_CMP_P_LIKE]> {
let cppNamespace = "::mlir::db";
}

def DB_CmpOp : DB_Op<"compare",
    [NoSideEffect,DeclareOpInterfaceMethods<InferTypeOpInterface>,DeclareOpInterfaceMethods<Relalg_CmpOpInterface>,DeclareOpInterfaceMethods<DB_HandleNullInterface>]> {
let arguments = (ins DB_CmpPredicateAttr:$predicate, NullableDBType:$left, NullableDBType:$right);
let results = (outs AnyBool:$result);
let assemblyFormat = "$predicate $left `:` type($left) `,` $right `:` type($right) attr-dict";
}

def DB_BetweenOp : DB_Op<"between",
    [NoSideEffect,DeclareOpInterfaceMethods<InferTypeOpInterface>]> {

let arguments = (ins NullableDBType:$val, NullableDBType:$lower, NullableDBType:$upper, I1Attr:$lowerInclusive, I1Attr:$upperInclusive);
let results = (outs AnyBool:$result);

  let extraClassDefinition = [{
    LogicalResult $cppClass::inferReturnTypes(MLIRContext* context, Optional<Location>, ValueRange operands,
   DictionaryAttr, RegionRange,SmallVectorImpl<Type>& inferredReturnTypes) {
    inferredReturnTypes.assign({constructNullableBool(context, operands)});
    return success();
    }
  }];
let assemblyFormat = "$val `:` type ($val) `between` $lower `:` type($lower) `,` $upper `:` type($upper) `,` `lowerInclusive` `:` $lowerInclusive `,` `upperInclusive` `:` $upperInclusive attr-dict";
}

def DB_OneOfOp : DB_Op<"oneof",
    [NoSideEffect,DeclareOpInterfaceMethods<InferTypeOpInterface>]> {

let arguments = (ins NullableDBType:$val,Variadic<NullableDBType>:$vals );
let results = (outs AnyBool:$result);

  let extraClassDefinition = [{
    LogicalResult $cppClass::inferReturnTypes(MLIRContext* context, Optional<Location>, ValueRange operands,
   DictionaryAttr, RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
    inferredReturnTypes.assign({constructNullableBool(context, operands)});
    return success();
    }
  }];
let assemblyFormat = "$val `:` type ($val) `?` $vals `:` type ($vals) attr-dict";
}





def AndOp : DB_Op<"and", [NoSideEffect,DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
let arguments = (ins Variadic<AnyBool>: $vals);
let results = (outs  AnyBool : $res);
let hasCanonicalizeMethod = 1;
let assemblyFormat = "$vals `:` type($vals) attr-dict";
  let extraClassDefinition = [{
    LogicalResult $cppClass::inferReturnTypes(MLIRContext* context, Optional<Location>, ValueRange operands,
   DictionaryAttr, RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
    inferredReturnTypes.assign({constructNullableBool(context, operands)});
    return success();
    }
  }];
}
def OrOp : DB_Op<"or", [NoSideEffect,DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
let arguments = (ins Variadic<AnyType>: $vals);
let results = (outs  AnyType : $res);
let assemblyFormat = "$vals `:` type($vals) attr-dict";
  let extraClassDefinition = [{
    LogicalResult $cppClass::inferReturnTypes(MLIRContext* context, Optional<Location>, ValueRange operands,
   DictionaryAttr, RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
    inferredReturnTypes.assign({constructNullableBool(context, operands)});
    return success();
    }
  }];
let hasCanonicalizeMethod = 1;
let assemblyFormat = "$vals `:` type($vals) attr-dict";

}
def NotOp : DB_Op<"not", [NoSideEffect,SameOperandsAndResultType,DeclareOpInterfaceMethods<DB_HandleNullInterface>]> {
let arguments = (ins AnyBool: $val);
let results = (outs  AnyBool : $res);
let assemblyFormat = "$val `:` type($val) attr-dict";
  let extraClassDefinition = [{
    bool $cppClass::canHandleInvalidValues(){ return true; }
  }];
}
def FreeOp : DB_Op<"free"> {
let arguments = (ins AnyType: $val);
let assemblyFormat = "$val `:` type($val) attr-dict";
}



def CastOp : DB_Op<"cast",[DeclareOpInterfaceMethods<DB_HandleNullInterface>]> {
let arguments = (ins NullableDBType:$val);
let results=(outs NullableDBType:$res);
let assemblyFormat = "$val `:` type($val) `->` type($res) attr-dict";
}



def YieldOp : DB_Op<"yield", [NoSideEffect, Terminator,
        ParentOneOf<["ForOp","SortOp","HashtableInsertReduce"]>]> {
let arguments = (ins Variadic<AnyType>:$results);
let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
let assemblyFormat = [{  attr-dict ($results^ `:` type($results))? }];
}
def CondSkipOp : DB_Op<"cond_skip", [ParentOneOf<["ForOp"]>]> {
let arguments = (ins AnyType:$condition, Variadic<AnyType>:$args);
let assemblyFormat = [{ `(` $condition `:` type($condition) `)` attr-dict ( `(` $args^ `:` type($args)`)`)? }];
}
#endif// OPS
