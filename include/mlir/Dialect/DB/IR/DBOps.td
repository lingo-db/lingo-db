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

def DB_Table: DB_Type<"Table","table"> {
let summary = "table";
let assemblyFormat = "";
}

def DB_TableBuilder : DB_Type<"TableBuilder","table_builder"> {
let summary = "builder for table";
let parameters = (ins "TupleType":$rowType);
let assemblyFormat = "`<` $rowType `>`";
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

def DB_DeriveTruth : DB_Op<"derive_truth",
[NoSideEffect]> {
let arguments = (ins AnyType:$val);
let results= (outs I1:$res);
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
def DB_FinalizeTable : DB_Op<"finalize_table",
[NoSideEffect]> {
let arguments = (ins AnyType:$builder);
let results = (outs DB_Table:$builded);
let assemblyFormat = "$builder `:` type($builder) attr-dict";
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



def DB_IsNullOp : DB_Op<"isnull",
[NoSideEffect]> {
let arguments = (ins NullableType:$val);
let results = (outs I1:$result);
let assemblyFormat = "$val `:` type($val) attr-dict";

}
def DB_Hash : DB_Op<"hash", [NoSideEffect]> {
let arguments = (ins AnyType: $val);
let results = (outs  Index : $hash);
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

def CastOp : DB_Op<"cast",[DeclareOpInterfaceMethods<DB_HandleNullInterface>]> {
let arguments = (ins NullableDBType:$val);
let results=(outs NullableDBType:$res);
let assemblyFormat = "$val `:` type($val) `->` type($res) attr-dict";
}
#endif// OPS
