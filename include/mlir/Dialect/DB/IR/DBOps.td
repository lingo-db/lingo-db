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

class DB_Type<string name, string typeMnemonic> : TypeDef<DB_Dialect, name,"::mlir::db::DBType"> {
  let mnemonic = typeMnemonic;
}
def DBType : Type<CPred<"$_self.isa<::mlir::db::DBType>()">,"DB dialect type">;

class DB_Collection<string name, string typeMnemonic> : TypeDef<DB_Dialect, name,"::mlir::db::CollectionType"> {
    let mnemonic = typeMnemonic;
    let parameters = (ins "TypeRange":$types);
    let genStorageClass = 0;
}
def DB_MaterializedCollection : DB_Collection<"MaterializedCollection","matcollection"> {
let summary = "materialized collection";
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
let parameters = (ins "bool":$nullable);
}
def DB_IntervalType : DB_Type<"Interval","interval"> {
let summary = "database interval type";
let parameters = (ins "bool":$nullable,"std::string":$unit);
}
def DB_TimeType : DB_Type<"Timestamp","timestamp"> {
let summary = "database date type";
let parameters = (ins "bool":$nullable);
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

def DB_IsNullOp : DB_Op<"isnull",
[NoSideEffect]> {
let summary = "isnull";
let arguments = (ins DBType:$val);
let results = (outs DB_BoolType:$result);
let assemblyFormat = "$val `:` type($val) attr-dict";

}


class DB_BinaryOp<string mnemonic, list<OpTrait> traits = []> :
Op<DB_Dialect, mnemonic,
    !listconcat(traits, [NoSideEffect])>,Arguments<(ins DBType:$lhs, DBType:$rhs)> {
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
DBType:$lhs,
DBType:$rhs
);
let results = (outs DB_BoolType:$result);

let builders = [
OpBuilder<(ins "mlir::db::DBCmpPredicate":$predicate, "Value":$lhs,
"Value":$rhs), [{
::buildDBCmpOp($_builder, $_state, predicate, lhs, rhs);
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
let arguments = (ins StrAttr:$unit, DB_DateType:$date );
let results = (outs  DB_IntType : $res);
}
def CastOp : DB_Op<"cast"> {
let summary = "cast function";
let arguments = (ins DBType:$val);
let results=(outs DBType:$res);
let assemblyFormat = "$val `:` type($val) `->` type($res) attr-dict";
}


def IfOp : DB_Op<"if",
[SingleBlockImplicitTerminator<"db::YieldOp">, RecursiveSideEffects,
NoRegionArguments]> {
let summary = "if-then-else operation";

let arguments = (ins DB_BoolType:$condition);
let results = (outs Variadic<AnyType>:$results);
let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);
}




def YieldOp : DB_Op<"yield", [NoSideEffect, Terminator,
        ParentOneOf<["IfOp"]>]> {
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
