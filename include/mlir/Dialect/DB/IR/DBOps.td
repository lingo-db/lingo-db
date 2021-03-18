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
    return $_get($_ctxt, "false");
    }]>
    ];
}
def DB_IntType : DB_Type<"Int","int"> {
let summary = "database integer type";
let parameters = (ins "bool":$nullable,"unsigned":$width);
}
def DB_DateType : DB_Type<"Date","date"> {
let summary = "database date type";
let parameters = (ins "bool":$nullable);
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
    [ConstantLike, NoSideEffect]> {
let summary = "constant";


let arguments = (ins AnyAttr:$value);
let results = (outs DBType);

let extraClassDeclaration = [{
Attribute getValue() { return (*this)->getAttr("value"); }
}];

}

class DB_BinaryOp<string mnemonic, list<OpTrait> traits = []> :
Op<DB_Dialect, mnemonic,
    !listconcat(traits, [NoSideEffect,
        SameOperandsAndResultType])>,Arguments<(ins DBType:$lhs, DBType:$rhs)> {
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
def DB_NegOp : DB_BinaryOp<"neg"> {}

def DB_CMP_P_EQ  : I64EnumAttrCase<"eq", 0>;
def DB_CMP_P_NE  : I64EnumAttrCase<"ne", 1>;
def DB_CMP_P_SLT : I64EnumAttrCase<"slt", 2>;
def DB_CMP_P_SLE : I64EnumAttrCase<"sle", 3>;
def DB_CMP_P_SGT : I64EnumAttrCase<"sgt", 4>;
def DB_CMP_P_SGE : I64EnumAttrCase<"sge", 5>;
def DB_CMP_P_ULT : I64EnumAttrCase<"ult", 6>;
def DB_CMP_P_ULE : I64EnumAttrCase<"ule", 7>;
def DB_CMP_P_UGT : I64EnumAttrCase<"ugt", 8>;
def DB_CMP_P_UGE : I64EnumAttrCase<"uge", 9>;

def DB_CmpPredicateAttr : I64EnumAttr<
    "DBCmpPredicate", "",
    [DB_CMP_P_EQ, DB_CMP_P_NE, DB_CMP_P_SLT, DB_CMP_P_SLE, DB_CMP_P_SGT,
        DB_CMP_P_SGE, DB_CMP_P_ULT, DB_CMP_P_ULE, DB_CMP_P_UGT, DB_CMP_P_UGE]> {
let cppNamespace = "::mlir::db";
}

def DB_CmpOp : DB_Op<"compare",
    [NoSideEffect, SameTypeOperands]> {
let summary = "integer comparison operation";

let arguments = (ins
DB_CmpPredicateAttr:$predicate,
DBType:$lhs,
DBType:$rhs
);
let results = (outs DB_BoolType:$result);

let builders = [
OpBuilderDAG<(ins "CmpIPredicate":$predicate, "Value":$lhs,
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

def PrintOp : DB_Op<"print"> {
let summary = "print function";
let arguments = (ins StrAttr:$format, Variadic<DBType>: $to_print);
let assemblyFormat = "$format `,` $to_print `:` type($to_print) attr-dict";
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

#endif// OPS
