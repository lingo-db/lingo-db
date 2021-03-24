#ifndef OPS
#define OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

include "RelAlgInterfaces.td"

//===----------------------------------------------------------------------===//
// realg dialect definition.
//===----------------------------------------------------------------------===//

def RelAlg_Dialect : Dialect {
let name = "relalg";
let summary = "A dialect for relational algebra";
let cppNamespace = "::mlir::relalg";

let extraClassDeclaration = [{
private:
RelationalAttributeManager relationalAttributeManager;
public:
RelationalAttributeManager& getRelationalAttributeManager(){
    return relationalAttributeManager;
}
}];
}

//===----------------------------------------------------------------------===//
// Base relAlg operation definition.
//===----------------------------------------------------------------------===//

class RelAlg_Op<string mnemonic, list<OpTrait> traits = []> :
Op<RelAlg_Dialect, mnemonic, traits>{
let printer = [{ return ::print(p, *this); }];
let parser = [{ return ::parse$cppClass(parser, result); }];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Types
////////////////////////////////////////////////////////////////////////////////////////////////////
def DBType : Type<CPred<"$_self.isa<::mlir::db::DBType>()">,"DB dialect type">;
def DB_BoolType : Type<CPred<"$_self.isa<::mlir::db::BoolType>()">,"DB bool type">{
let builderCall = "::mlir::db::BoolType::get($_builder.getContext(),false)";
}
def DB_CollectionType : Type<CPred<"$_self.isa<::mlir::db::CollectionType>()">,"DB collection type">;
def DB_IntType : Type<CPred<"$_self.isa<::mlir::db::IntType>()">,"DB int type">{
let builderCall = "::mlir::db::IntType::get($_builder.getContext(),false,64)";
}

def RelAlg_Relation :  TypeDef<RelAlg_Dialect, "Relation"> {
let summary = "relation type";
}
def RelAlg_Tuple :  TypeDef<RelAlg_Dialect, "Tuple"> {
let summary = "tuple type";
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// Attributes
////////////////////////////////////////////////////////////////////////////////////////////////////
def RelationalAttributeDefAttr :
DialectAttr<RelAlg_Dialect,
    CPred<"$_self.isa<RelationalAttributeDefAttr>()">, "AttributeDefAttr">{
let storageType = "::mlir::relalg::RelationalAttributeDefAttr";
let returnType = "::mlir::relalg::RelationalAttributeDefAttr";
let convertFromStorage = "$_self";
}
def RelationalAttributeRefAttr :
DialectAttr<RelAlg_Dialect,
    CPred<"$_self.isa<RelationalAttributeRefAttr>()">, "AttributeRefAttr">{
let storageType = "::mlir::relalg::RelationalAttributeRefAttr";
let returnType = "::mlir::relalg::RelationalAttributeRefAttr";
let convertFromStorage = "$_self";
}
def SortSpecificationAttr :
DialectAttr<RelAlg_Dialect,
        CPred<"$_self.isa<SortSpecificationAttr>()">, "SortSpecificationAttr">{
let storageType = "::mlir::relalg::SortSpecificationAttr";
let returnType = "::mlir::relalg::SortSpecificationAttr";
let convertFromStorage = "$_self";
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// Enums
////////////////////////////////////////////////////////////////////////////////////////////////////

def RelAlg_SortSpec_DESC  : I64EnumAttrCase<"desc", 0>;
def RelAlg_AggrFunc_ASC  : I64EnumAttrCase<"asc", 1>;
def RelAlg_SortSpec : I64EnumAttr<
        "SortSpec", "",
        [RelAlg_SortSpec_DESC,RelAlg_AggrFunc_ASC]> {
let cppNamespace = "::mlir::relalg";
}



def RelAlg_AggrFunc_Sum  : I64EnumAttrCase<"sum", 0>;
def RelAlg_AggrFunc_Min  : I64EnumAttrCase<"min", 1>;
def RelAlg_AggrFunc_Max  : I64EnumAttrCase<"max", 2>;
def RelAlg_AggrFunc_Avg  : I64EnumAttrCase<"avg", 3>;
def RelAlg_AggrFunc_Count  : I64EnumAttrCase<"count", 4>;
def RelAlg_AggrFunc : I64EnumAttr<
        "AggrFunc", "",
        [RelAlg_AggrFunc_Sum,RelAlg_AggrFunc_Min,RelAlg_AggrFunc_Max,RelAlg_AggrFunc_Avg,RelAlg_AggrFunc_Count]> {
let cppNamespace = "::mlir::relalg";
}
def RelAlg_JoinType_Inner  : I64EnumAttrCase<"inner", 0>;
def RelAlg_JoinType_Semi  : I64EnumAttrCase<"semi", 1>;
def RelAlg_JoinType_AntiSemi  : I64EnumAttrCase<"antisemi", 2>;
def RelAlg_JoinType_Left  : I64EnumAttrCase<"left", 3>;
def RelAlg_JoinType_Right  : I64EnumAttrCase<"right", 4>;
def RelAlg_JoinType_Full  : I64EnumAttrCase<"full", 5>;


def RelAlg_NormalJoinType : I64EnumAttr<
        "NormalJoinType", "",
        [RelAlg_JoinType_Inner,RelAlg_JoinType_Semi,RelAlg_JoinType_AntiSemi]> {
let cppNamespace = "::mlir::relalg";
}
def RelAlg_OuterJoinType : I64EnumAttr<
        "OuterJoinType", "",
        [RelAlg_JoinType_Left,RelAlg_JoinType_Right,RelAlg_JoinType_Full]> {
let cppNamespace = "::mlir::relalg";
}



def RelAlg_SetSemantic_Distinct  : I64EnumAttrCase<"distinct", 0>;
def RelAlg_SetSemantic_All : I64EnumAttrCase<"all", 1>;


def RelAlg_SetSemantic : I64EnumAttr<
        "SetSemantic", "",
        [RelAlg_SetSemantic_Distinct,RelAlg_SetSemantic_All]> {
let cppNamespace = "::mlir::relalg";
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Operations that "create" relations
////////////////////////////////////////////////////////////////////////////////////////////////////
def ConstRelationOp : RelAlg_Op<"const_relation",
        [ RecursiveSideEffects,DeclareOpInterfaceMethods<Operator>]> {
let summary = "constant relation";
let arguments = (ins SymbolNameAttr:$sym_name,ArrayAttr:$attributes,ArrayAttr:$values);
let results = (outs RelAlg_Relation:$result);
let extraClassDeclaration = [{
llvm::SmallPtrSet<mlir::relalg::RelationalAttribute*,8> getCreatedAttributes();
}];
}

def BaseTableOp : RelAlg_Op<"basetable",
[SingleBlockImplicitTerminator<"relalg::ReturnOp">,RecursiveSideEffects,DeclareOpInterfaceMethods<Operator>]> {
let summary = "base table operation";
let arguments = (ins SymbolNameAttr:$sym_name, StrAttr: $table_identifier, DictionaryAttr: $columns);
let results = (outs RelAlg_Relation:$result);
let extraClassDeclaration = [{
llvm::SmallPtrSet<mlir::relalg::RelationalAttribute*,8> getCreatedAttributes();
}];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Operators
////////////////////////////////////////////////////////////////////////////////////////////////////

def SelectionOp : RelAlg_Op<"selection",[RecursiveSideEffects,DeclareOpInterfaceMethods<Operator>,DeclareOpInterfaceMethods<LoopLikeOpInterface>,TupleLamdaOperator]> {
let summary = "selection operation";
let arguments = (ins RelAlg_Relation:$rel);
let results = (outs RelAlg_Relation:$result);
let regions = (region SizedRegion<1>:$predicate);
}
def MapOp : RelAlg_Op<"map",[RecursiveSideEffects,DeclareOpInterfaceMethods<Operator>,DeclareOpInterfaceMethods<LoopLikeOpInterface>,TupleLamdaOperator]> {
let summary = "map operation";
let arguments = (ins SymbolNameAttr:$sym_name, RelAlg_Relation:$rel);
let results = (outs RelAlg_Relation:$result);
let regions = (region SizedRegion<1>:$predicate);
}
def LimitOp : RelAlg_Op<"limit",[NoSideEffect]> {
let summary = "limit operation";
let arguments = (ins I32Attr:$rows, RelAlg_Relation:$rel);
let results = (outs RelAlg_Relation:$result);
let assemblyFormat =
[{ $rows $rel attr-dict }];
}
def CrossProductOp : RelAlg_Op<"crossproduct",[RecursiveSideEffects,DeclareOpInterfaceMethods<Operator>]> {
let summary = "crossproduct operation";
let arguments = (ins RelAlg_Relation:$left,RelAlg_Relation:$right);
let results = (outs RelAlg_Relation:$result);
let assemblyFormat =
[{ $left `,` $right attr-dict }];
}

def AggregationOp : RelAlg_Op<"aggregation",[RecursiveSideEffects,DeclareOpInterfaceMethods<Operator>,DeclareOpInterfaceMethods<LoopLikeOpInterface>]> {
let summary = "aggregation operation";
let arguments = (ins SymbolNameAttr:$sym_name, RelAlg_Relation:$rel, ArrayAttr:$group_by_attrs);
let results = (outs RelAlg_Relation:$result);
let regions = (region SizedRegion<1>:$aggr_func);
let extraClassDeclaration = [{
llvm::SmallPtrSet<mlir::relalg::RelationalAttribute*,8> getUsedAttributes();
}];
}
def SortOp : RelAlg_Op<"sort",[NoSideEffect,DeclareOpInterfaceMethods<Operator>]> {
let summary = "sort operation";
let arguments = (ins RelAlg_Relation:$rel, ArrayAttr:$sortspecs);
let results = (outs RelAlg_Relation:$result);
let extraClassDeclaration = [{
llvm::SmallPtrSet<mlir::relalg::RelationalAttribute*,8> getUsedAttributes();
}];
}

def JoinOp : RelAlg_Op<"join",[RecursiveSideEffects,DeclareOpInterfaceMethods<LoopLikeOpInterface>,DeclareOpInterfaceMethods<Operator>,TupleLamdaOperator]> {
let summary = "normal join operation";
let arguments = (ins RelAlg_NormalJoinType:$type,RelAlg_Relation:$left,RelAlg_Relation:$right);
let results = (outs RelAlg_Relation:$result);
let regions = (region SizedRegion<1>:$predicate);
}
def OuterJoinOp : RelAlg_Op<"outerjoin",[RecursiveSideEffects,DeclareOpInterfaceMethods<LoopLikeOpInterface>,DeclareOpInterfaceMethods<Operator>,TupleLamdaOperator]> {
let summary = "outer join operation";
let arguments = (ins RelAlg_OuterJoinType:$type,RelAlg_Relation:$left,RelAlg_Relation:$right);
let results = (outs RelAlg_Relation:$result);
let regions = (region SizedRegion<1>:$predicate);
}

def DistinctOp : RelAlg_Op<"distinct",[RecursiveSideEffects,DeclareOpInterfaceMethods<Operator>]> {
let summary = "distinct operation";
let arguments = (ins RelAlg_Relation:$rel, ArrayAttr:$attrs);
let results = (outs RelAlg_Relation:$result);
}

def UnionOp :  RelAlg_Op<"union",[RecursiveSideEffects,DeclareOpInterfaceMethods<Operator>]> {
let summary = "union operation";
let arguments = (ins  SymbolNameAttr:$sym_name,  RelAlg_SetSemantic:$set_semantic,RelAlg_Relation:$left,RelAlg_Relation:$right,ArrayAttr:$mapping);
let results = (outs RelAlg_Relation:$result);
}
def IntersectOp :  RelAlg_Op<"intersect",[RecursiveSideEffects,DeclareOpInterfaceMethods<Operator>]> {
let summary = "intersect operation";
let arguments = (ins  SymbolNameAttr:$sym_name,  RelAlg_SetSemantic:$set_semantic,RelAlg_Relation:$left,RelAlg_Relation:$right,ArrayAttr:$mapping);
let results = (outs RelAlg_Relation:$result);
}
def ExceptOp :  RelAlg_Op<"except",[RecursiveSideEffects,DeclareOpInterfaceMethods<Operator>]> {
let summary = "except operation";
let arguments = (ins  SymbolNameAttr:$sym_name,  RelAlg_SetSemantic:$set_semantic,RelAlg_Relation:$left,RelAlg_Relation:$right,ArrayAttr:$mapping);
let results = (outs RelAlg_Relation:$result);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Terminators
////////////////////////////////////////////////////////////////////////////////////////////////////

def RelAlg_ReturnOp : RelAlg_Op<"return", [NoSideEffect, ReturnLike, Terminator,
   ParentOneOf<["SelectionOp","AggregationOp","ForEachOp","MapOp","JoinOp","OuterJoinOp"]>
        ]> {
let summary = "loop yield and termination operation";
let arguments = (ins Variadic<DBType>:$results);
let builders = [OpBuilderDAG<(ins), [{ /* nothing to do */ }]>];
let assemblyFormat =
[{  attr-dict ($results^ `:` type($results))? }];
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility
////////////////////////////////////////////////////////////////////////////////////////////////////

def GetAttrOp : RelAlg_Op<"getattr"> {
let summary = "get attribute operation";
let arguments = (ins RelationalAttributeRefAttr:$attr, RelAlg_Tuple:$tuple);
let results = (outs DBType : $res);
}
def AddAttrOp : RelAlg_Op<"addattr"> {
let summary = "add attribute operation";
let arguments = (ins RelationalAttributeDefAttr:$attr, DBType:$val);
}

def AggrFuncOp : RelAlg_Op<"aggrfn"> {
let summary = "aggregation operation";
let arguments = (ins RelAlg_AggrFunc:$fn,RelAlg_Relation:$rel,RelationalAttributeRefAttr:$attr);
let results = (outs DBType:$result);
}
def CountRowsOp : RelAlg_Op<"count"> {
let summary = "count rows";
let arguments = (ins RelAlg_Relation:$rel);
let results = (outs DB_IntType:$res);
let assemblyFormat =
[{  $rel attr-dict }];
}


def ForEachOp : RelAlg_Op<"foreach",
[RecursiveSideEffects]> {
let arguments = (ins RelAlg_Relation: $rel,ArrayAttr:$attrs);
let regions = (region SizedRegion<1>:$region);
}

def ExistsOp : RelAlg_Op<"exists"> {
let summary = "exists operation";
let arguments = (ins RelAlg_Relation:$rel);
let results = (outs DB_BoolType:$result);
let assemblyFormat =
[{  $rel attr-dict }];
}
def MaterializeOp : RelAlg_Op<"materialize"> {
let summary = "materialize operation";
let arguments = (ins RelAlg_Relation:$rel,ArrayAttr:$attrs);
let results = (outs DB_CollectionType:$result);
}
def GetScalarOp : RelAlg_Op<"getscalar"> {
let summary = "get attribute operation";
let arguments = (ins RelationalAttributeRefAttr:$attr, RelAlg_Relation:$rel);
let results = (outs DBType : $res);
}
def InOp : RelAlg_Op<"in"> {
let summary = "in operation";
let arguments = (ins DBType:$val, RelAlg_Relation:$rel);
let results = (outs DB_BoolType : $res);
}
#endif// OPS
