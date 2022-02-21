#ifndef OPS
#define OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
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

class RelAlg_Op<string mnemonic, list<Trait> traits = []> :
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
def DB_TableType : Type<CPred<"$_self.isa<::mlir::db::TableType>()">,"DB table type">;
def DB_CollectionType : Type<CPred<"$_self.isa<::mlir::db::CollectionType>()">,"DB collection type">;
def DB_IntType : Type<CPred<"$_self.isa<::mlir::db::IntType>()">,"DB int type">{
let builderCall = "::mlir::db::IntType::get($_builder.getContext(),false,64)";
}

def RelAlg_TupleStream :  TypeDef<RelAlg_Dialect, "TupleStream"> {
let summary = "tuple stream type";
let mnemonic="tuplestream";
}
def RelAlg_Tuple :  TypeDef<RelAlg_Dialect, "Tuple"> {
let summary = "tuple type";
let mnemonic="tuple";
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// Attributes
////////////////////////////////////////////////////////////////////////////////////////////////////
class RelAlg_Attr<string name> : AttrDef<RelAlg_Dialect, name>;
def TableMetaDataAttr : RelAlg_Attr<"TableMetaData"> {
let mnemonic = "table_metadata";
let parameters = (ins "std::shared_ptr<runtime::TableMetaData>":$meta);
}

def RelationalAttributeDefAttr : RelAlg_Attr<"RelationalAttributeDef"> {
let mnemonic = "attrdef";
let parameters = (ins "std::string":$name,"std::shared_ptr<RelationalAttribute>":$relationalAttributePtr,"Attribute":$fromExisting);
let extraClassDeclaration = [{
  RelationalAttribute &getRelationalAttribute() { return *getRelationalAttributePtr();  }
}];
}
def RelationalAttributeRefAttr : RelAlg_Attr<"RelationalAttributeRef"> {
let mnemonic = "attrref";
let parameters = (ins "SymbolRefAttr":$name,"std::shared_ptr<RelationalAttribute>":$relationalAttributePtr);
let extraClassDeclaration = [{
  RelationalAttribute &getRelationalAttribute() { return *getRelationalAttributePtr(); }
}];
}

def SortSpecificationAttr : RelAlg_Attr<"SortSpecification"> {
let mnemonic = "sortspec";
let parameters = (ins "RelationalAttributeRefAttr":$attr,"SortSpec":$sortSpec);
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
        [ RecursiveSideEffects,Operator]> {
let summary = "constant relation";
let arguments = (ins SymbolNameAttr:$sym_name,ArrayAttr:$attributes,ArrayAttr:$values);
let results = (outs RelAlg_TupleStream:$result);
let extraClassDeclaration = [{
Attributes getCreatedAttributes();
}];
}

def BaseTableOp : RelAlg_Op<"basetable",
[SingleBlockImplicitTerminator<"relalg::ReturnOp">,RecursiveSideEffects,Operator]> {
let summary = "base table operation";
let arguments = (ins SymbolNameAttr:$sym_name, StrAttr: $table_identifier, TableMetaDataAttr: $meta, DictionaryAttr: $columns);
let results = (outs RelAlg_TupleStream:$result);
let extraClassDeclaration = [{
Attributes getCreatedAttributes();
}];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Operators
////////////////////////////////////////////////////////////////////////////////////////////////////

def SelectionOp : RelAlg_Op<"selection",[NoSideEffect,Operator,PredicateOperator,TupleLamdaOperator,UnaryOperator]> {
let summary = "selection operation";
let arguments = (ins RelAlg_TupleStream:$rel);
let results = (outs RelAlg_TupleStream:$result);
let regions = (region SizedRegion<1>:$predicate);
}
def MapOp : RelAlg_Op<"map",[NoSideEffect,Operator,TupleLamdaOperator,UnaryOperator]> {
let summary = "map operation";
let arguments = (ins SymbolNameAttr:$sym_name, RelAlg_TupleStream:$rel);
let results = (outs RelAlg_TupleStream:$result);
let regions = (region SizedRegion<1>:$predicate);
}
def LimitOp : RelAlg_Op<"limit",[NoSideEffect,Operator,UnaryOperator]> {
let summary = "limit operation";
let arguments = (ins I32Attr:$rows, RelAlg_TupleStream:$rel);
let results = (outs RelAlg_TupleStream:$result);
let assemblyFormat = [{ $rows $rel attr-dict }];
}
def TopKOp : RelAlg_Op<"topk",[NoSideEffect,Operator,UnaryOperator]> {
let summary = "top k operation";
let arguments = (ins I32Attr:$rows, RelAlg_TupleStream:$rel,ArrayAttr:$sortspecs);
let results = (outs RelAlg_TupleStream:$result);
}
def CrossProductOp : RelAlg_Op<"crossproduct",[RecursiveSideEffects,Operator,BinaryOperator]> {
let summary = "crossproduct operation";
let arguments = (ins RelAlg_TupleStream:$left,RelAlg_TupleStream:$right);
let results = (outs RelAlg_TupleStream:$result);
let assemblyFormat = [{ $left `,` $right attr-dict }];
}

def RenamingOp : RelAlg_Op<"renaming",
        [ RecursiveSideEffects,Operator,UnaryOperator]> {
let summary = "renaming attributes in relation";
let arguments = (ins RelAlg_TupleStream:$rel,SymbolNameAttr:$sym_name,ArrayAttr:$attributes);
let results = (outs RelAlg_TupleStream:$result);
let extraClassDeclaration = [{
Attributes getCreatedAttributes();
Attributes getUsedAttributes();
Attributes getAvailableAttributes();

}];
}

def AggregationOp : RelAlg_Op<"aggregation",[NoSideEffect,Operator,UnaryOperator]> {
let summary = "aggregation operation";
let arguments = (ins SymbolNameAttr:$sym_name, RelAlg_TupleStream:$rel, ArrayAttr:$group_by_attrs);
let results = (outs RelAlg_TupleStream:$result);
let regions = (region SizedRegion<1>:$aggr_func);
let extraClassDeclaration = [{
Attributes getUsedAttributes();
Attributes getAvailableAttributes();
}];
}
def SortOp : RelAlg_Op<"sort",[NoSideEffect,Operator,UnaryOperator]> {
let summary = "sort operation";
let arguments = (ins RelAlg_TupleStream:$rel, ArrayAttr:$sortspecs);
let results = (outs RelAlg_TupleStream:$result);
let extraClassDeclaration = [{
Attributes getUsedAttributes();
}];
}
// Joins
def InnerJoinOp : RelAlg_Op<"join",[NoSideEffect,Operator,BinaryOperator,PredicateOperator,TupleLamdaOperator]> {
let summary = "inner join operation";
let arguments = (ins RelAlg_TupleStream:$left,RelAlg_TupleStream:$right);
let results = (outs RelAlg_TupleStream:$result);
let regions = (region SizedRegion<1>:$predicate);
}

def SemiJoinOp : RelAlg_Op<"semijoin",[NoSideEffect,Operator,BinaryOperator,PredicateOperator,TupleLamdaOperator]> {
let summary = "semi join operation";
let arguments = (ins RelAlg_TupleStream:$left,RelAlg_TupleStream:$right);
let results = (outs RelAlg_TupleStream:$result);
let regions = (region SizedRegion<1>:$predicate);

let parser = [{
return parseNonCommutativeJoin(parser, result);
}];

let printer = [{
return printNonCommutativeJoin(this->getOperation(), p);
}];
let extraClassDeclaration = [{
Attributes getAvailableAttributes();
}];
}
def AntiSemiJoinOp : RelAlg_Op<"antisemijoin",[NoSideEffect,Operator,BinaryOperator,PredicateOperator,TupleLamdaOperator]> {
let summary = "antisemi join operation";
let arguments = (ins RelAlg_TupleStream:$left,RelAlg_TupleStream:$right);
let results = (outs RelAlg_TupleStream:$result);
let regions = (region SizedRegion<1>:$predicate);
let extraClassDeclaration = [{
Attributes getAvailableAttributes();
}];
let parser = [{
return parseNonCommutativeJoin(parser, result);
}];

let printer = [{
return printNonCommutativeJoin(this->getOperation(), p);
}];
}
def OuterJoinOp : RelAlg_Op<"outerjoin",[NoSideEffect,Operator,BinaryOperator,PredicateOperator,TupleLamdaOperator]> {
let summary = "outer join operation";
let arguments = (ins SymbolNameAttr:$sym_name,RelAlg_TupleStream:$left,RelAlg_TupleStream:$right,ArrayAttr:$mapping);
let results = (outs RelAlg_TupleStream:$result);
let regions = (region SizedRegion<1>:$predicate);
let extraClassDeclaration = [{
Attributes getCreatedAttributes();
Attributes getUsedAttributes();
Attributes getAvailableAttributes();}];
}
def FullOuterJoinOp : RelAlg_Op<"fullouterjoin",[NoSideEffect,Operator,BinaryOperator,PredicateOperator,TupleLamdaOperator]> {
let summary = "full outer join operation";
let arguments = (ins RelAlg_TupleStream:$left,RelAlg_TupleStream:$right);
let results = (outs RelAlg_TupleStream:$result);
let regions = (region SizedRegion<1>:$predicate);

}
def SingleJoinOp : RelAlg_Op<"singlejoin",[NoSideEffect,Operator,BinaryOperator,PredicateOperator,TupleLamdaOperator]> {
let summary = "single join operation";
let arguments = (ins SymbolNameAttr:$sym_name,RelAlg_TupleStream:$left,RelAlg_TupleStream:$right,ArrayAttr:$mapping);
let results = (outs RelAlg_TupleStream:$result);
let regions = (region SizedRegion<1>:$predicate);
let extraClassDeclaration = [{
Attributes getCreatedAttributes();
Attributes getUsedAttributes();
Attributes getAvailableAttributes();}];
}
def CollectionJoinOp : RelAlg_Op<"collectionjoin",[NoSideEffect,Operator,BinaryOperator,PredicateOperator,TupleLamdaOperator]> {
    let summary = "collection join operation";
    let arguments = (ins SymbolNameAttr:$sym_name,ArrayAttr:$attrs,RelationalAttributeDefAttr:$collAttr,RelAlg_TupleStream:$left,RelAlg_TupleStream:$right);
    let results = (outs RelAlg_TupleStream:$result);
    let regions = (region SizedRegion<1>:$predicate);
    let extraClassDeclaration = [{
        Attributes getCreatedAttributes();
        Attributes getUsedAttributes();
        Attributes getAvailableAttributes();}];
}
def MarkJoinOp : RelAlg_Op<"markjoin",[NoSideEffect,Operator,BinaryOperator,PredicateOperator,TupleLamdaOperator]> {
let summary = "mark join operation";
let arguments = (ins SymbolNameAttr:$sym_name,RelationalAttributeDefAttr:$markattr,RelAlg_TupleStream:$left,RelAlg_TupleStream:$right);
let results = (outs RelAlg_TupleStream:$result);
let regions = (region SizedRegion<1>:$predicate);
let extraClassDeclaration = [{
Attributes getAvailableAttributes();
Attributes getCreatedAttributes();
}];
}
// Set Operations
def ProjectionOp : RelAlg_Op<"projection",[NoSideEffect,Operator,UnaryOperator]> {
let summary = "projection operation";
let arguments = (ins  RelAlg_SetSemantic:$set_semantic,RelAlg_TupleStream:$rel, ArrayAttr:$attrs);
let results = (outs RelAlg_TupleStream:$result);
let extraClassDeclaration = [{
Attributes getAvailableAttributes();
}];
}

def UnionOp :  RelAlg_Op<"union",[NoSideEffect,Operator,BinaryOperator,UnaryOperator]> {
let summary = "union operation";
let arguments = (ins  SymbolNameAttr:$sym_name,  RelAlg_SetSemantic:$set_semantic,RelAlg_TupleStream:$left,RelAlg_TupleStream:$right,ArrayAttr:$mapping);
let results = (outs RelAlg_TupleStream:$result);
}
def IntersectOp :  RelAlg_Op<"intersect",[NoSideEffect,Operator,BinaryOperator]> {
let summary = "intersect operation";
let arguments = (ins  SymbolNameAttr:$sym_name,  RelAlg_SetSemantic:$set_semantic,RelAlg_TupleStream:$left,RelAlg_TupleStream:$right,ArrayAttr:$mapping);
let results = (outs RelAlg_TupleStream:$result);
}
def ExceptOp :  RelAlg_Op<"except",[NoSideEffect,Operator,BinaryOperator]> {
let summary = "except operation";
let arguments = (ins  SymbolNameAttr:$sym_name,  RelAlg_SetSemantic:$set_semantic,RelAlg_TupleStream:$left,RelAlg_TupleStream:$right,ArrayAttr:$mapping);
let results = (outs RelAlg_TupleStream:$result);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Terminators
////////////////////////////////////////////////////////////////////////////////////////////////////

def RelAlg_ReturnOp : RelAlg_Op<"return", [NoSideEffect, ReturnLike, Terminator]> {
let summary = "loop yield and termination operation";
let arguments = (ins Variadic<AnyType>:$results);
let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
let assemblyFormat =
[{  attr-dict ($results^ `:` type($results))? }];
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility
////////////////////////////////////////////////////////////////////////////////////////////////////

def GetAttrOp : RelAlg_Op<"getattr",[NoSideEffect]> {
let summary = "get attribute operation";
let arguments = (ins RelationalAttributeRefAttr:$attr, RelAlg_Tuple:$tuple);
let results = (outs AnyType : $res);
}
def AddAttrOp : RelAlg_Op<"addattr"> {
let summary = "add attribute operation";
let arguments = (ins RelAlg_Tuple:$tuple,RelationalAttributeDefAttr:$attr, DBType:$val);
let results = (outs RelAlg_Tuple:$tuple_out);
}

def AggrFuncOp : RelAlg_Op<"aggrfn"> {
let summary = "aggregation operation";
let arguments = (ins RelAlg_AggrFunc:$fn,RelAlg_TupleStream:$rel,RelationalAttributeRefAttr:$attr);
let results = (outs DBType:$result);
}
def CountRowsOp : RelAlg_Op<"count"> {
let summary = "count rows";
let arguments = (ins RelAlg_TupleStream:$rel);
let results = (outs DB_IntType:$res);
let assemblyFormat = [{  $rel attr-dict }];
}


def ExistsOp : RelAlg_Op<"exists"> {
let arguments = (ins RelAlg_TupleStream:$rel);
let results = (outs DB_BoolType:$result);
let assemblyFormat = [{  $rel attr-dict }];
}
def MaterializeOp : RelAlg_Op<"materialize",[NoSideEffect]> {
let summary = "materialize operation";
let arguments = (ins RelAlg_TupleStream:$rel,ArrayAttr:$attrs,ArrayAttr:$columns);
let results = (outs DB_TableType:$result);
}
def TmpOp : RelAlg_Op<"tmp",[NoSideEffect,Operator]> {
let summary = "create tmp";
let arguments = (ins RelAlg_TupleStream:$rel,ArrayAttr:$attrs);
let results = (outs RelAlg_TupleStream:$res_rel);
}
def GetScalarOp : RelAlg_Op<"getscalar"> {
let summary = "get attribute operation";
let arguments = (ins RelationalAttributeRefAttr:$attr, RelAlg_TupleStream:$rel);
let results = (outs DBType : $res);
}
def GetListOp : RelAlg_Op<"getlist"> {
    let summary = "get attribute operation";
    let arguments = (ins RelAlg_TupleStream:$rel,ArrayAttr:$attrs);
    let results = (outs  DB_CollectionType : $res);
}
def InOp : RelAlg_Op<"in"> {
let summary = "in operation";
let arguments = (ins DBType:$val, RelAlg_TupleStream:$rel);
let results = (outs DB_BoolType : $res);
let assemblyFormat = [{ $val `:` type($val) `,` $rel attr-dict }];
}
#endif// OPS
