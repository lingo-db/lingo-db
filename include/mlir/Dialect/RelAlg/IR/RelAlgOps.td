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
}

//===----------------------------------------------------------------------===//
// Base relAlg operation definition.
//===----------------------------------------------------------------------===//

class RelAlg_Op<string mnemonic, list<OpTrait> traits = []> :
Op<RelAlg_Dialect, mnemonic, traits>{
let printer = [{ return ::print(p, *this); }];
let parser = [{ return ::parse$cppClass(parser, result); }];
}

def DBType : Type<CPred<"$_self.isa<::mlir::db::DBType>()">,"DB dialect type">;

def RelAlg_Relation :  TypeDef<RelAlg_Dialect, "Relation"> {
let summary = "relation type";
}
def RelAlg_Tuple :  TypeDef<RelAlg_Dialect, "Tuple"> {
let summary = "tuple type";
}
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

def RelAlg_ReturnOp : RelAlg_Op<"return", [NoSideEffect, ReturnLike, Terminator,
   // ParentOneOf<[", ForOp", "ParallelOp","WhileOp"]>
        ]> {
let summary = "loop yield and termination operation";

let arguments = (ins Variadic<DBType>:$results);
let builders = [OpBuilderDAG<(ins), [{ /* nothing to do */ }]>];
let assemblyFormat =
[{  attr-dict ($results^ `:` type($results))? }];
}



def BaseTableOp : RelAlg_Op<"basetable",
[SingleBlockImplicitTerminator<"relalg::ReturnOp">,RecursiveSideEffects]> {
let summary = "base table operation";
let arguments = (ins SymbolNameAttr:$sym_name, StrAttr: $table_identifier, DictionaryAttr: $columns);
let results = (outs RelAlg_Relation:$result);
}

def GetAttrOp : RelAlg_Op<"getattr"> {
let summary = "get attribute operation";
let arguments = (ins RelationalAttributeRefAttr:$attr, RelAlg_Tuple:$tuple);
let results = (outs DBType : $res);
}
def AddAttrOp : RelAlg_Op<"addattr"> {
let summary = "add attribute operation";
let arguments = (ins RelationalAttributeDefAttr:$attr, DBType:$val);
}

def SelectionOp : RelAlg_Op<"selection",[NoSideEffect]> {
let summary = "selection operation";
let arguments = (ins RelAlg_Relation:$rel);
let results = (outs RelAlg_Relation:$result);
let regions = (region SizedRegion<1>:$predicate);
}
def AggregationOp : RelAlg_Op<"aggregation",[NoSideEffect]> {
let summary = "aggregation operation";
let arguments = (ins SymbolNameAttr:$sym_name, RelAlg_Relation:$rel, ArrayAttr:$group_by_attrs);
let results = (outs RelAlg_Relation:$result);
let regions = (region SizedRegion<1>:$aggr_func);
}

def SumAggrFuncOp : RelAlg_Op<"aggr.sum"> {
let summary = "aggregation operation";
let arguments = (ins RelAlg_Relation:$rel,RelationalAttributeRefAttr:$attr);
let results = (outs DBType:$result);
}
#endif// OPS
