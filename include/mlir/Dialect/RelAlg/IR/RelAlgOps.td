#ifndef LINGODB_DIALECT_RELALG_IR_OPS
#define LINGODB_DIALECT_RELALG_IR_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"

include "RelAlgInterfaces.td"

def DSA_Table : Type<CPred<"$_self.isa<::mlir::dsa::TableType>()">,"DB table type">;
def DSA_Collection : Type<CPred<"$_self.isa<::mlir::dsa::CollectionType>()">,"DB collection type">;

//===----------------------------------------------------------------------===//
// RelAlg dialect definition.
//===----------------------------------------------------------------------===//

def RelAlg_Dialect : Dialect {
    let name = "relalg";
    let summary = "A dialect for relational algebra";
    let cppNamespace = "::mlir::relalg";

    let extraClassDeclaration = [{
        private:
            ColumnManager columnManager;
        public:
            ColumnManager& getColumnManager(){
                return columnManager;
            }
    }];
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

class RelAlg_Type<string name, string typeMnemonic> : TypeDef<RelAlg_Dialect, name> {
    let mnemonic = typeMnemonic;
}

def RelAlg_TupleStream : RelAlg_Type<"TupleStream", "tuplestream"> {
    let summary = "tuple stream type";
}

def RelAlg_Tuple : RelAlg_Type<"Tuple", "tuple"> {
    let summary = "tuple type";
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

class RelAlg_Attr<string name> : AttrDef<RelAlg_Dialect, name>;

def TableMetaDataAttr : RelAlg_Attr<"TableMetaData"> {
    let mnemonic = "table_metadata";
    let parameters = (ins "std::shared_ptr<runtime::TableMetaData>":$meta);
    let hasCustomAssemblyFormat = 1;
}

def ColumnDefAttr : RelAlg_Attr<"ColumnDef"> {
    let mnemonic = "columndef";
    let parameters = (ins "SymbolRefAttr":$name,"std::shared_ptr<Column>":$columnPtr,"Attribute":$fromExisting);
    let extraClassDeclaration = [{
        Column &getColumn() const { return *getColumnPtr();  }
    }];
    let hasCustomAssemblyFormat=1;
}
def ColumnRefAttr : RelAlg_Attr<"ColumnRef"> {
    let mnemonic = "columnref";
    let parameters = (ins "SymbolRefAttr":$name,"std::shared_ptr<Column>":$columnPtr);
    let extraClassDeclaration = [{
        Column &getColumn() const { return *getColumnPtr();  }
    }];
        let hasCustomAssemblyFormat=1;
}

def SortSpecificationAttr : RelAlg_Attr<"SortSpecification"> {
    let mnemonic = "sortspec";
    let parameters = (ins "ColumnRefAttr":$attr,"SortSpec":$sortSpec);
        let hasCustomAssemblyFormat=1;

}

//===----------------------------------------------------------------------===//
// Enums
//===----------------------------------------------------------------------===//

def RelAlg_SortSpec_DESC  : I64EnumAttrCase<"desc", 0>;
def RelAlg_AggrFunc_ASC  : I64EnumAttrCase<"asc", 1>;

def RelAlg_SortSpec : I64EnumAttr<"SortSpec", "sort order",
        [RelAlg_SortSpec_DESC,RelAlg_AggrFunc_ASC]> {
    let cppNamespace = "::mlir::relalg";
}

def RelAlg_AggrFunc_Sum   : I64EnumAttrCase<"sum", 0>;
def RelAlg_AggrFunc_Min   : I64EnumAttrCase<"min", 1>;
def RelAlg_AggrFunc_Max   : I64EnumAttrCase<"max", 2>;
def RelAlg_AggrFunc_Avg   : I64EnumAttrCase<"avg", 3>;
def RelAlg_AggrFunc_Count : I64EnumAttrCase<"count", 4>;
def RelAlg_AggrFunc_Any : I64EnumAttrCase<"any", 5>;

def RelAlg_AggrFunc : I64EnumAttr<"AggrFunc", "aggregation function",
        [RelAlg_AggrFunc_Sum, RelAlg_AggrFunc_Min, RelAlg_AggrFunc_Max,
         RelAlg_AggrFunc_Avg, RelAlg_AggrFunc_Count, RelAlg_AggrFunc_Any]> {
    let cppNamespace = "::mlir::relalg";
}

def RelAlg_SetSemantic_Distinct : I64EnumAttrCase<"distinct", 0>;
def RelAlg_SetSemantic_All      : I64EnumAttrCase<"all", 1>;

def RelAlg_SetSemantic : I64EnumAttr<"SetSemantic", "",
        [RelAlg_SetSemantic_Distinct,RelAlg_SetSemantic_All]> {
    let cppNamespace = "::mlir::relalg";
}

//===----------------------------------------------------------------------===//
// Base RelAlg operation definition.
//===----------------------------------------------------------------------===//

class RelAlg_Op<string mnemonic, list<Trait> traits = []> : Op<RelAlg_Dialect, mnemonic, traits> {
  let hasCustomAssemblyFormat = 1;

}

//===----------------------------------------------------------------------===//
// Operations that "create" relations
//===----------------------------------------------------------------------===//

def ConstRelationOp : RelAlg_Op<"const_relation", [RecursiveSideEffects,Operator]> {
    let summary = "constant relation";

    let arguments = (ins ArrayAttr:$columns,ArrayAttr:$values);
    let results = (outs RelAlg_TupleStream:$result);
    let assemblyFormat = [{ `columns` `:` custom<CustDefArr>($columns) `values` `:` $values attr-dict}];

    let extraClassDeclaration = [{
        ColumnSet getCreatedColumns();
    }];
}

def BaseTableOp : RelAlg_Op<"basetable",
        [SingleBlockImplicitTerminator<"relalg::ReturnOp">,RecursiveSideEffects,
         Operator]> {
    let summary = "base table operation";

    let arguments = (ins StrAttr: $table_identifier, TableMetaDataAttr: $meta, DictionaryAttr: $columns);
    let results = (outs RelAlg_TupleStream:$result);

    let extraClassDeclaration = [{
        ColumnSet getCreatedColumns();
        mlir::relalg::FunctionalDependencies getFDs();
    }];
}

//===----------------------------------------------------------------------===//
// Operators
//===----------------------------------------------------------------------===//

def SelectionOp : RelAlg_Op<"selection",
        [NoSideEffect,Operator,PredicateOperator,TupleLamdaOperator,
         UnaryOperator]> {
    let summary = "selection operation";
    let description = [{
        Filter tuple stream, the region returns `1` iff the value should be
        contained in the output stream.
    }];

    let arguments = (ins RelAlg_TupleStream:$rel);
    let results = (outs RelAlg_TupleStream:$result);
    let regions = (region SizedRegion<1>:$predicate);
    let assemblyFormat = [{ $rel custom<CustRegion>($predicate) attr-dict-with-keyword }];
}

def MapOp : RelAlg_Op<"map", [NoSideEffect,Operator,TupleLamdaOperator,UnaryOperator]> {
    let summary = "map operation";

    let arguments = (ins RelAlg_TupleStream:$rel,ArrayAttr:$computed_cols);
    let results = (outs RelAlg_TupleStream:$result);
    let regions = (region SizedRegion<1>:$predicate);
    let assemblyFormat = [{  $rel `computes` `:` custom<CustDefArr>($computed_cols) custom<CustRegion>($predicate) attr-dict-with-keyword }];

    let extraClassDeclaration = [{
        ColumnSet getCreatedColumns();
    }];
}

def LimitOp : RelAlg_Op<"limit", [NoSideEffect,Operator,UnaryOperator]> {
    let summary = "limit operation";

    let arguments = (ins I32Attr:$rows, RelAlg_TupleStream:$rel);
    let results = (outs RelAlg_TupleStream:$result);
    let assemblyFormat = [{ $rows $rel attr-dict }];
}

def TopKOp : RelAlg_Op<"topk", [NoSideEffect,Operator,UnaryOperator]> {
    let summary = "top k operation";
    let description = [{
        Return top `rows` elements according to the sorting region. This is
        conceptually similar to a combination of `sort` and `limit`, but allows
        for further optimizations.
    }];

    let arguments = (ins I32Attr:$rows, RelAlg_TupleStream:$rel,ArrayAttr:$sortspecs);
    let results = (outs RelAlg_TupleStream:$result);
    let assemblyFormat = [{ $rows $rel custom<SortSpecs>($sortspecs) attr-dict}];
}

def CrossProductOp : RelAlg_Op<"crossproduct",
        [RecursiveSideEffects,Operator,BinaryOperator]> {
    let summary = "crossproduct operation";

    let arguments = (ins RelAlg_TupleStream:$left,RelAlg_TupleStream:$right);
    let results = (outs RelAlg_TupleStream:$result);
    let assemblyFormat = [{ $left `,` $right attr-dict }];
}

def RenamingOp : RelAlg_Op<"renaming",
        [RecursiveSideEffects,Operator,UnaryOperator]> {
    let summary = "renaming columns in relation";

    let arguments = (ins RelAlg_TupleStream:$rel,ArrayAttr:$columns);
    let results = (outs RelAlg_TupleStream:$result);
    let assemblyFormat = [{ $rel `renamed` `:` custom<CustDefArr>($columns) attr-dict-with-keyword }];

    let extraClassDeclaration = [{
        ColumnSet getCreatedColumns();
        ColumnSet getUsedColumns();
        ColumnSet getAvailableColumns();
    }];
}

def AggregationOp : RelAlg_Op<"aggregation", [NoSideEffect,Operator,UnaryOperator]> {
    let summary = "aggregation operation";

    let arguments = (ins RelAlg_TupleStream:$rel, ArrayAttr:$group_by_cols, ArrayAttr:$computed_cols);
    let results = (outs RelAlg_TupleStream:$result);
    let regions = (region SizedRegion<1>:$aggr_func);
    let assemblyFormat = [{ $rel custom<CustRefArr>($group_by_cols) `computes` `:` custom<CustDefArr>($computed_cols) custom<CustRegion>($aggr_func)  attr-dict-with-keyword }];

    let extraClassDeclaration = [{
        ColumnSet getUsedColumns();
        ColumnSet getAvailableColumns();
        ColumnSet getCreatedColumns();
    }];
}

def SortOp : RelAlg_Op<"sort", [NoSideEffect,Operator,UnaryOperator]> {
    let summary = "sort operation";

    let arguments = (ins RelAlg_TupleStream:$rel, ArrayAttr:$sortspecs);
    let results = (outs RelAlg_TupleStream:$result);
    let assemblyFormat = [{ $rel custom<SortSpecs>($sortspecs) attr-dict}];
    let extraClassDeclaration = [{
        ColumnSet getUsedColumns();
    }];
}

// Joins
class RelAlg_JoinOp<string mnemonic> : RelAlg_Op<mnemonic,
        [NoSideEffect,Operator,BinaryOperator,PredicateOperator,
         TupleLamdaOperator]> {
    let arguments = (ins RelAlg_TupleStream:$left,RelAlg_TupleStream:$right);
    let results = (outs RelAlg_TupleStream:$result);
    let regions = (region SizedRegion<1>:$predicate);
    let assemblyFormat = [{ $left `,` $right custom<CustRegion>($predicate) attr-dict-with-keyword }];
}

def InnerJoinOp : RelAlg_JoinOp<"join"> {
    let summary = "inner join operation";
}

def SemiJoinOp : RelAlg_JoinOp<"semijoin"> {
    let summary = "semi join operation";

    let extraClassDeclaration = [{
        ColumnSet getAvailableColumns();
    }];
}

def AntiSemiJoinOp : RelAlg_JoinOp<"antisemijoin"> {
    let summary = "antisemi join operation";

    let extraClassDeclaration = [{
        ColumnSet getAvailableColumns();
    }];
}

def OuterJoinOp : RelAlg_JoinOp<"outerjoin"> {
    let summary = "outer join operation";

    let arguments = (ins RelAlg_TupleStream:$left,RelAlg_TupleStream:$right,ArrayAttr:$mapping);
    let assemblyFormat = [{ $left `,` $right custom<CustRegion>($predicate)  custom<CustAttrMapping>($mapping) attr-dict-with-keyword }];

    let extraClassDeclaration = [{
        ColumnSet getCreatedColumns();
        ColumnSet getUsedColumns();
        ColumnSet getAvailableColumns();
    }];

}

def FullOuterJoinOp : RelAlg_JoinOp<"fullouterjoin"> {
    let summary = "full outer join operation";
}

def SingleJoinOp : RelAlg_JoinOp<"singlejoin"> {
    let summary = "single join operation";

    let arguments = (ins RelAlg_TupleStream:$left,RelAlg_TupleStream:$right,ArrayAttr:$mapping);
    let assemblyFormat = [{  $left `,` $right custom<CustRegion>($predicate)  custom<CustAttrMapping>($mapping) attr-dict-with-keyword }];

    let extraClassDeclaration = [{
        ColumnSet getCreatedColumns();
        ColumnSet getUsedColumns();
        ColumnSet getAvailableColumns();
    }];
}

def CollectionJoinOp : RelAlg_JoinOp<"collectionjoin"> {
    let summary = "collection join operation";

    let arguments = (ins ArrayAttr:$cols,ColumnDefAttr:$collAttr,RelAlg_TupleStream:$left,RelAlg_TupleStream:$right);
    let assemblyFormat = [{ custom<CustRefArr>($cols) custom<CustDef>($collAttr) $left `,` $right custom<CustRegion>($predicate)  attr-dict }];

    let extraClassDeclaration = [{
        ColumnSet getCreatedColumns();
        ColumnSet getUsedColumns();
        ColumnSet getAvailableColumns();
    }];
}

def MarkJoinOp : RelAlg_JoinOp<"markjoin"> {
    let summary = "mark join operation";

    let arguments = (ins ColumnDefAttr:$markattr,RelAlg_TupleStream:$left,RelAlg_TupleStream:$right);
    let assemblyFormat = [{ custom<CustDef>($markattr) $left `,` $right custom<CustRegion>($predicate)  attr-dict }];

    let extraClassDeclaration = [{
        ColumnSet getAvailableColumns();
        ColumnSet getCreatedColumns();
    }];
}

// Set Operations
def ProjectionOp : RelAlg_Op<"projection",[NoSideEffect,Operator,UnaryOperator]> {
    let summary = "projection operation";

    let arguments = (ins RelAlg_SetSemantic:$set_semantic,RelAlg_TupleStream:$rel, ArrayAttr:$cols);
    let results = (outs RelAlg_TupleStream:$result);
    let assemblyFormat = [{ $set_semantic custom<CustRefArr>($cols) $rel attr-dict }];

    let extraClassDeclaration = [{
        ColumnSet getAvailableColumns();
        ColumnSet getUsedColumns();
    }];
}

class RelAlg_BinarySetOp<string mnemonic> : RelAlg_Op<mnemonic,
        [NoSideEffect,Operator,BinaryOperator]> {
    let arguments = (ins RelAlg_SetSemantic:$set_semantic,RelAlg_TupleStream:$left,RelAlg_TupleStream:$right,ArrayAttr:$mapping);
    let results = (outs RelAlg_TupleStream:$result);
    let assemblyFormat = [{ $set_semantic $left `,` $right custom<CustAttrMapping>($mapping) attr-dict }];
    let extraClassDeclaration = [{
        ColumnSet getCreatedColumns();
        ColumnSet getUsedColumns();
        ColumnSet getAvailableColumns();
    }];
    let extraClassDefinition =[{
        ColumnSet $cppClass::getCreatedColumns(){ return mlir::relalg::detail::getSetOpCreatedColumns(this->getOperation()); }
        ColumnSet $cppClass::getUsedColumns(){ return mlir::relalg::detail::getSetOpUsedColumns(this->getOperation()); }
        ColumnSet $cppClass::getAvailableColumns(){ return mlir::relalg::detail::getSetOpCreatedColumns(this->getOperation()); }
    }];
}

def UnionOp : RelAlg_BinarySetOp<"union"> {
    let summary = "union operation";
}

def IntersectOp : RelAlg_BinarySetOp<"intersect"> {
    let summary = "intersect operation";
}

def ExceptOp : RelAlg_BinarySetOp<"except"> {
    let summary = "except operation";
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Terminators
////////////////////////////////////////////////////////////////////////////////////////////////////

def RelAlg_ReturnOp : RelAlg_Op<"return", [NoSideEffect,ReturnLike,Terminator]> {
    let summary = "loop yield and termination operation";

    let arguments = (ins Variadic<AnyType>:$results);
    let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
    let assemblyFormat = "attr-dict ($results^ `:` type($results))?";
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility
////////////////////////////////////////////////////////////////////////////////////////////////////

def GetColumnOp : RelAlg_Op<"getcol", [NoSideEffect]> {
    let summary = "get column operation";

    let arguments = (ins ColumnRefAttr:$attr,RelAlg_Tuple:$tuple);
    let results = (outs AnyType : $res);
    let assemblyFormat = [{  $tuple custom<CustRef>($attr) `:` type($res) attr-dict }];
}

def AggrFuncOp : RelAlg_Op<"aggrfn", [NoSideEffect]> {
    let summary = "aggregation operation";

    let arguments = (ins RelAlg_AggrFunc:$fn,RelAlg_TupleStream:$rel,ColumnRefAttr:$attr);
    let results = (outs AnyType:$result);
    let assemblyFormat = [{  $fn custom<CustRef>($attr) $rel `:` type($result) attr-dict }];
}

def CountRowsOp : RelAlg_Op<"count", [NoSideEffect]> {
    let summary = "count rows";

    let arguments = (ins RelAlg_TupleStream:$rel);
    let results = (outs I64:$res);
    let assemblyFormat = [{  $rel attr-dict }];
}

def ExistsOp : RelAlg_Op<"exists", [NoSideEffect]> {
    let arguments = (ins RelAlg_TupleStream:$rel);
    let results = (outs I1:$result);
    let assemblyFormat = [{  $rel attr-dict }];
}

def MaterializeOp : RelAlg_Op<"materialize", [NoSideEffect]> {
    let arguments = (ins RelAlg_TupleStream:$rel,ArrayAttr:$cols,ArrayAttr:$columns);
    let results = (outs DSA_Table:$result);
    let assemblyFormat = [{  $rel custom<CustRefArr>($cols) `=` `>` $columns `:` type($result) attr-dict }];
}

def TmpOp : RelAlg_Op<"tmp", [NoSideEffect,Operator]> {
    let summary = "temporary materialization";

    let arguments = (ins RelAlg_TupleStream:$rel,ArrayAttr:$cols);
    let results = (outs RelAlg_TupleStream:$res_rel);
    let assemblyFormat = [{  $rel custom<CustRefArr>($cols) attr-dict }];
}

def GetScalarOp : RelAlg_Op<"getscalar", [NoSideEffect]> {
    let summary = "get column";

    let arguments = (ins ColumnRefAttr:$attr, RelAlg_TupleStream:$rel);
    let results = (outs AnyType : $res);
    let assemblyFormat = [{  custom<CustRef>($attr) $rel `:` type($res) attr-dict }];
}

def GetListOp : RelAlg_Op<"getlist", [NoSideEffect]> {
    let summary = "get multiple columns";

    let arguments = (ins RelAlg_TupleStream:$rel,ArrayAttr:$cols);
    let results = (outs DSA_Collection:$res);
    let assemblyFormat = [{ $rel custom<CustRefArr>($cols) `:` type($res) attr-dict }];
}

def InOp : RelAlg_Op<"in", [NoSideEffect]> {
    let summary = "in operation";
    let arguments = (ins AnyType:$val, RelAlg_TupleStream:$rel);
    let results = (outs I1 : $res);
    let assemblyFormat = [{ $val `:` type($val) `,` $rel attr-dict }];
}

#endif // LINGODB_DIALECT_RELALG_IR_OPS
