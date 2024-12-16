#ifndef LINGODB_DIALECT_RELALG_IR_OPS
#define LINGODB_DIALECT_RELALG_IR_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "lingodb/compiler/Dialect/TupleStream/TupleStreamBase.td"

include "RelAlgInterfaces.td"

def SubOp_ResultTable : Type<CPred<"mlir::isa<::lingodb::compiler::dialect::subop::LocalTableType>($_self)">,"result table type">;
def DSA_Collection : Type<CPred<"mlir::isa<::lingodb::compiler::dialect::util::CollectionType>($_self)">,"DB collection type">;


//===----------------------------------------------------------------------===//
// RelAlg dialect definition.
//===----------------------------------------------------------------------===//

def RelAlg_Dialect : Dialect {
    let name = "relalg";
    let summary = "A dialect for relational algebra";
    let cppNamespace = "::lingodb::compiler::dialect::relalg";
    let useDefaultAttributePrinterParser=1;
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

def SortSpecificationAttr : RelAlg_Attr<"SortSpecification"> {
    let mnemonic = "sortspec";
    let parameters = (ins "lingodb::compiler::dialect::tuples::ColumnRefAttr":$attr,"SortSpec":$sortSpec);
        let hasCustomAssemblyFormat=1;

}

//===----------------------------------------------------------------------===//
// Enums
//===----------------------------------------------------------------------===//

def RelAlg_SortSpec_DESC  : I64EnumAttrCase<"desc", 0>;
def RelAlg_SortSpec_ASC  : I64EnumAttrCase<"asc", 1>;

def RelAlg_SortSpec : I64EnumAttr<"SortSpec", "sort order",
        [RelAlg_SortSpec_DESC,RelAlg_SortSpec_ASC]> {
    let cppNamespace = "::lingodb::compiler::dialect::relalg";
}

def RelAlg_GroupJoinBehavior_Inner  : I64EnumAttrCase<"inner", 0>;
def RelAlg_GroupJoinBehavior_Outer  : I64EnumAttrCase<"outer", 1>;

def RelAlg_GroupJoinBehavior: I64EnumAttr<"GroupJoinBehavior", "groupjoin behavior",
        [RelAlg_GroupJoinBehavior_Inner,RelAlg_GroupJoinBehavior_Outer]> {
    let cppNamespace = "::lingodb::compiler::dialect::relalg";
}

def RelAlg_Func_Sum   : I64EnumAttrCase<"sum", 0>;
def RelAlg_Func_Min   : I64EnumAttrCase<"min", 1>;
def RelAlg_Func_Max   : I64EnumAttrCase<"max", 2>;
def RelAlg_Func_Avg   : I64EnumAttrCase<"avg", 3>;
def RelAlg_Func_Count : I64EnumAttrCase<"count", 4>;
def RelAlg_Func_Any : I64EnumAttrCase<"any", 5>;
def RelAlg_Func_StdDevSamp : I64EnumAttrCase<"stddev_samp", 6>;
def RelAlg_Func_VarSamp : I64EnumAttrCase<"var_samp", 7>;


def RelAlg_AggrFunc : I64EnumAttr<"AggrFunc", "aggregation function",
        [RelAlg_Func_Sum, RelAlg_Func_Min, RelAlg_Func_Max,
         RelAlg_Func_Avg, RelAlg_Func_Count, RelAlg_Func_Any,RelAlg_Func_StdDevSamp,RelAlg_Func_VarSamp]> {
    let cppNamespace = "::lingodb::compiler::dialect::relalg";
}

def RelAlg_SetSemantic_Distinct : I64EnumAttrCase<"distinct", 0>;
def RelAlg_SetSemantic_All      : I64EnumAttrCase<"all", 1>;

def RelAlg_SetSemantic : I64EnumAttr<"SetSemantic", "",
        [RelAlg_SetSemantic_Distinct,RelAlg_SetSemantic_All]> {
    let cppNamespace = "::lingodb::compiler::dialect::relalg";
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

def ConstRelationOp : RelAlg_Op<"const_relation", [Pure,Operator]> {
    let summary = "constant relation";

    let arguments = (ins ArrayAttr:$columns,ArrayAttr:$values);
    let results = (outs TupleStream:$result);
    let assemblyFormat = [{ `columns` `:` custom<CustDefArr>($columns) `values` `:` $values attr-dict}];

    let extraClassDeclaration = [{
        ColumnSet getCreatedColumns();
    }];
}

def BaseTableOp : RelAlg_Op<"basetable",[Pure,Operator]> {
    let summary = "base table operation";

    let arguments = (ins StrAttr: $table_identifier, TableMetaDataAttr: $meta, DictionaryAttr: $columns);
    let results = (outs TupleStream:$result);

    let extraClassDeclaration = [{
        ColumnSet getCreatedColumns();
        lingodb::compiler::dialect::relalg::FunctionalDependencies getFDs();
    }];
}

//===----------------------------------------------------------------------===//
// Operators
//===----------------------------------------------------------------------===//

def SelectionOp : RelAlg_Op<"selection",
        [Pure,Operator,PredicateOperator,TupleLamdaOperator,
         UnaryOperator,DeclareOpInterfaceMethods<ColumnFoldable>]> {
    let summary = "selection operation";
    let description = [{
        Filter tuple stream, the region returns `1` iff the value should be
        contained in the output stream.
    }];

    let arguments = (ins TupleStream:$rel);
    let results = (outs TupleStream:$result);
    let regions = (region SizedRegion<1>:$predicate);
    let assemblyFormat = [{ $rel custom<CustRegion>($predicate) attr-dict-with-keyword }];
    let extraClassDeclaration = [{
        mlir::LogicalResult foldColumns(dialect::relalg::ColumnFoldInfo& columnInfo);
        lingodb::compiler::dialect::relalg::FunctionalDependencies getFDs();
    }];
}

def MapOp : RelAlg_Op<"map", [Pure,Operator,TupleLamdaOperator,UnaryOperator,DeclareOpInterfaceMethods<ColumnFoldable>]> {
    let summary = "map operation";

    let arguments = (ins TupleStream:$rel,ArrayAttr:$computed_cols);
    let results = (outs TupleStream:$result);
    let regions = (region SizedRegion<1>:$predicate);
    let assemblyFormat = [{  $rel `computes` `:` custom<CustDefArr>($computed_cols) custom<CustRegion>($predicate) attr-dict-with-keyword }];

    let extraClassDeclaration = [{
        ColumnSet getCreatedColumns();
        mlir::LogicalResult foldColumns(dialect::relalg::ColumnFoldInfo& columnInfo);
        mlir::LogicalResult eliminateDeadColumns(dialect::relalg::ColumnSet& usedColumns, mlir::Value& newStream);
    }];
    let hasVerifier=1;
}

def LimitOp : RelAlg_Op<"limit", [Pure,Operator,UnaryOperator]> {
    let summary = "limit operation";

    let arguments = (ins I32Attr:$max_rows, TupleStream:$rel);
    let results = (outs TupleStream:$result);
    let assemblyFormat = [{ $max_rows $rel attr-dict }];
}

def TopKOp : RelAlg_Op<"topk", [Pure,Operator,UnaryOperator]> {
    let summary = "top k operation";
    let description = [{
        Return top `max_rows` elements according to the sorting region. This is
        conceptually similar to a combination of `sort` and `limit`, but allows
        for further optimizations.
    }];

    let arguments = (ins I32Attr:$max_rows, TupleStream:$rel,ArrayAttr:$sortspecs);
    let results = (outs TupleStream:$result);
    let assemblyFormat = [{ $max_rows $rel custom<SortSpecs>($sortspecs) attr-dict}];
    let extraClassDeclaration = [{
        ColumnSet getUsedColumns();
    }];
}

def CrossProductOp : RelAlg_Op<"crossproduct",
        [Pure,Operator,BinaryOperator,DeclareOpInterfaceMethods<ColumnFoldable>]> {
    let summary = "crossproduct operation";

    let arguments = (ins TupleStream:$left,TupleStream:$right);
    let results = (outs TupleStream:$result);
    let assemblyFormat = [{ $left `,` $right attr-dict }];
    let extraClassDeclaration = [{
        mlir::LogicalResult foldColumns(dialect::relalg::ColumnFoldInfo& columnInfo);
    }];
}

def RenamingOp : RelAlg_Op<"renaming",
        [Pure,Operator,UnaryOperator]> {
    let summary = "renaming columns in relation";

    let arguments = (ins TupleStream:$rel,ArrayAttr:$columns);
    let results = (outs TupleStream:$result);
    let assemblyFormat = [{ $rel `renamed` `:` custom<CustDefArr>($columns) attr-dict-with-keyword }];

    let extraClassDeclaration = [{
        ColumnSet getCreatedColumns();
        ColumnSet getUsedColumns();
        ColumnSet getAvailableColumns();
        bool canColumnReach(Operator source, Operator target, const lingodb::compiler::dialect::tuples::Column* col);
    }];
}

def AggregationOp : RelAlg_Op<"aggregation", [Pure,Operator,UnaryOperator]> {
    let summary = "aggregation operation";

    let arguments = (ins TupleStream:$rel, ArrayAttr:$group_by_cols, ArrayAttr:$computed_cols);
    let results = (outs TupleStream:$result);
    let regions = (region SizedRegion<1>:$aggr_func);
    let assemblyFormat = [{ $rel custom<CustRefArr>($group_by_cols) `computes` `:` custom<CustDefArr>($computed_cols) custom<CustRegion>($aggr_func)  attr-dict-with-keyword }];

    let extraClassDeclaration = [{
        ColumnSet getUsedColumns();
        ColumnSet getAvailableColumns();
        ColumnSet getCreatedColumns();
        bool canColumnReach(Operator source, Operator target, const lingodb::compiler::dialect::tuples::Column* col);
        lingodb::compiler::dialect::relalg::FunctionalDependencies getFDs();
    }];
}

def GroupJoinOp : RelAlg_Op<"groupjoin", [Pure,Operator,BinaryOperator]> {
    let summary = "groupjoin operation";

    let arguments = (ins TupleStream:$left,TupleStream:$right, RelAlg_GroupJoinBehavior:$behavior, ArrayAttr:$leftCols, ArrayAttr:$rightCols, ArrayAttr:$mapped_cols,ArrayAttr:$computed_cols);
    let results = (outs TupleStream:$result);
    let regions = (region SizedRegion<1>:$predicate,SizedRegion<1>:$map_func,SizedRegion<1>:$aggr_func);
    let assemblyFormat = [{ $left `,` $right `behavior` `=` $behavior custom<CustRefArr>($leftCols) `=` custom<CustRefArr>($rightCols) custom<CustRegion>($predicate) `mapped` `:` custom<CustDefArr>($mapped_cols) custom<CustRegion>($map_func) `computes` `:` custom<CustDefArr>($computed_cols) custom<CustRegion>($aggr_func)  attr-dict-with-keyword }];
    let extraClassDeclaration = [{
        ColumnSet getUsedColumns();
        ColumnSet getAvailableColumns();
        ColumnSet getCreatedColumns();
        bool canColumnReach(Operator source, Operator target, const lingodb::compiler::dialect::tuples::Column* col);
    }];
}
def WindowOp : RelAlg_Op<"window", [Pure,Operator,UnaryOperator]> {
    let summary = "window operation";

    let arguments = (ins TupleStream:$rel, ArrayAttr:$partition_by,ArrayAttr:$order_by, ArrayAttr:$computed_cols,I64Attr:$from,I64Attr:$to);
    let results = (outs TupleStream:$result);
    let regions = (region SizedRegion<1>:$aggr_func);
    let assemblyFormat = [{ $rel `partitionBy` `:` custom<CustRefArr>($partition_by) `orderBy` `:` custom<SortSpecs>($order_by) `rows_between` `:` $from `and` $to `computes` `:` custom<CustDefArr>($computed_cols) custom<CustRegion>($aggr_func)  attr-dict-with-keyword }];

    let extraClassDeclaration = [{
        ColumnSet getUsedColumns();
        ColumnSet getCreatedColumns();
    }];
}

def SortOp : RelAlg_Op<"sort", [Pure,Operator,UnaryOperator]> {
    let summary = "sort operation";

    let arguments = (ins TupleStream:$rel, ArrayAttr:$sortspecs);
    let results = (outs TupleStream:$result);
    let assemblyFormat = [{ $rel custom<SortSpecs>($sortspecs) attr-dict}];
    let extraClassDeclaration = [{
        ColumnSet getUsedColumns();
    }];
}

// Joins
class RelAlg_JoinOp<string mnemonic> : RelAlg_Op<mnemonic,
        [Pure,Operator,BinaryOperator,PredicateOperator,
         TupleLamdaOperator,DeclareOpInterfaceMethods<ColumnFoldable>]> {
    let arguments = (ins TupleStream:$left,TupleStream:$right);
    let results = (outs TupleStream:$result);
    let regions = (region SizedRegion<1>:$predicate);
    let assemblyFormat = [{ $left `,` $right custom<CustRegion>($predicate) attr-dict-with-keyword }];
}

def InnerJoinOp : RelAlg_JoinOp<"join"> {
    let summary = "inner join operation";
    let extraClassDeclaration = [{
        mlir::LogicalResult foldColumns(dialect::relalg::ColumnFoldInfo& columnInfo);
        lingodb::compiler::dialect::relalg::FunctionalDependencies getFDs();
    }];
}

def SemiJoinOp : RelAlg_JoinOp<"semijoin"> {
    let summary = "semi join operation";
    let extraClassDeclaration = [{
        ColumnSet getAvailableColumns();
        mlir::LogicalResult foldColumns(dialect::relalg::ColumnFoldInfo& columnInfo);
        bool canColumnReach(Operator source, Operator target, const lingodb::compiler::dialect::tuples::Column* col);
        lingodb::compiler::dialect::relalg::FunctionalDependencies getFDs();
    }];
}

def AntiSemiJoinOp : RelAlg_JoinOp<"antisemijoin"> {
    let summary = "antisemi join operation";

    let extraClassDeclaration = [{
        ColumnSet getAvailableColumns();
        mlir::LogicalResult foldColumns(dialect::relalg::ColumnFoldInfo& columnInfo);
        bool canColumnReach(Operator source, Operator target, const lingodb::compiler::dialect::tuples::Column* col);
        lingodb::compiler::dialect::relalg::FunctionalDependencies getFDs();
    }];
}

def OuterJoinOp : RelAlg_JoinOp<"outerjoin"> {
    let summary = "outer join operation";

    let arguments = (ins TupleStream:$left,TupleStream:$right,ArrayAttr:$mapping);
    let assemblyFormat = [{ $left `,` $right custom<CustRegion>($predicate)  custom<CustAttrMapping>($mapping) attr-dict-with-keyword }];

    let extraClassDeclaration = [{
        ColumnSet getCreatedColumns();
        ColumnSet getUsedColumns();
        ColumnSet getAvailableColumns();
        bool canColumnReach(Operator source, Operator target, const lingodb::compiler::dialect::tuples::Column* col);
    }];

}

def FullOuterJoinOp : RelAlg_JoinOp<"fullouterjoin"> {
    let summary = "full outer join operation";
    let arguments = (ins TupleStream:$left,TupleStream:$right,ArrayAttr:$mapping);
    let assemblyFormat = [{ $left `,` $right custom<CustRegion>($predicate)  custom<CustAttrMapping>($mapping) attr-dict-with-keyword }];

    let extraClassDeclaration = [{
        ColumnSet getCreatedColumns();
        ColumnSet getUsedColumns();
        ColumnSet getAvailableColumns();
        bool canColumnReach(Operator source, Operator target, const lingodb::compiler::dialect::tuples::Column* col);
    }];
}

def SingleJoinOp : RelAlg_JoinOp<"singlejoin"> {
    let summary = "single join operation";

    let arguments = (ins TupleStream:$left,TupleStream:$right,ArrayAttr:$mapping);
    let assemblyFormat = [{  $left `,` $right custom<CustRegion>($predicate)  custom<CustAttrMapping>($mapping) attr-dict-with-keyword }];

    let extraClassDeclaration = [{
        ColumnSet getCreatedColumns();
        ColumnSet getUsedColumns();
        ColumnSet getAvailableColumns();
        bool canColumnReach(Operator source, Operator target, const lingodb::compiler::dialect::tuples::Column* col);
    }];
}

def CollectionJoinOp : RelAlg_JoinOp<"collectionjoin"> {
    let summary = "collection join operation";

    let arguments = (ins ArrayAttr:$cols,ColumnDefAttr:$collAttr,TupleStream:$left,TupleStream:$right);
    let assemblyFormat = [{ custom<CustRefArr>($cols) custom<CustDef>($collAttr) $left `,` $right custom<CustRegion>($predicate)  attr-dict }];

    let extraClassDeclaration = [{
        ColumnSet getCreatedColumns();
        ColumnSet getUsedColumns();
        ColumnSet getAvailableColumns();
        bool canColumnReach(Operator source, Operator target, const lingodb::compiler::dialect::tuples::Column* col);
    }];
}

def MarkJoinOp : RelAlg_JoinOp<"markjoin"> {
    let summary = "mark join operation";

    let arguments = (ins ColumnDefAttr:$markattr,TupleStream:$left,TupleStream:$right);
    let assemblyFormat = [{ custom<CustDef>($markattr) $left `,` $right custom<CustRegion>($predicate)  attr-dict }];

    let extraClassDeclaration = [{
        ColumnSet getAvailableColumns();
        ColumnSet getCreatedColumns();
        bool canColumnReach(Operator source, Operator target, const lingodb::compiler::dialect::tuples::Column* col);
    }];
}

// Set Operations
def ProjectionOp : RelAlg_Op<"projection",[Pure,Operator,UnaryOperator]> {
    let summary = "projection operation";

    let arguments = (ins RelAlg_SetSemantic:$set_semantic,TupleStream:$rel, ArrayAttr:$cols);
    let results = (outs TupleStream:$result);
    let assemblyFormat = [{ $set_semantic custom<CustRefArr>($cols) $rel attr-dict }];

    let extraClassDeclaration = [{
        ColumnSet getAvailableColumns();
        ColumnSet getUsedColumns();
        bool canColumnReach(Operator source, Operator target, const lingodb::compiler::dialect::tuples::Column* col);
    }];
}

class RelAlg_BinarySetOp<string mnemonic> : RelAlg_Op<mnemonic,
        [Pure,Operator,BinaryOperator]> {
    let arguments = (ins RelAlg_SetSemantic:$set_semantic,TupleStream:$left,TupleStream:$right,ArrayAttr:$mapping);
    let results = (outs TupleStream:$result);
    let assemblyFormat = [{ $set_semantic $left `,` $right custom<CustAttrMapping>($mapping) attr-dict }];
    let extraClassDeclaration = [{
        ColumnSet getCreatedColumns();
        ColumnSet getUsedColumns();
        ColumnSet getAvailableColumns();
        bool canColumnReach(Operator source, Operator target, const lingodb::compiler::dialect::tuples::Column* col);
    }];
    let extraClassDefinition =[{
        ColumnSet $cppClass::getCreatedColumns(){ return lingodb::compiler::dialect::relalg::detail::getSetOpCreatedColumns(this->getOperation()); }
        ColumnSet $cppClass::getUsedColumns(){ return lingodb::compiler::dialect::relalg::detail::getSetOpUsedColumns(this->getOperation()); }
        ColumnSet $cppClass::getAvailableColumns(){ return lingodb::compiler::dialect::relalg::detail::getSetOpCreatedColumns(this->getOperation()); }
        bool $cppClass::canColumnReach(Operator source, Operator target, const lingodb::compiler::dialect::tuples::Column* col){
            if(!source){
                return lingodb::compiler::dialect::relalg::detail::canColumnReach(this->getOperation(),source,target,col);
            }
            return false;
        };

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

def NestedOp : RelAlg_Op<"nested", [Pure,Operator]> {
    let summary = "nested";
    let arguments = (ins Variadic<TupleStream>:$inputs,ArrayAttr:$used_cols,ArrayAttr:$available_cols);
    let results = (outs TupleStream:$result);
    let regions = (region SizedRegion<1>:$nested_fn);
    let extraClassDeclaration = [{
        ColumnSet getCreatedColumns();
        ColumnSet getUsedColumns();
        ColumnSet getAvailableColumns();
        bool canColumnReach(Operator source, Operator target, const lingodb::compiler::dialect::tuples::Column* col);
    }];
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility
////////////////////////////////////////////////////////////////////////////////////////////////////

def AggrFuncOp : RelAlg_Op<"aggrfn", [Pure]> {
    let summary = "aggregation operation";

    let arguments = (ins RelAlg_AggrFunc:$fn,TupleStream:$rel,ColumnRefAttr:$attr);
    let results = (outs AnyType:$result);
    let assemblyFormat = [{  $fn custom<CustRef>($attr) $rel `:` type($result) attr-dict }];
}

def CountRowsOp : RelAlg_Op<"count", [Pure]> {
    let summary = "count rows";

    let arguments = (ins TupleStream:$rel);
    let results = (outs I64:$res);
    let assemblyFormat = [{  $rel attr-dict }];
}
def RankOp : RelAlg_Op<"rank", [Pure]> {
    let summary = "rank inside window";

    let arguments = (ins TupleStream:$rel);
    let results = (outs I64:$res);
    let assemblyFormat = [{  $rel attr-dict }];
}

def ExistsOp : RelAlg_Op<"exists", [Pure]> {
    let arguments = (ins TupleStream:$rel);
    let results = (outs I1:$result);
    let assemblyFormat = [{  $rel attr-dict }];
}

def MaterializeOp : RelAlg_Op<"materialize", [Pure]> {
    let arguments = (ins TupleStream:$rel,ArrayAttr:$cols,ArrayAttr:$columns);
    let results = (outs SubOp_ResultTable:$result);
    let assemblyFormat = [{  $rel custom<CustRefArr>($cols) `=` `>` $columns `:` type($result) attr-dict }];
}

def TmpOp : RelAlg_Op<"tmp", [Pure,Operator]> {
    let summary = "temporary materialization";

    let arguments = (ins TupleStream:$rel,ArrayAttr:$cols);
    let results = (outs Variadic<TupleStream>:$resulting);
    let assemblyFormat = [{  $rel custom<CustRefArr>($cols) `->` type($resulting) attr-dict }];
}

def GetScalarOp : RelAlg_Op<"getscalar", [Pure]> {
    let summary = "get column";

    let arguments = (ins ColumnRefAttr:$attr, TupleStream:$rel);
    let results = (outs AnyType : $res);
    let assemblyFormat = [{  custom<CustRef>($attr) $rel `:` type($res) attr-dict }];
}

def GetListOp : RelAlg_Op<"getlist", [Pure]> {
    let summary = "get multiple columns";

    let arguments = (ins TupleStream:$rel,ArrayAttr:$cols);
    let results = (outs DSA_Collection:$res);
    let assemblyFormat = [{ $rel custom<CustRefArr>($cols) `:` type($res) attr-dict }];
}

def InOp : RelAlg_Op<"in", [Pure]> {
    let summary = "in operation";
    let arguments = (ins AnyType:$val, TupleStream:$rel);
    let results = (outs I1 : $res);
    let assemblyFormat = [{ $val `:` type($val) `,` $rel attr-dict }];
}

def TrackTuplesOP : RelAlg_Op<"track_tuples", [Operator]> {
    let summary = "track tuples";
    let arguments = (ins TupleStream:$rel, I32Attr:$resultId);
    let assemblyFormat = [{ $rel $resultId attr-dict }];
}

def QueryOp : RelAlg_Op<"query"> {
    let summary = "query";
    let arguments = (ins Variadic<AnyType>:$inputs);
    let results = (outs Variadic<AnyType>:$results);
    let regions = (region SizedRegion<1>:$query_ops);
    let assemblyFormat = [{ ($inputs^ `:` type($inputs))? custom<CustRegion>($query_ops)  `->` type($results) attr-dict-with-keyword }];
}
def QueryReturnOp : RelAlg_Op<"query_return",[Terminator]> {
   let summary = "query return";
   let arguments = (ins Variadic<AnyType>:$inputs);
   let assemblyFormat = [{ ($inputs^ `:` type($inputs))? attr-dict-with-keyword }];
}


#endif // LINGODB_DIALECT_RELALG_IR_OPS
