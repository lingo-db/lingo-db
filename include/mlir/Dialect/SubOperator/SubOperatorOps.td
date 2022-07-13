#ifndef LINGODB_DIALECT_SubOperator_IR_OPS
#define LINGODB_DIALECT_SubOperator_IR_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Dialect/TupleStream/TupleStreamBase.td"


//===----------------------------------------------------------------------===//
// SubOperator dialect definition.
//===----------------------------------------------------------------------===//

def SubOperator_Dialect : Dialect {
    let name = "subop";
    let summary = "A dialect around sub-operators that manipulate a tuplestream, but state is managed explicitly";
    let cppNamespace = "::mlir::subop";
        let extraClassDeclaration = [{
            private:
            void registerTypes();
            void registerAttrs();
            public:
        }];
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

class SubOperator_Attr<string name> : AttrDef<SubOperator_Dialect, name>;

def StateMembers : SubOperator_Attr<"StateMembers"> {
    let mnemonic = "state_members";
    let parameters = (ins "ArrayAttr":$names,"ArrayAttr":$types);
    let assemblyFormat ="`<` $names `:` $types `>`";
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

class SubOperator_Type<string name, string typeMnemonic> : TypeDef<SubOperator_Dialect, name> {
    let mnemonic = typeMnemonic;
}
def TableRef : SubOperator_Type<"TableRef", "table_ref"> {
    let summary = "reference to table";
    let parameters = (ins "StateMembersAttr":$columns);
    let assemblyFormat = "`<` custom<StateMembers>($columns) `>`";
}
def Table : SubOperator_Type<"Table", "table"> {
    let summary = "reference to table";
    let parameters = (ins "StateMembersAttr":$columns);
    let assemblyFormat = "`<` custom<StateMembers>($columns) `>`";
}
def SimpleState : SubOperator_Type<"SimpleState", "simple_state"> {
    let summary = "simple state with multiple members";
    let parameters = (ins "StateMembersAttr":$columns);
    let assemblyFormat = "`<` custom<StateMembers>($columns) `>`";
}
def Vector : SubOperator_Type<"Vector", "vector"> {
    let summary = "vector type";
    let parameters = (ins "StateMembersAttr":$columns);
    let assemblyFormat = "`<` custom<StateMembers>($columns) `>`";
}

def Entry : SubOperator_Type<"Entry", "entry"> {
    let summary = "entry of some state";
    let parameters = (ins "Type":$t);
    let assemblyFormat = "`<` $t `>`";
}
def EntryRef : SubOperator_Type<"EntryRef", "entry_ref"> {
    let summary = "reference to entry of some state";
    let parameters = (ins "Type":$t);
    let assemblyFormat = "`<` $t `>`";
}
def List : SubOperator_Type<"List", "list"> {
    let summary = "list type";
    let parameters = (ins "Type":$t);
    let assemblyFormat = "`<` $t `>`";
}

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//
class SubOperator_Op<string mnemonic, list<Trait> traits = []> : Op<SubOperator_Dialect, mnemonic, traits> {}

def ScatterByEntryRef : SubOperator_Op<"scatter_ref"> {
    let summary = "writes to the entry of a state (randomly) using a supplied entry reference";
}

def GatherOp : SubOperator_Op<"gather"> {
  let summary = "reads member values  using a given value";
}
def FlattenOp : SubOperator_Op<"flatten"> {
  let summary = "flattens an list column";
  let description = [{#
   Example:
   ```mlir
   %stream2 = subop.flatten %stream @a::@list => @scope::@list_entry({})
   ```
  }];
}

def ReduceOp : SubOperator_Op<"reduce"> {
  let summary = "reduce";
  let description =[{
  ```mlir
    %5 = subop.reduce %2 %0 [@studenten::@semester] with ["ctr","sum"] ([%v1], [%curr_ctr, %curr_sum]){
      %c1 = db.constant %c1
      %updated_ctr = db.add %curr_ctr, %c1
      tuples.return %updated_ctr, %updated_sum
    } eq: (){} combine: (){} initial: (){}
  ```
  }];
}

def NestedMapOp : SubOperator_Op<"nested_map"> {
   let summary = "compute nested map";
  let description = [{#
     Example:
     ```mlir
     %stream2 = subop.nested_map %stream [@x::@a, @y::@b] (%a, %b){
        %stream3 = subop.scan %some_table ...
        %stream4 = subop.filter ... {
            ... %a
            ... %b
            tuples.return %pred
        }
        tuples.return %stream5
     }
     ```
    Semantics of the produced stream: each tuple is combined with the tuples returned by the nested_map
    If no tuple is returned -> the original tuple is ommited (-> "inner-join semantic")
  }];
    let arguments= (ins TupleStream : $stream,ArrayAttr : $parameters);
    let results = (outs TupleStream : $res);
    let regions = (region SizedRegion<1>:$region);
    let hasCustomAssemblyFormat = 1;
}
def GetReferenceOp : SubOperator_Op<"get_reference", [NoSideEffect]> {
    let summary = "get reference to an externally defined object";

    let arguments = (ins StrAttr: $descr);
    let results = (outs AnyType : $res);
    let assemblyFormat = [{  $descr `:` type($res) attr-dict }];
}
def CreateOp : SubOperator_Op<"create", [NoSideEffect]> {
    let summary = "create new state";

    let arguments = (ins StrAttr: $descr);
    let results = (outs AnyType : $res);
    let assemblyFormat = [{  $descr `:` type($res) attr-dict }];
}

def ScanOp : SubOperator_Op<"scan", [NoSideEffect]> {
    let summary = "scan state (sequentially)";

    let arguments = (ins AnyType : $state, DictionaryAttr : $mapping );
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{  $state `:` type($state) custom<StateColumnMapping>($mapping) attr-dict }];
}

def InFlightOp : SubOperator_Op<"in_flight", [NoSideEffect]> {
    let summary = "operation symbolizes 'in flight' tuple stream -> values => columns";

    let arguments = (ins Variadic<AnyType> : $values, ArrayAttr : $columns );
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{  $values `:` type($values) `=` `>` custom<CustDefArr>($columns) attr-dict }];
}
def CombineInFlightOp : SubOperator_Op<"combine_in_flight", [NoSideEffect]> {
    let summary = "combines in flight tuple streams";

    let arguments = (ins TupleStream :$left, TupleStream:$right);
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{  $left `,` $right attr-dict }];
}

def MapOp : SubOperator_Op<"map", [NoSideEffect]> {
    let summary = "map operation";

    let arguments = (ins TupleStream:$rel,ArrayAttr:$computed_cols);
    let results = (outs TupleStream:$result);
    let regions = (region SizedRegion<1>:$fn);
    let assemblyFormat = [{  $rel `computes` `:` custom<CustDefArr>($computed_cols) custom<CustRegion>($fn) attr-dict-with-keyword }];
}
def SortOp : SubOperator_Op<"sort", [IsolatedFromAbove]> {
    let summary = "sort vector";
    let description = [{
        Sort a records in a vector. The region yields the order of two arguments as in C++ `std::sort`.
    }];

    let arguments = (ins Vector:$toSort,ArrayAttr: $sortBy);
    let regions = (region SizedRegion<1>:$region);
  let hasCustomAssemblyFormat = 1;
}


def FilterOp : SubOperator_Op<"filter", [NoSideEffect]> {
    let summary = "filter stream of tuples according to one or more boolean columns (all 1 => keep tuple, otherwise discard) ";
    let arguments = (ins TupleStream : $stream, ArrayAttr : $conditions );
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{  $stream custom<CustRefArr>($conditions)  attr-dict }];
}
def RenamingOp : SubOperator_Op<"rename", [NoSideEffect]> {
    let summary = "rename columns";

    let arguments = (ins TupleStream:$stream,ArrayAttr:$columns);
    let results = (outs TupleStream:$result);
    let assemblyFormat = [{ $stream `renamed` `:` custom<CustDefArr>($columns) attr-dict-with-keyword }];
}
def MaterializeOp : SubOperator_Op<"materialize"> {
    let summary = "materialize state (sequentially)";

    let arguments = (ins TupleStream : $stream, AnyType : $state, DictionaryAttr : $mapping );
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{  $stream custom<ColumnStateMapping>($mapping) `,` $state `:` type($state)  attr-dict }];
}
def ConvertToExplicit : SubOperator_Op<"convert_to_explicit"> {
    let summary = "converts a declarative state to an explicit state";

    let arguments = (ins AnyType : $state);
    let results = (outs AnyType : $explicit_state);
    let assemblyFormat = [{  $state `:` type($state) `->` type($explicit_state)  attr-dict }];
}
#endif // LINGODB_DIALECT_SubOperator_IR_OPS
