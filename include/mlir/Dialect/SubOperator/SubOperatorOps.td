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
include "SubOpInterfaces.td"


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
    let useDefaultTypePrinterParser=1;
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

class SubOperator_Type<string name, string typeMnemonic, list<Trait> traits = []> : TypeDef<SubOperator_Dialect, name,traits> {
    let mnemonic = typeMnemonic;
}
def TableRef : SubOperator_Type<"TableRef", "table_ref"> {
    let summary = "reference to table";
    let parameters = (ins "StateMembersAttr":$members);
    let assemblyFormat = "`<` custom<StateMembers>($members) `>`";
}
def Table : SubOperator_Type<"Table", "table"> {
    let summary = "reference to table";
    let parameters = (ins "StateMembersAttr":$members);
    let assemblyFormat = "`<` custom<StateMembers>($members) `>`";
}
def SimpleState : SubOperator_Type<"SimpleState", "simple_state",[State,StateSupportingLookup]> {
    let summary = "simple state with multiple members";
    let parameters = (ins "StateMembersAttr":$members);
    let assemblyFormat = "`<` custom<StateMembers>($members) `>`";
        let extraClassDeclaration = [{
            StateMembersAttr getValueMembers();
        }];
}
def Hashmap : SubOperator_Type<"HashMap", "hashmap",[State,StateSupportingLookup]> {
    let summary = "simple state with multiple members";
    let parameters = (ins "StateMembersAttr":$keyMembers,"StateMembersAttr":$valueMembers );
    let assemblyFormat = "`<` custom<StateMembers>($keyMembers) `,` custom<StateMembers>($valueMembers) `>`";
    let extraClassDeclaration = [{
        StateMembersAttr getMembers();
    }];
}
def LazyMultiMap : SubOperator_Type<"LazyMultiMap", "lazy_multi_map",[State,StateSupportingLookup]> {
    let summary = "simple state with multiple members";
    let parameters = (ins "StateMembersAttr":$keyMembers,"StateMembersAttr":$valueMembers );
    let assemblyFormat = "`<` custom<StateMembers>($keyMembers) `,` custom<StateMembers>($valueMembers) `>`";
    let extraClassDeclaration = [{
        StateMembersAttr getMembers();
    }];
}
def Vector : SubOperator_Type<"Vector", "vector", [State,StateSupportingLookup]> {
    let summary = "vector type";
    let parameters = (ins "StateMembersAttr":$members);
    let assemblyFormat = "`<` custom<StateMembers>($members) `>`";
            let extraClassDeclaration = [{
                StateMembersAttr getValueMembers();
            }];
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

def LookupOrInsertOp : SubOperator_Op<"lookup_or_insert",[SubOperator]> {
    let summary = "performs a lookup in a state and annotates the result as attribute ";
    let arguments = (ins TupleStream: $stream, AnyType: $state, ArrayAttr: $keys,ColumnDefAttr:$ref);
    let results = (outs TupleStream: $res);
    let regions = (region AnyRegion:$eqFn, AnyRegion:$initFn);
      let hasCustomAssemblyFormat = 1;
      let extraClassDeclaration = [{
          std::vector<std::string> getWrittenMembers();
      }];
}
def LookupOp : SubOperator_Op<"lookup",[SubOperator]> {
    let summary = "performs a lookup in a state and annotates the result as attribute ";
    let arguments = (ins TupleStream: $stream, AnyType: $state, ArrayAttr: $keys,ColumnDefAttr:$ref);
    let results = (outs TupleStream: $res);
    let regions = (region AnyRegion:$eqFn, AnyRegion:$initFn);
      let hasCustomAssemblyFormat = 1;
      let extraClassDeclaration = [{
          std::vector<std::string> getReadMembers();
      }];
}

def MaintainOp : SubOperator_Op<"maintain",[SubOperator]> {
      let summary = "maintain";

      let arguments = (ins AnyType : $state, StrAttr: $description );
      let assemblyFormat = [{  $description $state `:` type($state)  attr-dict }];
      let extraClassDeclaration = [{
          std::vector<std::string> getReadMembers();
          std::vector<std::string> getWrittenMembers();
      }];
  }

def GatherOp : SubOperator_Op<"gather",[SubOperator]> {
    let summary = "scatter";

    let arguments = (ins TupleStream : $stream, ColumnRefAttr: $ref, DictionaryAttr : $mapping );
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{  $stream custom<CustRef>($ref) custom<StateColumnMapping>($mapping)  attr-dict }];
    let extraClassDeclaration = [{
        std::vector<std::string> getReadMembers();
    }];
}
def ScatterOp : SubOperator_Op<"scatter",[SubOperator]> {
    let summary = "scatter";

    let arguments = (ins TupleStream : $stream, ColumnRefAttr: $ref, DictionaryAttr : $mapping );
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{  $stream custom<CustRef>($ref) custom<ColumnStateMapping>($mapping)  attr-dict }];
    let extraClassDeclaration = [{
        std::vector<std::string> getWrittenMembers();
    }];
}
def FlattenOp : SubOperator_Op<"flatten",[SubOperator]> {
  let summary = "flattens an list column";
  let description = [{#
   Example:
   ```mlir
   %stream2 = subop.flatten %stream @a::@list => @scope::@list_entry({})
   ```
  }];
}

def ReduceOp : SubOperator_Op<"reduce",[SubOperator]> {
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
  let regions = (region SizedRegion<1>:$region);
  let arguments = (ins TupleStream: $stream, ColumnRefAttr: $ref, ArrayAttr: $columns, ArrayAttr: $members);
  let hasCustomAssemblyFormat = 1;
        let extraClassDeclaration = [{
            std::vector<std::string> getWrittenMembers();
            std::vector<std::string> getReadMembers();
        }];
}

def NestedMapOp : SubOperator_Op<"nested_map",[SubOperator]> {
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
    let arguments= (ins TupleStream : $stream, ArrayAttr: $parameters);
    let results = (outs TupleStream : $res);
    let regions = (region SizedRegion<1>:$region);
    let hasCustomAssemblyFormat = 1;
    let extraClassDeclaration = [{
        std::vector<std::string> getWrittenMembers();
        std::vector<std::string> getReadMembers();
    }];
}
def GetReferenceOp : SubOperator_Op<"get_reference", [Pure]> {
    let summary = "get reference to an externally defined object";

    let arguments = (ins StrAttr: $descr);
    let results = (outs AnyType : $res);
    let assemblyFormat = [{  $descr `:` type($res) attr-dict }];
}
def CreateOp : SubOperator_Op<"create", [Pure]> {
    let summary = "create new state";

    let arguments = (ins OptionalAttr<AnyAttr>: $descr);
    let results = (outs AnyType : $res);
    let regions = (region VariadicRegion<AnyRegion>:$initFn);
    let assemblyFormat = [{  ($descr^)? `->` type($res) (`initial` `:` $initFn^)? attr-dict }];

}

def ScanOp : SubOperator_Op<"scan", [Pure,SubOperator]> {
    let summary = "scan state (sequentially)";

    let arguments = (ins AnyType : $state, DictionaryAttr : $mapping );
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{  $state `:` type($state) custom<StateColumnMapping>($mapping) attr-dict }];
    let extraClassDeclaration = [{
        std::vector<std::string> getReadMembers();
    }];
}
def ScanListOp : SubOperator_Op<"scan_list", [Pure,SubOperator]> {
    let summary = "scan state (sequentially)";

    let arguments = (ins AnyType : $list, ColumnDefAttr: $elem);
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{  $list `:` type($list) custom<CustDef>($elem) attr-dict }];
}

def ScanRefsOp : SubOperator_Op<"scan_refs", [Pure,SubOperator]> {
    let summary = "scan state (sequentially) and emits references to its elements";

    let arguments = (ins AnyType : $state, ColumnDefAttr:$ref);
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{  $state `:` type($state) custom<CustDef>($ref) attr-dict }];
}

def InFlightOp : SubOperator_Op<"in_flight", [Pure,SubOperator]> {
    let summary = "operation symbolizes 'in flight' tuple stream -> values => columns";

    let arguments = (ins Variadic<AnyType> : $values, ArrayAttr : $columns );
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{  $values `:` type($values) `=` `>` custom<CustDefArr>($columns) attr-dict }];
}
def InFlightTupleOp : SubOperator_Op<"in_flight_tuple", [Pure,SubOperator]> {
    let summary = "operation symbolizes 'in flight' tuple -> values => columns";

    let arguments = (ins Variadic<AnyType> : $values, ArrayAttr : $columns );
    let results = (outs TupleStream_Tuple : $res);
    let assemblyFormat = [{  $values `:` type($values) `=` `>` custom<CustDefArr>($columns) attr-dict }];
}

def CombineTupleOp : SubOperator_Op<"combine_tuple", [Pure,SubOperator]> {
    let summary = "combines stream with tuple";

    let arguments = (ins TupleStream :$stream, TupleStream_Tuple:$right);
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{  $stream `,` $right attr-dict }];
}
def UnionOp : SubOperator_Op<"union", [Pure,SubOperator]> {
    let summary = "union of two stream";

    let arguments = (ins Variadic<TupleStream> :$streams);
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{  $streams attr-dict }];
}

def MapOp : SubOperator_Op<"map", [Pure,SubOperator]> {
    let summary = "map operation";

    let arguments = (ins TupleStream:$stream,ArrayAttr:$computed_cols);
    let results = (outs TupleStream:$result);
    let regions = (region SizedRegion<1>:$fn);
    let assemblyFormat = [{  $stream `computes` `:` custom<CustDefArr>($computed_cols) custom<CustRegion>($fn) attr-dict-with-keyword }];
}
def SortOp : SubOperator_Op<"sort", [IsolatedFromAbove,SubOperator]> {
    let summary = "sort vector";
    let description = [{
        Sort a records in a vector. The region yields the order of two arguments as in C++ `std::sort`.
    }];

    let arguments = (ins Vector:$toSort,ArrayAttr: $sortBy);
    let regions = (region SizedRegion<1>:$region);
  let hasCustomAssemblyFormat = 1;
      let extraClassDeclaration = [{
          std::vector<std::string> getWrittenMembers();
          std::vector<std::string> getReadMembers();
      }];
}

def FilterSemantic_AllTrue : I64EnumAttrCase<"all_true", 0>;
def FilterSemantic_NoneTrue      : I64EnumAttrCase<"none_true", 1>;

def FilterSemantic : I64EnumAttr<"FilterSemantic", "",
        [FilterSemantic_AllTrue,FilterSemantic_NoneTrue]> {
    let cppNamespace = "::mlir::subop";
}

def FilterOp : SubOperator_Op<"filter", [Pure,SubOperator]> {
    let summary = "filter stream of tuples according to one or more boolean columns (all 1 => keep tuple, otherwise discard) ";
    let arguments = (ins TupleStream : $stream, FilterSemantic:$filterSemantic,ArrayAttr : $conditions );
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{  $stream $filterSemantic custom<CustRefArr>($conditions)  attr-dict }];
}
def RepeatOp : SubOperator_Op<"repeat", [Pure,SubOperator]> {
    let summary = "repeat the current tuple x times";
    let arguments = (ins TupleStream : $stream, ColumnRefAttr:$times);
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{  $stream custom<CustRef>($times)  attr-dict }];
}
def RenamingOp : SubOperator_Op<"rename", [Pure,SubOperator]> {
    let summary = "rename columns";

    let arguments = (ins TupleStream:$stream,ArrayAttr:$columns);
    let results = (outs TupleStream:$result);
    let assemblyFormat = [{ $stream `renamed` `:` custom<CustDefArr>($columns) attr-dict-with-keyword }];
}
def MaterializeOp : SubOperator_Op<"materialize",[SubOperator]> {
    let summary = "materialize state (sequentially)";

    let arguments = (ins TupleStream : $stream, AnyType : $state, DictionaryAttr : $mapping );
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{  $stream custom<ColumnStateMapping>($mapping) `,` $state `:` type($state)  attr-dict }];
    let extraClassDeclaration = [{
        std::vector<std::string> getWrittenMembers();
    }];
}
def ConvertToExplicit : SubOperator_Op<"convert_to_explicit",[SubOperator]> {
    let summary = "converts a declarative state to an explicit state";

    let arguments = (ins AnyType : $state);
    let results = (outs AnyType : $explicit_state);
    let assemblyFormat = [{  $state `:` type($state) `->` type($explicit_state)  attr-dict }];
    let extraClassDeclaration = [{
        std::vector<std::string> getReadMembers();
    }];
}
#endif // LINGODB_DIALECT_SubOperator_IR_OPS
