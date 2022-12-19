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
    let useDefaultAttributePrinterParser=1;
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
def Table : SubOperator_Type<"Table", "table",[State]> {
    let summary = "external table";
    let parameters = (ins "StateMembersAttr":$members);
    let assemblyFormat = "`<` custom<StateMembers>($members) `>`";
}
def ResultTable : SubOperator_Type<"ResultTable", "result_table",[State]> {
    let summary = "reference to table";
    let parameters = (ins "StateMembersAttr":$members);
    let assemblyFormat = "`<` custom<StateMembers>($members) `>`";
}
def SimpleState : SubOperator_Type<"SimpleState", "simple_state",[State,LookupAbleState]> {
    let summary = "simple state with multiple members";
    let parameters = (ins "StateMembersAttr":$members);
    let assemblyFormat = "`<` custom<StateMembers>($members) `>`";
        let extraClassDeclaration = [{
            StateMembersAttr getKeyMembers(){
            return mlir::subop::StateMembersAttr::get(getContext(), mlir::ArrayAttr::get(getContext(), {}), mlir::ArrayAttr::get(getContext(), {}));
            }
            StateMembersAttr getValueMembers();
        }];
}
def Hashmap : SubOperator_Type<"HashMap", "hashmap",[State,LookupAbleState]> {
    let summary = "simple state with multiple members";
    let parameters = (ins "StateMembersAttr":$keyMembers,"StateMembersAttr":$valueMembers );
    let assemblyFormat = "`<` custom<StateMembers>($keyMembers) `,` custom<StateMembers>($valueMembers) `>`";
    let extraClassDeclaration = [{
        StateMembersAttr getMembers();
    }];
}
def Buffer : SubOperator_Type<"Buffer", "buffer", [State]> {
    let summary = "growing buffer type";
    let parameters = (ins "StateMembersAttr":$members);
    let assemblyFormat = "`<` custom<StateMembers>($members) `>`";
}
def ContinuousView : SubOperator_Type<"ContinuousView", "continuous_view", [State]> {
    let summary = "continuous view type";
    let parameters = (ins "State":$based_on);
    let assemblyFormat = "`<` $based_on `>`";
     let extraClassDeclaration = [{
         StateMembersAttr getMembers(){return getBasedOn().getMembers();}
     }];
}

def Heap : SubOperator_Type<"Heap", "heap", [State]> {
    let summary = "heap type";
    let parameters = (ins "StateMembersAttr":$members, "uint32_t": $max_elements);
    let assemblyFormat = "`<` $max_elements `,` custom<StateMembers>($members) `>`";
}
def SortedView : SubOperator_Type<"SortedView", "sorted_view", [State]> {
    let summary = "sorted view type";
    let parameters = (ins "State":$based_on);
    let assemblyFormat = "`<` $based_on `>`";
     let extraClassDeclaration = [{
         StateMembersAttr getMembers(){return getBasedOn().getMembers();}
     }];
}
def HashIndexedView : SubOperator_Type<"HashIndexedView", "hash_indexed_view", [State,LookupAbleState]> {
    let summary = "sorted view type";
    let parameters = (ins "StateMembersAttr":$keyMembers,"StateMembersAttr":$valueMembers );
    let assemblyFormat = "`<` custom<StateMembers>($keyMembers) `,` custom<StateMembers>($valueMembers) `>`";
     let extraClassDeclaration = [{
        StateMembersAttr getMembers();
     }];
}

def SegmentTreeView : SubOperator_Type<"SegmentTreeView", "segment_tree_view", [State,LookupAbleState]> {
    let summary = "segment tree view type";
    let parameters = (ins "StateMembersAttr":$keyMembers,"StateMembersAttr":$valueMembers );
    let assemblyFormat = "`<` custom<StateMembers>($keyMembers) `,` custom<StateMembers>($valueMembers) `>`";
     let extraClassDeclaration = [{
        StateMembersAttr getMembers();
     }];
}

def Entry : SubOperator_Type<"Entry", "entry"> {
    let summary = "entry of some state";
    let parameters = (ins "Type":$t);
    let assemblyFormat = "`<` $t `>`";
}
def GenericEntryRef : SubOperator_Type<"EntryRef", "entry_ref",[StateEntryReference]> {
    let summary = "reference to entry of some state";
    let parameters = (ins "State":$state);
    let assemblyFormat = "`<` $state `>`";
    let extraClassDeclaration = [{
        bool isReadable(){return true;}
        bool isWriteable(){return true;}
        bool isStable(){return false;}
        bool canBeOffset(){return false;}
        StateMembersAttr getMembers() { return getState().getMembers(); }
    }];
}
def TableEntryRef : SubOperator_Type<"TableEntryRef", "table_entry_ref",[StateEntryReference]> {
    let summary = "reference to entry of table";
    let parameters = (ins "StateMembersAttr": $tableColumns);
    let assemblyFormat = "`<` custom<StateMembers>($tableColumns) `>`";
    let extraClassDeclaration = [{
        bool isReadable(){return true;}
        bool isWriteable(){return false;}
        bool isStable(){return false;}
        bool canBeOffset(){return false;}
        StateMembersAttr getMembers() { return getTableColumns(); }
    }];
}
def GenericLookupEntryRef : SubOperator_Type<"LookupEntryRef", "lookup_entry_ref",[StateEntryReference]> {
    let summary = "reference to entry of some state";
    let parameters = (ins "LookupAbleState":$state);
    let assemblyFormat = "`<` $state `>`";
    let extraClassDeclaration = [{
        bool isReadable(){return true;}
        bool isWriteable(){return true;}
        bool isStable(){return false;}
        bool canBeOffset(){return false;}
        StateMembersAttr getMembers(){ return getState().getValueMembers();}
    }];
}
def ContinuousViewEntryRef : SubOperator_Type<"ContinuousViewEntryRef", "continous_view_entry_ref",[StateEntryReference]> {
    let summary = "reference to entry of some state";
    let parameters = (ins "ContinuousViewType":$view);
    let assemblyFormat = "`<` $view `>`";
    let extraClassDeclaration = [{
        bool isReadable(){return true;}
        bool isWriteable(){return true;}
        bool isStable(){return true;}
        bool canBeOffset(){return true;}
        StateMembersAttr getMembers(){ return getView().getMembers();}
    }];
}
def EntryList : SubOperator_Type<"List", "list"> {
    let summary = "list type";
    let parameters = (ins "StateEntryReference":$t);
    let assemblyFormat = "`<` $t `>`";
}

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//
class SubOperator_Op<string mnemonic, list<Trait> traits = []> : Op<SubOperator_Dialect, mnemonic, traits> {}

//===----------------------------------------------------------------------===//
// Management
//===----------------------------------------------------------------------===//

def GetExternalOp : SubOperator_Op<"get_external", [Pure]> {
    let summary = "get externally defined object";

    let arguments = (ins StrAttr: $descr);
    let results = (outs AnyType : $res);
    let assemblyFormat = [{  $descr `:` type($res) attr-dict }];
}

def SetResultOp : SubOperator_Op<"set_result"> {
    let summary = "set result";
    let arguments = (ins I32Attr:$result_id, AnyType : $state);
    let assemblyFormat = [{ $result_id $state `:` type($state)  attr-dict }];
}

//===----------------------------------------------------------------------===//
// Creating Tuple Streams
//===----------------------------------------------------------------------===//

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
def GenerateOp : SubOperator_Op<"generate",[Pure,SubOperator]> {
    let summary = "generate tuples using imperative code";
    let arguments = (ins ArrayAttr:$generated_columns);
    let results = (outs TupleStream : $res);
    let regions = (region SizedRegion<1>:$region);
    let hasCustomAssemblyFormat = 1;
}

def GenerateEmitOp : SubOperator_Op<"generate_emit"> {
    let summary = "set result";
    let arguments = (ins Variadic<AnyType>:$values);
    let assemblyFormat = [{ ($values^ `:` type($values))?  attr-dict }];
}

//===----------------------------------------------------------------------===//
// Stream Manipulation
//===----------------------------------------------------------------------===//

def UnionOp : SubOperator_Op<"union", [Pure,SubOperator]> {
    let summary = "union of two stream";

    let arguments = (ins Variadic<TupleStream> :$streams);
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{  $streams attr-dict }];
}

def MapOp : SubOperator_Op<"map", [Pure,SubOperator,DeclareOpInterfaceMethods<ColumnFoldable>]> {
    let summary = "map operation";

    let arguments = (ins TupleStream:$stream,ArrayAttr:$computed_cols);
    let results = (outs TupleStream:$result);
    let regions = (region SizedRegion<1>:$fn);
    let assemblyFormat = [{  $stream `computes` `:` custom<CustDefArr>($computed_cols) custom<CustRegion>($fn) attr-dict-with-keyword }];
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

def RenamingOp : SubOperator_Op<"rename", [Pure,SubOperator]> {
    let summary = "rename columns";

    let arguments = (ins TupleStream:$stream,ArrayAttr:$columns);
    let results = (outs TupleStream:$result);
    let assemblyFormat = [{ $stream `renamed` `:` custom<CustDefArr>($columns) attr-dict-with-keyword }];
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

//===----------------------------------------------------------------------===//
// State Creation
//===----------------------------------------------------------------------===//

def CreateSimpleStateOp : SubOperator_Op<"create_simple_state", [Pure]> {
    let summary = "create new state";
    let results = (outs SimpleState : $res);
    let regions = (region AnyRegion:$initFn);
    let assemblyFormat = [{ type($res) (`initial` `:` $initFn^)? attr-dict }];
}

def CreateHeapOp : SubOperator_Op<"create_heap", [Pure]> {
    let summary = "create new heap";
    let arguments = (ins ArrayAttr: $sortBy);
    let results = (outs Heap : $res);
    let regions = (region AnyRegion:$initFn);
    let regions = (region SizedRegion<1>:$region);
    let hasCustomAssemblyFormat = 1;
}

def GenericCreateOp : SubOperator_Op<"create", [Pure]> {
    let summary = "create new state";
    let results = (outs AnyType : $res);
    let assemblyFormat = [{ type($res) attr-dict }];
}
def CreateResultTableOp : SubOperator_Op<"create_result_table", [Pure]> {
    let summary = "create new state";
    let arguments = (ins ArrayAttr: $columns);
    let results = (outs ResultTable : $res);
    let assemblyFormat = [{ $columns `->` type($res) attr-dict }];
}

def CreateSortedViewOp : SubOperator_Op<"create_sorted_view", [IsolatedFromAbove,SubOperator]> {
    let summary = "sort vector";
    let description = [{
        Sort a records in a vector. The region yields the order of two arguments as in C++ `std::sort`.
    }];

    let arguments = (ins Buffer:$toSort,ArrayAttr: $sortBy);
    let results = (outs SortedView:$result);
    let regions = (region SizedRegion<1>:$region);
  let hasCustomAssemblyFormat = 1;
      let extraClassDeclaration = [{
          std::vector<std::string> getWrittenMembers();
          std::vector<std::string> getReadMembers();
      }];
}
def CreateHashIndexedView : SubOperator_Op<"create_hash_indexed_view", [SubOperator]> {
    let arguments = (ins Buffer:$source,StrAttr: $hash_member, StrAttr: $link_member);
    let results = (outs HashIndexedView:$result);
    let assemblyFormat = [{ $source `:` type($source) `hash` `(` $hash_member `)` `link` `(` $link_member `)` `->` type($result) attr-dict }];
      let extraClassDeclaration = [{
          std::vector<std::string> getWrittenMembers();
          std::vector<std::string> getReadMembers();
      }];
}
def CreateContinuousView : SubOperator_Op<"create_continuous_view", [SubOperator]> {
    let arguments = (ins AnyType:$source);
    let results = (outs ContinuousView:$result);
    let assemblyFormat = [{ $source `:` type($source)  `->` type($result) attr-dict }];
    let extraClassDeclaration = [{
      std::vector<std::string> getWrittenMembers();
      std::vector<std::string> getReadMembers();
    }];
}

def CreateSegmentTreeView : SubOperator_Op<"create_segment_tree_view",[SubOperator]> {
  let regions = (region SizedRegion<1>:$initial_fn,SizedRegion<1>:$combine_fn);
  let arguments = (ins ContinuousView:$source,ArrayAttr: $relevant_members);
  let results = (outs SegmentTreeView:$result);
  let hasCustomAssemblyFormat = 1;
        let extraClassDeclaration = [{
            std::vector<std::string> getWrittenMembers();
            std::vector<std::string> getReadMembers();
        }];
}
//todo: add subop.create_view

//===----------------------------------------------------------------------===//
// Streams & States
//===----------------------------------------------------------------------===//

def MaterializeOp : SubOperator_Op<"materialize",[SubOperator,DeclareOpInterfaceMethods<ColumnFoldable>]> {
    let summary = "materialize state (sequentially)";

    let arguments = (ins TupleStream : $stream, AnyType : $state, DictionaryAttr : $mapping );
    let assemblyFormat = [{  $stream custom<ColumnStateMapping>($mapping) `,` $state `:` type($state)  attr-dict }];
    let extraClassDeclaration = [{
        std::vector<std::string> getWrittenMembers();
    }];
}

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
def GetBeginReferenceOp : SubOperator_Op<"get_begin_ref",[SubOperator]> {
    let arguments = (ins TupleStream : $stream, AnyType : $state, ColumnDefAttr:$ref);
    let results = (outs TupleStream: $res);

    let assemblyFormat = [{  $stream $state `:` type($state) custom<CustDef>($ref)   attr-dict }];
}
def GetEndReferenceOp : SubOperator_Op<"get_end_ref",[SubOperator]> {
    let arguments = (ins TupleStream : $stream, AnyType : $state, ColumnDefAttr:$ref);
    let results = (outs TupleStream: $res);
    let assemblyFormat = [{  $stream $state `:` type($state) custom<CustDef>($ref)   attr-dict }];
}
def EntriesBetweenOp : SubOperator_Op<"entries_between",[SubOperator]> {
    let arguments = (ins TupleStream : $stream, ColumnRefAttr:$left_ref,ColumnRefAttr:$right_ref, ColumnDefAttr: $between);
    let results = (outs TupleStream: $res);
    let assemblyFormat = [{  $stream custom<CustRef>($left_ref) custom<CustRef>($right_ref)  custom<CustDef>($between) attr-dict }];
}
def ReferenceValidOp : SubOperator_Op<"ref_valid",[SubOperator]> {
    let arguments = (ins TupleStream : $stream, ColumnRefAttr:$ref,ColumnDefAttr: $is_valid);
    let results = (outs TupleStream: $res);
    let assemblyFormat = [{  $stream custom<CustRef>($ref) custom<CustDef>($is_valid) attr-dict }];
}



//===----------------------------------------------------------------------===//
// Streams & References
//===----------------------------------------------------------------------===//


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
    let assemblyFormat = [{  $stream custom<CustRef>($ref) custom<ColumnStateMapping>($mapping)  attr-dict }];
    let extraClassDeclaration = [{
        std::vector<std::string> getWrittenMembers();
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

//===----------------------------------------------------------------------===//
// Utility Operations
//===----------------------------------------------------------------------===//

def GetSingleValOp : SubOperator_Op<"get_single_val", [Pure]> {
    let arguments = (ins TupleStream : $stream, ColumnRefAttr:$column);
    let results = (outs AnyType : $res);
    let assemblyFormat = [{  $stream `[` custom<CustRef>($column) `]` `->` type($res) attr-dict }];
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

//===----------------------------------------------------------------------===//
// Control-Flow
//===----------------------------------------------------------------------===//

def LoopOp : SubOperator_Op<"loop",[SubOperator]> {
  let summary = "loop";
  let regions = (region SizedRegion<1>:$body);
  let arguments = (ins Variadic<AnyType>: $args);
  let results = (outs Variadic<AnyType>:$res);
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    std::vector<std::string> getWrittenMembers();
    std::vector<std::string> getReadMembers();
  }];
}
def LoopContinueOp : SubOperator_Op<"loop_continue",[Terminator]> {
    let arguments = (ins TupleStream:$cond_stream, ColumnRefAttr: $cond_column, Variadic<AnyType>:$values);
    let assemblyFormat = [{ `(` $cond_stream `[` custom<CustRef>($cond_column) `]` `)` ($values^ `:` type($values))?  attr-dict }];
}


//===----------------------------------------------------------------------===//
// Not used in the future (replaced by views) //todo: remove
//===----------------------------------------------------------------------===//

def MaintainOp : SubOperator_Op<"maintain",[SubOperator]> {
      let summary = "maintain";

      let arguments = (ins AnyType : $state, StrAttr: $description );
      let assemblyFormat = [{  $description $state `:` type($state)  attr-dict }];
      let extraClassDeclaration = [{
          std::vector<std::string> getReadMembers();
          std::vector<std::string> getWrittenMembers();
      }];
}

#endif // LINGODB_DIALECT_SubOperator_IR_OPS
