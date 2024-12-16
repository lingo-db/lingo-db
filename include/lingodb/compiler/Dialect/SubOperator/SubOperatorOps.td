#ifndef LINGODB_DIALECT_SubOperator_IR_OPS
#define LINGODB_DIALECT_SubOperator_IR_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "lingodb/compiler/Dialect/TupleStream/TupleStreamBase.td"
include "SubOpInterfaces.td"


//===----------------------------------------------------------------------===//
// SubOperator dialect definition.
//===----------------------------------------------------------------------===//

def SubOperator_Dialect : Dialect {
    let name = "subop";
    let summary = "A dialect around sub-operators that manipulate a tuplestream, but state is managed explicitly";
    let cppNamespace = "::lingodb::compiler::dialect::subop";
        let extraClassDeclaration = [{
            private:
            void registerTypes();
            void registerAttrs();
            MemberManager memberManager;
            public:
            MemberManager& getMemberManager(){
                return memberManager;
            }
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
    let parameters = (ins "mlir::ArrayAttr":$names,"mlir::ArrayAttr":$types);
    let assemblyFormat ="`<` $names `:` $types `>`";
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

class SubOperator_Type<string name, string typeMnemonic, list<Trait> traits = []> : TypeDef<SubOperator_Dialect, name,traits> {
    let mnemonic = typeMnemonic;
}
def ThreadLocal : SubOperator_Type<"ThreadLocal", "thread_local"> {
    let summary = "wrapper around any State that makes it thread local";
    let parameters = (ins "State":$wrapped);
    let assemblyFormat = "`<` $wrapped `>`";
}
def Table : SubOperator_Type<"Table", "table",[State]> {
    let summary = "external table";
    let parameters = (ins "StateMembersAttr":$members);
    let assemblyFormat = "`<` custom<StateMembers>($members) `>`";
}
def LocalTable : SubOperator_Type<"LocalTable", "local_table",[State]> {
    let summary = "local table";
    let parameters = (ins "StateMembersAttr":$members, "mlir::ArrayAttr": $columnNames);
    let assemblyFormat = "`<` custom<StateMembers>($members) `,` $columnNames `>`";
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
            return lingodb::compiler::dialect::subop::StateMembersAttr::get(getContext(), mlir::ArrayAttr::get(getContext(), {}), mlir::ArrayAttr::get(getContext(), {}));
            }
            StateMembersAttr getValueMembers();
        }];
}
def Map : SubOperator_Type<"Map", "map",[State,LookupAbleState]> {
    let summary = "simple state with multiple members";
    let parameters = (ins "StateMembersAttr":$keyMembers,"StateMembersAttr":$valueMembers );
    let assemblyFormat = "`<` custom<StateMembers>($keyMembers) `,` custom<StateMembers>($valueMembers) `>`";
    let extraClassDeclaration = [{
        StateMembersAttr getMembers();
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
def PreAggrHtFragment : SubOperator_Type<"PreAggrHtFragment", "optimistic_ht_fragment",[State,LookupAbleState]> {
    let parameters = (ins "StateMembersAttr":$keyMembers,"StateMembersAttr":$valueMembers );
    let assemblyFormat = "`<` custom<StateMembers>($keyMembers) `,` custom<StateMembers>($valueMembers) `>`";
    let extraClassDeclaration = [{
        StateMembersAttr getMembers();
    }];
}
def PreAggrHt : SubOperator_Type<"PreAggrHt", "optimistic_ht",[State,LookupAbleState]> {
    let parameters = (ins "StateMembersAttr":$keyMembers,"StateMembersAttr":$valueMembers );
    let assemblyFormat = "`<` custom<StateMembers>($keyMembers) `,` custom<StateMembers>($valueMembers) `>`";
    let extraClassDeclaration = [{
        StateMembersAttr getMembers();
    }];
}
def HashMultimap : SubOperator_Type<"HashMultiMap", "hashmultimap",[State,LookupAbleState]> {
    let summary = "simple state with multiple members";
    let parameters = (ins "StateMembersAttr":$keyMembers,"StateMembersAttr":$valueMembers );
    let assemblyFormat = "`<` custom<StateMembers>($keyMembers) `,` custom<StateMembers>($valueMembers) `>`";
    let extraClassDeclaration = [{
        StateMembersAttr getMembers();
    }];
}
def MultiMap : SubOperator_Type<"MultiMap", "multimap",[State,LookupAbleState]> {
    let summary = "multi map: key -> [values]";
    let parameters = (ins "StateMembersAttr":$keyMembers,"StateMembersAttr":$valueMembers );
    let assemblyFormat = "`<` custom<StateMembers>($keyMembers) `,` custom<StateMembers>($valueMembers) `>`";
    let extraClassDeclaration = [{
        StateMembersAttr getMembers();
    }];
}
def ExternalHashIndex : SubOperator_Type<"ExternalHashIndex", "externalhashindex",[State,LookupAbleState]> {
    let summary = "external hash index: key -> [values]";
    let parameters = (ins "StateMembersAttr": $keyMembers, "StateMembersAttr": $valueMembers);
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

def Array : SubOperator_Type<"Array", "array", [State]> {
    let summary = "fixed sized array";
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
    let parameters = (ins "mlir::Type":$t);
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
def ContinuousEntryRef : SubOperator_Type<"ContinuousEntryRef", "continous_entry_ref",[StateEntryReference]> {
    let summary = "reference to entry of some state";
    let parameters = (ins "State":$state);
    let assemblyFormat = "`<` $state `>`";
    let extraClassDeclaration = [{
        bool isReadable(){return true;}
        bool isWriteable(){return true;}
        bool isStable(){return true;}
        bool canBeOffset(){return true;}
        StateMembersAttr getMembers(){ return getState().getMembers();}
    }];
}
def MultiMapEntryRef : SubOperator_Type<"MultiMapEntryRef", "multi_map_entry_ref",[StateEntryReference]> {
    let summary = "reference to entry of some state";
    let parameters = (ins "MultiMapType":$multi_map);
    let assemblyFormat = "`<` $multi_map `>`";
    let extraClassDeclaration = [{
        bool isReadable(){return true;}
        bool isWriteable(){return true;}
        bool isStable(){return true;}
        bool canBeOffset(){return true;}
        StateMembersAttr getMembers(){ return getMultiMap().getMembers();}
    }];
}
def MapEntryRef : SubOperator_Type<"MapEntryRef", "map_entry_ref",[StateEntryReference]> {
    let summary = "reference to entry of some state";
    let parameters = (ins "MapType":$map);
    let assemblyFormat = "`<` $map `>`";
    let extraClassDeclaration = [{
        bool isReadable(){return true;}
        bool isWriteable(){return true;}
        bool isStable(){return true;}
        bool canBeOffset(){return true;}
        StateMembersAttr getMembers();
    }];
    let extraClassDefinition= [{ StateMembersAttr $cppClass::getMembers(){ return getMap().getMembers();} }];
}
def HashMapEntryRef : SubOperator_Type<"HashMapEntryRef", "hash_map_entry_ref",[StateEntryReference]> {
    let summary = "reference to entry of some state";
    let parameters = (ins "HashMapType":$hash_map);
    let assemblyFormat = "`<` $hash_map `>`";
    let extraClassDeclaration = [{
        bool isReadable(){return true;}
        bool isWriteable(){return true;}
        bool isStable(){return true;}
        bool canBeOffset(){return true;}
        StateMembersAttr getMembers();
    }];
    let extraClassDefinition= [{ StateMembersAttr $cppClass::getMembers(){ return getHashMap().getMembers();} }];
}
def PreAggrHTEntryRef : SubOperator_Type<"PreAggrHTEntryRef", "optimistic_ht_entry_ref",[StateEntryReference]> {
    let summary = "reference to entry of some state";
    let parameters = (ins "PreAggrHtType":$hash_map);
    let assemblyFormat = "`<` $hash_map `>`";
    let extraClassDeclaration = [{
        bool isReadable(){return true;}
        bool isWriteable(){return true;}
        bool isStable(){return true;}
        bool canBeOffset(){return true;}
        StateMembersAttr getMembers();
    }];
    let extraClassDefinition= [{ StateMembersAttr $cppClass::getMembers(){ return getHashMap().getMembers();} }];
}
def ExternalHashIndexEntryRef : SubOperator_Type<"ExternalHashIndexEntryRef", "external_hash_index_entry_ref",[StateEntryReference]> {
    let summary = "reference to entry of some state";
    let parameters = (ins "ExternalHashIndexType":$external_hash_index);
    let assemblyFormat = "`<` $external_hash_index `>`";
    let extraClassDeclaration = [{
       bool isReadable(){return true;}
       bool isWriteable(){return true;}
       bool isStable(){return true;}
       bool canBeOffset(){return true;}
       StateMembersAttr getMembers();
    }];
    let extraClassDefinition= [{ StateMembersAttr $cppClass::getMembers(){ return getExternalHashIndex().getMembers();} }];
}
def HashMultiMapEntryRef : SubOperator_Type<"HashMultiMapEntryRef", "hash_multimap_entry_ref",[StateEntryReference]> {
    let summary = "reference to entry of some state";
    let parameters = (ins "HashMultiMapType":$hash_multimap);
    let assemblyFormat = "`<` $hash_multimap `>`";
    let extraClassDeclaration = [{
        bool isReadable(){return true;}
        bool isWriteable(){return true;}
        bool isStable(){return true;}
        bool canBeOffset(){return true;}
        StateMembersAttr getMembers();
    }];
    let extraClassDefinition= [{ StateMembersAttr $cppClass::getMembers(){ return getHashMultimap().getMembers();} }];
}
def EntryList : SubOperator_Type<"List", "list"> {
    let summary = "list type";
    let parameters = (ins "StateEntryReference":$t);
    let assemblyFormat = "`<` $t `>`";
}
def OptionalType : SubOperator_Type<"Optional", "optional"> {
    let summary = "optional type";
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
def CreateThreadLocalOp : SubOperator_Op<"create_thread_local", [Pure]> {
    let summary = "create new thread local";
    let results = (outs ThreadLocal : $res);
    let regions = (region AnyRegion:$initFn);
    let assemblyFormat = [{ type($res) (`initial` `:` $initFn^)? attr-dict }];
}
def MergeOp : SubOperator_Op<"merge", [Pure,SubOperator]> {
  let summary = "merge thread_local ";
  let arguments = (ins ThreadLocal: $threadLocal);
  let results = (outs AnyType : $res);
  let extraClassDeclaration = [{
    std::vector<std::string> getWrittenMembers();
    std::vector<std::string> getReadMembers();
  }];
  let regions = (region AnyRegion:$combineFn,AnyRegion:$eqFn);
  let assemblyFormat = [{ type($res)  attr-dict }];
  let assemblyFormat = [{ $threadLocal `:` type($threadLocal) `->` type($res) (`combine` `:` $combineFn^)? (`eq` `:` $eqFn^)? attr-dict }];
}
def LockOp : SubOperator_Op<"lock", [SubOperator,DeclareOpInterfaceMethods<StateUsingSubOperator>]> {
  let summary = "merge thread_local ";
  let arguments = (ins TupleStream: $stream, ColumnRefAttr:$ref);
  let results = (outs TupleStream : $res);
  let extraClassDeclaration = [{
    std::vector<std::string> getWrittenMembers();
    std::vector<std::string> getReadMembers();
    mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
  }];
  let regions = (region SizedRegion<1>:$nested);
  let assemblyFormat = [{ $stream `lock` `(` $ref `)` $nested attr-dict }];
}


//===----------------------------------------------------------------------===//
// Creating Tuple Streams
//===----------------------------------------------------------------------===//

def ScanOp : SubOperator_Op<"scan", [Pure,SubOperator,DeclareOpInterfaceMethods<StateUsingSubOperator>]> {
    let summary = "scan state (sequentially)";

    let arguments = (ins AnyType : $state, DictionaryAttr : $mapping );
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{  $state `:` type($state) custom<StateColumnMapping>($mapping) attr-dict }];
    let extraClassDeclaration = [{
        std::vector<std::string> getReadMembers();
            mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);

    }];
}

def ScanListOp : SubOperator_Op<"scan_list", [Pure,SubOperator,ReferenceProducer,DeclareOpInterfaceMethods<StateUsingSubOperator>]> {
    let summary = "scan state (sequentially)";

    let arguments = (ins AnyType : $list, ColumnDefAttr: $elem);
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{  $list `:` type($list) custom<CustDef>($elem) attr-dict }];
    let extraClassDeclaration = [{
        dialect::tuples::ColumnDefAttr getProducedReference(){ return getElem(); }
        mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);

    }];
}

def ScanRefsOp : SubOperator_Op<"scan_refs", [Pure,SubOperator,ReferenceProducer,DeclareOpInterfaceMethods<StateUsingSubOperator>]> {
    let summary = "scan state (sequentially) and emits references to its elements";

    let arguments = (ins AnyType : $state, ColumnDefAttr:$ref);
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{  $state `:` type($state) custom<CustDef>($ref) attr-dict }];
    let extraClassDeclaration = [{
        dialect::tuples::ColumnDefAttr getProducedReference(){ return getRef(); }
        mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
    }];
}
def GenerateOp : SubOperator_Op<"generate",[Pure,SubOperator]> {
    let summary = "generate tuples using imperative code";
    let arguments = (ins ArrayAttr:$generated_columns);
    let results = (outs TupleStream : $res, Variadic<TupleStream>:$streams);
    let regions = (region SizedRegion<1>:$region);
    let hasCustomAssemblyFormat = 1;
    let extraClassDeclaration = [{
        mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
    }];
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

    let arguments = (ins TupleStream:$stream,ArrayAttr:$computed_cols, ArrayAttr:$input_cols);
    let results = (outs TupleStream:$result);
    let regions = (region SizedRegion<1>:$fn);
    let assemblyFormat = [{  $stream `computes` `:` custom<CustDefArr>($computed_cols) `input` `:` custom<CustRefArr>($input_cols) custom<CustRegion>($fn) attr-dict-with-keyword }];
        let extraClassDeclaration = [{
            mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
        }];
}

def FilterSemantic_AllTrue : I64EnumAttrCase<"all_true", 0>;
def FilterSemantic_NoneTrue      : I64EnumAttrCase<"none_true", 1>;

def FilterSemantic : I64EnumAttr<"FilterSemantic", "",
        [FilterSemantic_AllTrue,FilterSemantic_NoneTrue]> {
    let cppNamespace = "::lingodb::compiler::dialect::subop";
}

def FilterOp : SubOperator_Op<"filter", [Pure,SubOperator,DeclareOpInterfaceMethods<ColumnFoldable>]> {
    let summary = "filter stream of tuples according to one or more boolean columns (all 1 => keep tuple, otherwise discard) ";
    let arguments = (ins TupleStream : $stream, FilterSemantic:$filterSemantic,ArrayAttr : $conditions );
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{  $stream $filterSemantic custom<CustRefArr>($conditions)  attr-dict }];
        let extraClassDeclaration = [{
            mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
        }];
}

def RenamingOp : SubOperator_Op<"rename", [Pure,SubOperator]> {
    let summary = "rename columns";

    let arguments = (ins TupleStream:$stream,ArrayAttr:$columns);
    let results = (outs TupleStream:$result);
    let assemblyFormat = [{ $stream `renamed` `:` custom<CustDefArr>($columns) attr-dict-with-keyword }];
	let extraClassDeclaration = [{
		mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
	}];
}

def NestedMapOp : SubOperator_Op<"nested_map",[DeclareOpInterfaceMethods<SubOperator>,DeclareOpInterfaceMethods<StateUsingSubOperator>, ContainsNestedSubOps]> {
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
        mlir::Block* getBody(){ return &getRegion().front();}
       	mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
    }];
}

//===----------------------------------------------------------------------===//
// State Creation
//===----------------------------------------------------------------------===//
def CreateFrom: SubOperator_Op<"create_from", [Pure,SubOperator]> {
    let summary = "create new state from existing state";
    let arguments = (ins ArrayAttr: $columns,State:$state);
    let results = (outs State:$result);
    let assemblyFormat = [{$columns $state `:` type($state) `->` type($result) attr-dict }];
      let extraClassDeclaration = [{
          std::vector<std::string> getWrittenMembers();
          std::vector<std::string> getReadMembers();
      }];
}
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

def GenericCreateOp : SubOperator_Op<"create", [Pure,StateCreator]> {
    let summary = "create new state";
    let results = (outs AnyType : $res);
    let assemblyFormat = [{ type($res) attr-dict }];
      let extraClassDeclaration = [{
          std::vector<std::string> getCreatedMembers();
          }];
}
def CreateArrayOp : SubOperator_Op<"create_array", [Pure,SubOperator]> {
    let summary = "create new state";
    let arguments = (ins SimpleState: $numElements);
    let results = (outs Array : $res);
    let assemblyFormat = [{ $numElements `:` type($numElements) `->` type($res) attr-dict }];
      let hasCustomAssemblyFormat = 1;
          let extraClassDeclaration = [{
              std::vector<std::string> getWrittenMembers();
              std::vector<std::string> getReadMembers();
          }];
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

def MaterializeOp : SubOperator_Op<"materialize",[SubOperator,DeclareOpInterfaceMethods<ColumnFoldable>,DeclareOpInterfaceMethods<StateUsingSubOperator>]> {
    let summary = "materialize state (sequentially)";

    let arguments = (ins TupleStream : $stream, AnyType : $state, DictionaryAttr : $mapping );
    let assemblyFormat = [{  $stream custom<ColumnStateMapping>($mapping) `,` $state `:` type($state)  attr-dict }];
    let extraClassDeclaration = [{
        std::vector<std::string> getWrittenMembers();
		mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
    }];
}

def LookupOrInsertOp : SubOperator_Op<"lookup_or_insert",[SubOperator,ReferenceProducer,DeclareOpInterfaceMethods<StateUsingSubOperator>]> {
    let summary = "performs a lookup in a state and annotates the result as attribute ";
    let arguments = (ins TupleStream: $stream, AnyType: $state, ArrayAttr: $keys,ColumnDefAttr:$ref);
    let results = (outs TupleStream: $res);
    let regions = (region AnyRegion:$eqFn, AnyRegion:$initFn);
      let hasCustomAssemblyFormat = 1;
      let extraClassDeclaration = [{
          std::vector<std::string> getWrittenMembers();
          dialect::tuples::ColumnDefAttr getProducedReference(){ return getRef(); }
          mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
      }];
}
def InsertOp : SubOperator_Op<"insert",[SubOperator,DeclareOpInterfaceMethods<StateUsingSubOperator>,DeclareOpInterfaceMethods<ColumnFoldable>]> {
    let summary = "performs a lookup in a state and annotates the result as attribute ";
    let arguments = (ins TupleStream: $stream, LookupAbleState: $state, DictionaryAttr : $mapping);
    let regions = (region AnyRegion:$eqFn);
      let hasCustomAssemblyFormat = 1;
      let extraClassDeclaration = [{
          std::vector<std::string> getWrittenMembers();
          mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
      }];
}

def LookupOp : SubOperator_Op<"lookup",[SubOperator,DeclareOpInterfaceMethods<StateUsingSubOperator>,ReferenceProducer]> {
    let summary = "performs a lookup in a state and annotates the result as attribute ";
    let arguments = (ins TupleStream: $stream, AnyType: $state, ArrayAttr: $keys,ColumnDefAttr:$ref);
    let results = (outs TupleStream: $res);
    let regions = (region AnyRegion:$eqFn, AnyRegion:$initFn);
      let hasCustomAssemblyFormat = 1;
      let extraClassDeclaration = [{
          std::vector<std::string> getReadMembers();
          dialect::tuples::ColumnDefAttr getProducedReference(){ return getRef(); }
          mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
      }];
}
def GetBeginReferenceOp : SubOperator_Op<"get_begin_ref",[SubOperator]> {
    let arguments = (ins TupleStream : $stream, AnyType : $state, ColumnDefAttr:$ref);
    let results = (outs TupleStream: $res);

    let assemblyFormat = [{  $stream $state `:` type($state) custom<CustDef>($ref)   attr-dict }];
	let extraClassDeclaration = [{
		mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
	}];
}
def GetEndReferenceOp : SubOperator_Op<"get_end_ref",[SubOperator]> {
    let arguments = (ins TupleStream : $stream, AnyType : $state, ColumnDefAttr:$ref);
    let results = (outs TupleStream: $res);
    let assemblyFormat = [{  $stream $state `:` type($state) custom<CustDef>($ref)   attr-dict }];
	let extraClassDeclaration = [{
		mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
	}];
}
def EntriesBetweenOp : SubOperator_Op<"entries_between",[SubOperator]> {
    let arguments = (ins TupleStream : $stream, ColumnRefAttr:$left_ref,ColumnRefAttr:$right_ref, ColumnDefAttr: $between);
    let results = (outs TupleStream: $res);
    let assemblyFormat = [{  $stream custom<CustRef>($left_ref) custom<CustRef>($right_ref)  custom<CustDef>($between) attr-dict }];
	let extraClassDeclaration = [{
		mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
	}];
}
def OffsetReferenceBy : SubOperator_Op<"offset_ref_by",[SubOperator]> {
    let arguments = (ins TupleStream : $stream, ColumnRefAttr:$ref,ColumnRefAttr:$idx, ColumnDefAttr: $newRef);
    let results = (outs TupleStream: $res);
    let assemblyFormat = [{  $stream custom<CustRef>($ref) custom<CustRef>($idx) custom<CustDef>($newRef) attr-dict }];
	let extraClassDeclaration = [{
		mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
	}];
}
def UnwrapOptionalRefOp : SubOperator_Op<"unwrap_optional_ref",[SubOperator,DeclareOpInterfaceMethods<StateUsingSubOperator>]> {
    let arguments = (ins TupleStream : $stream, ColumnRefAttr:$optional_ref, ColumnDefAttr: $ref);
    let results = (outs TupleStream: $res);
    let assemblyFormat = [{  $stream custom<CustRef>($optional_ref) custom<CustDef>($ref) attr-dict }];
	let extraClassDeclaration = [{
		mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
	}];
}



//===----------------------------------------------------------------------===//
// Streams & References
//===----------------------------------------------------------------------===//


def GatherOp : SubOperator_Op<"gather",[SubOperator,DeclareOpInterfaceMethods<StateUsingSubOperator>]> {
    let summary = "scatter";

    let arguments = (ins TupleStream : $stream, ColumnRefAttr: $ref, DictionaryAttr : $mapping );
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{  $stream custom<CustRef>($ref) custom<StateColumnMapping>($mapping)  attr-dict }];
    let extraClassDeclaration = [{
        std::vector<std::string> getReadMembers();
        mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
    }];
}
def ScatterOp : SubOperator_Op<"scatter",[SubOperator,DeclareOpInterfaceMethods<StateUsingSubOperator>]> {
    let summary = "scatter";

    let arguments = (ins TupleStream : $stream, ColumnRefAttr: $ref, DictionaryAttr : $mapping );
    let assemblyFormat = [{  $stream custom<CustRef>($ref) custom<ColumnStateMapping>($mapping)  attr-dict }];
    let extraClassDeclaration = [{
        std::vector<std::string> getWrittenMembers();
        mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
    }];
}

def ReduceOp : SubOperator_Op<"reduce",[SubOperator,DeclareOpInterfaceMethods<StateUsingSubOperator>]> {
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
  let regions = (region SizedRegion<1>:$region,AnyRegion:$combine);
  let arguments = (ins TupleStream: $stream, ColumnRefAttr: $ref, ArrayAttr: $columns, ArrayAttr: $members);
  let hasCustomAssemblyFormat = 1;
        let extraClassDeclaration = [{
            std::vector<std::string> getWrittenMembers();
            std::vector<std::string> getReadMembers();
            mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
        }];
}

//===----------------------------------------------------------------------===//
// Utility Operations
//===----------------------------------------------------------------------===//

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

def LoopOp : SubOperator_Op<"loop",[SubOperator,ContainsNestedSubOps]> {
  let summary = "loop";
  let regions = (region SizedRegion<1>:$bodyRegion);
  let arguments = (ins Variadic<AnyType>: $args);
  let results = (outs Variadic<AnyType>:$res);
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    std::vector<std::string> getWrittenMembers();
    std::vector<std::string> getReadMembers();
            mlir::Block* getBody(){ return &getRegion().front();}

  }];
}
def LoopContinueOp : SubOperator_Op<"loop_continue",[Terminator]> {
    let arguments = (ins AnyType:$cond_state, StrAttr: $cond_member, Variadic<AnyType>:$values);
    let assemblyFormat = [{ `(` $cond_state `:` type($cond_state) `[` $cond_member `]` `)` ($values^ `:` type($values))?  attr-dict }];
}

def SetTrackedCountOp : SubOperator_Op<"set_tracked_count",[SubOperator,DeclareOpInterfaceMethods<StateUsingSubOperator>]> {
   let summary = "set tracked count";
   let arguments = (ins SimpleState:$tupleCount, I32Attr:$resultId, StrAttr:$readState);
   let extraClassDeclaration = [{
      std::vector<std::string> getReadMembers();
   }];
}



def SimpleStateGetScalar : SubOperator_Op<"simple_state_get_scalar",[SubOperator]> {
      let summary = "get scalar value of simple state";

      let arguments = (ins SimpleState : $state, StrAttr: $member );
      let results = (outs AnyType : $res);
      let assemblyFormat = [{ $member $state `:` type($state) `->` type($res) attr-dict }];
      let extraClassDeclaration = [{
          std::vector<std::string> getReadMembers();
      }];
}

def ExecutionGroupOp : SubOperator_Op<"execution_group",[IsolatedFromAbove]> {
    let summary = "execution group";
    let arguments = (ins Variadic<AnyType>:$inputs);
    let results = (outs Variadic<AnyType>:$results);
    let regions = (region SizedRegion<1>:$sub_ops);
    let assemblyFormat = [{ ($inputs^ `:` type($inputs))? custom<CustRegion>($sub_ops)  (`->` type($results)^ )? attr-dict-with-keyword }];
}
def ExecutionGroupReturnOp : SubOperator_Op<"execution_group_return",[Terminator]> {
   let summary = "execution group return";
   let arguments = (ins Variadic<AnyType>:$inputs);
   let assemblyFormat = [{ ($inputs^ `:` type($inputs))? attr-dict-with-keyword }];
}
def ExecutionStepOp : SubOperator_Op<"execution_step",[SubOperator,IsolatedFromAbove,DeclareOpInterfaceMethods<StateUsingSubOperator>]> {
    let summary = "execution step";
    let arguments = (ins Variadic<AnyType>:$inputs, BoolArrayAttr:$is_thread_local);
    let results = (outs Variadic<AnyType>:$results);
    let regions = (region SizedRegion<1>:$sub_ops);
    let assemblyFormat = [{ ($inputs^ `:` type($inputs))? $is_thread_local custom<CustRegion>($sub_ops)  (`->` type($results)^ )? attr-dict-with-keyword }];
}
def ExecutionStepReturnOp : SubOperator_Op<"execution_step_return",[Terminator]> {
   let summary = "execution step return";
   let arguments = (ins Variadic<AnyType>:$inputs);
   let assemblyFormat = [{ ($inputs^ `:` type($inputs))? attr-dict-with-keyword }];
}
def NestedExecutionGroupOp : SubOperator_Op<"nested_execution_group",[IsolatedFromAbove]> {
    let summary = "nested execution group";
    let arguments = (ins Variadic<AnyType>:$inputs);
    let results = (outs Variadic<AnyType>:$results);
    let regions = (region SizedRegion<1>:$sub_ops);
    let assemblyFormat = [{ ($inputs^ `:` type($inputs))? custom<CustRegion>($sub_ops)  (`->` type($results)^ )? attr-dict-with-keyword }];
}
def NestedExecutionGroupReturnOp : SubOperator_Op<"nested_execution_group_return",[Terminator]> {
   let summary = "nested execution group return";
   let arguments = (ins Variadic<AnyType>:$inputs);
   let assemblyFormat = [{ ($inputs^ `:` type($inputs))? attr-dict-with-keyword }];
}

#endif // LINGODB_DIALECT_SubOperator_IR_OPS
