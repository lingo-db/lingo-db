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

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//
class SubOperator_Op<string mnemonic, list<Trait> traits = []> : Op<SubOperator_Dialect, mnemonic, traits> {}


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
def MapOp : SubOperator_Op<"map", [NoSideEffect]> {
    let summary = "map operation";

    let arguments = (ins TupleStream:$rel,ArrayAttr:$computed_cols);
    let results = (outs TupleStream:$result);
    let regions = (region SizedRegion<1>:$fn);
    let assemblyFormat = [{  $rel `computes` `:` custom<CustDefArr>($computed_cols) custom<CustRegion>($fn) attr-dict-with-keyword }];
}


def FilterOp : SubOperator_Op<"filter", [NoSideEffect]> {
    let summary = "filter stream of tuples according to one or more boolean columns (all 1 => keep tuple, otherwise discard) ";
    let arguments = (ins TupleStream : $stream, ArrayAttr : $conditions );
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{  $stream custom<CustRefArr>($conditions)  attr-dict }];
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
