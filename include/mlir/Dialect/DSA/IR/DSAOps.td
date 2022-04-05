#ifndef OPS
#define OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Dialect/DSA/IR/DSAInterfaces.td"
include "mlir/Dialect/RelAlg/IR/RelAlgInterfaces.td"

//===----------------------------------------------------------------------===//
// DSA dialect definition.
//===----------------------------------------------------------------------===//

def DSA_Dialect : Dialect {
let name = "dsa";
let cppNamespace = "::mlir::dsa";
let extraClassDeclaration = [{
private:
void registerTypes();

public:

}];
}

class DSA_Op<string mnemonic, list<Trait> traits = []> :
Op<DSA_Dialect, mnemonic, traits>{
let printer = [{ return ::print(p, *this); }];
let parser = [{ return ::parse$cppClass(parser, result); }];
}



class DSA_Type<string name, string typeMnemonic> : TypeDef<DSA_Dialect, name> {
  let mnemonic = typeMnemonic;
}

class DSA_Collection<string name, string typeMnemonic> : TypeDef<DSA_Dialect, name,[],"::mlir::dsa::CollectionType"> {
    let mnemonic = typeMnemonic;
}

def DSACollection : Type<CPred<"$_self.isa<::mlir::dsa::CollectionType>()">,"DSA dialect collection">;

def DSA_GenericIterable : DSA_Collection<"GenericIterable","iterable"> {
let summary = "generic iterable";
let parameters = (ins "Type":$elementType,"std::string":$iteratorName);

}


def DSA_Table: DSA_Type<"Table","table"> {
let summary = "table";
let assemblyFormat = "";
}

def DSA_Flag: DSA_Type<"Flag","flag"> {
let summary = "flag";
let assemblyFormat = "";
}
def DSA_TableBuilder : DSA_Type<"TableBuilder","table_builder"> {
let summary = "builder for table";
let parameters = (ins "TupleType":$rowType);
let assemblyFormat = "`<` $rowType `>`";
}
def DSA_Vector : DSA_Collection<"Vector","vector"> {
let summary = "vector";
let parameters = (ins "Type":$elementType);
let assemblyFormat = "`<` $elementType `>`";
}
def DSA_RecordBatch: DSA_Collection<"RecordBatch","record_batch"> {
let summary = "record_batch";
let parameters = (ins "TupleType":$rowType);
let assemblyFormat = "`<` $rowType `>`";
}
def DSA_Record: DSA_Collection<"Record","record"> {
let summary = "record";
let parameters = (ins "TupleType":$rowType);
let assemblyFormat = "`<` $rowType `>`";
}
def DSA_At : DSA_Op<"at",
[NoSideEffect]> {
let arguments = (ins AnyType:$collection,I32Attr:$pos);
let results = (outs AnyType:$val,Optional<I1>:$valid);
let assemblyFormat = "$collection  `[` $pos `]` `:` type($collection) `->` type($val) (`,` type($valid)^)? attr-dict";
}
def DSA_AggregationHashtable : DSA_Collection<"AggregationHashtable","aggr_ht"> {
let summary = "aggregation hashtable";
let parameters = (ins "TupleType":$keyType,"TupleType":$valType);
let assemblyFormat = "`<` $keyType `,` $valType `>`";
}
def DSA_JoinHashtable : DSA_Collection<"JoinHashtable","join_ht"> {
let summary = "join hashtable";
let parameters = (ins "TupleType":$keyType,"TupleType":$valType);
let assemblyFormat = "`<` $keyType `,` $valType `>`";
}


def DSA_CreateFlag : DSA_Op<"createflag"> {
let summary = "create flag";
let results = (outs DSA_Flag:$flag);
let assemblyFormat = "attr-dict";
}
def DSA_SetFlag : DSA_Op<"setflag"> {
let summary = "set flag to true";
let arguments = (ins DSA_Flag:$flag, I1:$val);
let assemblyFormat = "$flag `,` $val attr-dict";
}
def DSA_GetFlag : DSA_Op<"getflag"> {
let summary = "get flag value";
let arguments = (ins DSA_Flag:$flag);
let results= (outs I1:$res);
let assemblyFormat = "$flag attr-dict";
}



def DSA_NextRow : DSA_Op<"next_row"> {
let arguments = (ins DSA_TableBuilder:$builder);
let assemblyFormat = "$builder `:` type($builder) attr-dict";

}
def DSA_CreateDS : DSA_Op<"create_ds"> {
    let arguments =(ins Optional<AnyType>:$init_val,OptionalAttr<AnyAttr>:$init_attr);
    let results =(outs AnyType:$ds);
    let assemblyFormat = " ( `(` $init_attr^ `)` `->`)? ( $init_val^ `:` type($init_val) `->`)? type($ds) attr-dict";
    let builders = [OpBuilder<(ins "mlir::Type":$t), [{
    build($_builder, $_state, t, mlir::Value());
    }]>,OpBuilder<(ins "mlir::Type":$t,"mlir::Value":$v), [{
            build($_builder, $_state, t,v, mlir::Attribute());
            }]>,
            OpBuilder<(ins "mlir::Type":$t,"mlir::Attribute":$attr), [{
                            build($_builder, $_state, t,mlir::Value(),attr);
                            }]>];
}

def DSA_HashtableInsert: DSA_Op<"ht_insert"> {
    let arguments = (ins AnyType:$ht, AnyType:$key, Optional<AnyType>: $val);
    let regions = (region AnyRegion:$hash, AnyRegion:$equal, AnyRegion:$reduce);
}

def DSA_Finalize: DSA_Op<"finalize"> {
    let arguments = (ins AnyType:$ht );
    let results = (outs Optional<AnyType>:$res);
    let assemblyFormat = "$ht `:` type($ht)  (`->` type($res)^)? attr-dict";
    let builders = [OpBuilder<(ins "mlir::Value":$v), [{
    build($_builder, $_state, mlir::TypeRange(),v);
    }]>];
}
def DSA_Append : DSA_Op<"ds_append"> {
    let arguments = (ins AnyType:$ds, AnyType:$val,Optional<I1>:$valid);
    let assemblyFormat = "$ds `:` type($ds) `,` $val `:` type($val) ( `,` $valid^ )? attr-dict";
        let builders = [OpBuilder<(ins "mlir::Value":$v1, "mlir::Value":$v2), [{
        build($_builder, $_state, v1,v2, mlir::Value());
    }]>];
}


def DSA_ScanSource : DSA_Op<"scan_source",
[NoSideEffect]> {
let arguments = (ins StrAttr:$descr);
let results = (outs DSA_GenericIterable:$iterable);
let assemblyFormat = "$descr `:` type($iterable) attr-dict";
}
def DSA_Lookup : DSA_Op<"lookup",
[NoSideEffect]> {
let arguments = (ins DSACollection:$collection,AnyType:$key);
let results = (outs DSA_GenericIterable:$iterable);
let assemblyFormat = "$collection `:` type($collection) `,` $key `:` type($key) `->` type($iterable) attr-dict";
}


def DSA_SortOp : DSA_Op<"sort",
[IsolatedFromAbove]> {
let arguments = (ins DSA_Vector:$toSort);
let regions = (region SizedRegion<1>:$region);
}





def ForOp : DSA_Op<"for",
[SingleBlockImplicitTerminator<"dsa::YieldOp">,
RecursiveSideEffects,AttrSizedOperandSegments]> {
let arguments = (ins DSACollection:$collection,Optional<DSA_Flag>:$until,Variadic<AnyType>:$initArgs);
let results = (outs Variadic<AnyType>:$results);
let regions = (region SizedRegion<1>:$region);

let extraClassDeclaration = [{
Value getInductionVar() { return getBody()->getArgument(0); }
Block::BlockArgListType getRegionIterArgs() {
    return getBody()->getArguments().drop_front();
}
Operation::operand_range getIterOperands() {
    return getOperands().drop_front(getNumControlOperands());
}


/// Number of region arguments for loop-carried values
unsigned getNumRegionIterArgs() {
    return getBody()->getNumArguments() - 1;
}
/// Number of operands controlling the loop: lb, ub, step
unsigned getNumControlOperands() { return until()?2:1; }
/// Does the operation hold operands for loop-carried values
bool hasIterOperands() {
    return getOperation()->getNumOperands() > getNumControlOperands();
}
/// Get Number of loop-carried values
unsigned getNumIterOperands() {
    return getOperation()->getNumOperands() - getNumControlOperands();
}


}];

}

def FreeOp : DSA_Op<"free"> {
let arguments = (ins AnyType: $val);
let assemblyFormat = "$val `:` type($val) attr-dict";
}




def YieldOp : DSA_Op<"yield", [NoSideEffect, Terminator,
        ParentOneOf<["ForOp","SortOp","HashtableInsert"]>]> {
let arguments = (ins Variadic<AnyType>:$results);
let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
let assemblyFormat = [{  attr-dict ($results^ `:` type($results))? }];
}
def CondSkipOp : DSA_Op<"cond_skip", [ParentOneOf<["ForOp"]>]> {
let arguments = (ins AnyType:$condition, Variadic<AnyType>:$args);
let assemblyFormat = [{ `(` $condition `:` type($condition) `)` attr-dict ( `(` $args^ `:` type($args)`)`)? }];
}
#endif// OPS
