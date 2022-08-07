#ifndef OPS
#define OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Dialect/DSA/IR/DSAInterfaces.td"
include "mlir/Dialect/RelAlg/IR/RelAlgInterfaces.td"
include "mlir/Dialect/util/UtilBase.td"

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

class DSA_Op<string mnemonic, list<Trait> traits = []> : Op<DSA_Dialect, mnemonic, traits> {
  let hasCustomAssemblyFormat = 1;

}

class DSA_Type<string name, string typeMnemonic, list<Trait> traits = []> : TypeDef<DSA_Dialect, name,traits> {
    let mnemonic = typeMnemonic;
}



def DSA_GenericIterable : DSA_Type<"GenericIterable","iterable",[CollectionType]> {
    let summary = "generic iterable";
    let parameters = (ins "Type":$elementType,"std::string":$iteratorName);
    let hasCustomAssemblyFormat=1;
}


def DSA_Table : DSA_Type<"Table", "table"> {
    let summary = "materialized table";
    let assemblyFormat = "";
}

def DSA_TableBuilder : DSA_Type<"TableBuilder", "table_builder"> {
    let summary = "table builder";
    let description = [{
        Stateful table builder with a fixed schema. Values to the last row can
        be added using `ds_append` and once all columns are filled, the row can
        be finalized using `next_row`. The final table can be created using
        `finalize`.
    }];

    let parameters = (ins "TupleType":$rowType);
    let assemblyFormat = "`<` $rowType `>`";
}


def DSA_Record: DSA_Type<"Record","record"> {
    let summary = "record";
    let parameters = (ins "TupleType":$rowType);
    let assemblyFormat = "`<` $rowType `>`";
}
def DSA_RecordBatch: DSA_Type<"RecordBatch","record_batch",[CollectionType]> {
    let summary = "record_batch";
    let parameters = (ins "TupleType":$rowType);
    let assemblyFormat = "`<` $rowType `>`";
    let extraClassDeclaration = [{
        mlir::Type getElementType();
    }];
}
def DSA_AggregationHashtable : DSA_Type<"AggregationHashtable","aggr_ht",[CollectionType]> {
    let summary = "aggregation hashtable";
    let parameters = (ins "TupleType":$keyType,"TupleType":$valType);
    let assemblyFormat = "`<` $keyType `,` $valType `>`";
    let extraClassDeclaration = [{
        mlir::Type getElementType();
    }];
}
def DSA_JoinHashtable : DSA_Type<"JoinHashtable","join_ht",[CollectionType]> {
    let summary = "join hashtable";
    let parameters = (ins "TupleType":$keyType,"TupleType":$valType);
    let assemblyFormat = "`<` $keyType `,` $valType `>`";
    let extraClassDeclaration = [{
        mlir::Type getElementType();
    }];
}


def DSA_CreateDS : DSA_Op<"create_ds"> {
    let summary = "create data structure";
    let description = [{
        Create a Vector/JoinHashtable/AggregationHashTable/TableBuilder.
    }];

    let arguments = (ins Optional<AnyType>:$init_val,OptionalAttr<AnyAttr>:$init_attr);
    let results = (outs AnyType:$ds);
    let assemblyFormat = " ( `(` $init_attr^ `)` `->`)? ( $init_val^ `:` type($init_val) `->`)? type($ds) attr-dict";
    let builders = [
        OpBuilder<(ins "mlir::Type":$t), [{
            build($_builder, $_state, t, mlir::Value());
        }]>,
        OpBuilder<(ins "mlir::Type":$t,"mlir::Value":$v), [{
            build($_builder, $_state, t,v, mlir::Attribute());
        }]>,
        OpBuilder<(ins "mlir::Type":$t,"mlir::Attribute":$attr), [{
            build($_builder, $_state, t,mlir::Value(),attr);
        }]>
    ];
}

def DSA_At : DSA_Op<"at", [NoSideEffect]> {
    let summary = "access column in a record";

    let arguments = (ins DSA_Record:$collection,I32Attr:$pos);
    let results = (outs AnyType:$val,Optional<I1>:$valid);
    let assemblyFormat = "$collection  `[` $pos `]` `:` type($collection) `->` type($val) (`,` type($valid)^)? attr-dict";
}

def DSA_NextRow : DSA_Op<"next_row"> {
    let summary = "start next row in a table builder";

    let arguments = (ins DSA_TableBuilder:$builder);
    let assemblyFormat = "$builder `:` type($builder) attr-dict";
}

def DSA_JoinHashtableInsert : DSA_Op<"joinht_insert"> {
    let summary = "insert into a JoinHashtable or AggregationHashTable";

    let arguments = (ins AnyType:$ht, Index:$hash, AnyType: $val);
    let assemblyFormat = "$ht `:` type($ht) `,` $hash `,` $val `:` type($val) attr-dict";
}
def DSA_HashtableInsert : DSA_Op<"ht_insert"> {
    let summary = "insert into a JoinHashtable or AggregationHashTable";

    let arguments = (ins AnyType:$ht, AnyType:$key, Optional<AnyType>: $val);
    let regions = (region AnyRegion:$hash, AnyRegion:$equal, AnyRegion:$reduce);
}
def DSA_HashtableGetRefOrInsert : DSA_Op<"ht_get_ref_or_insert",[IsolatedFromAbove]> {
    let summary = "returns the reference to a hashtable entry matching the key (if none is found, a new one is created)";

    let arguments = (ins AnyType:$ht, Index:$hash, AnyType:$key);
    let results = (outs AnyType:$ref);
    let regions = (region SizedRegion<1>:$equal, SizedRegion<1>:$initial);
}

def DSA_Finalize : DSA_Op<"finalize"> {
    let summary = "finalize TableBuilder or JoinHashtable";

    let arguments = (ins AnyType:$ht);
    let results = (outs Optional<AnyType>:$res);
    let assemblyFormat = "$ht `:` type($ht)  (`->` type($res)^)? attr-dict";
    let builders = [OpBuilder<(ins "mlir::Value":$v), [{
        build($_builder, $_state, mlir::TypeRange(),v);
    }]>];
}

def DSA_Append : DSA_Op<"ds_append"> {
    let summary = "append column to Vector or last table row in a TableBuilder";

    let arguments = (ins AnyType:$ds, AnyType:$val,Optional<I1>:$valid);
    let assemblyFormat = "$ds `:` type($ds) `,` $val `:` type($val) ( `,` $valid^ )? attr-dict";
        let builders = [OpBuilder<(ins "mlir::Value":$v1, "mlir::Value":$v2), [{
        build($_builder, $_state, v1,v2, mlir::Value());
    }]>];
}


def DSA_ScanSource : DSA_Op<"scan_source"> {
    let arguments = (ins StrAttr:$descr);
    let results = (outs DSA_GenericIterable:$iterable);
    let assemblyFormat = "$descr `:` type($iterable) attr-dict";
}

def DSA_Lookup : DSA_Op<"lookup", [NoSideEffect]> {
    let summary = "lookup in a hash table";

    let arguments = (ins AnyType:$collection,AnyType:$key);
    let results = (outs DSA_GenericIterable:$iterable);
    let assemblyFormat = "$collection `:` type($collection) `,` $key `:` type($key) `->` type($iterable) attr-dict";
}


def ForOp : DSA_Op<"for", [SingleBlockImplicitTerminator<"dsa::YieldOp">,
                           RecursiveSideEffects]> {
    let summary = "iterate over collection";
    let description = [{
        Iterate over all elements of a collection. The iteration can be interrupted when the optionally specified flag becomes set.
    }];

    let arguments = (ins AnyType:$collection,Variadic<AnyType>:$initArgs);
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
        unsigned getNumControlOperands() { return 1; }
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

def YieldOp : DSA_Op<"yield", [NoSideEffect, Terminator,
                               ParentOneOf<["ForOp","HashtableInsert","HashtableGetRefOrInsert"]>]> {
    let arguments = (ins Variadic<AnyType>:$results);
    let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
    let assemblyFormat = [{  attr-dict ($results^ `:` type($results))? }];
}


def FreeOp : DSA_Op<"free"> {
    let arguments = (ins AnyType: $val);
    let assemblyFormat = "$val `:` type($val) attr-dict";
}

#endif// OPS
