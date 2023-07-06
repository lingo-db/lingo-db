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
        let useDefaultTypePrinterParser=1;
}

class DSA_Op<string mnemonic, list<Trait> traits = []> : Op<DSA_Dialect, mnemonic, traits> {
  let hasCustomAssemblyFormat = 1;

}

class DSA_Type<string name, string typeMnemonic, list<Trait> traits = []> : TypeDef<DSA_Dialect, name,traits> {
    let mnemonic = typeMnemonic;
}


def DSA_ResultTable : DSA_Type<"ResultTable", "result_table"> {
    let summary = "result_table";
    let description = [{
        Stateful result_table with a fixed schema. Values to the last row can
        be added using `ds_append` and once all columns are filled, the row can
        be finalized using `next_row`.
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

def DSA_At : DSA_Op<"at", [Pure]> {
    let summary = "access column in a record";

    let arguments = (ins DSA_Record:$collection,I32Attr:$pos);
    let results = (outs AnyType:$val,Optional<I1>:$valid);
    let assemblyFormat = "$collection  `[` $pos `]` `:` type($collection) `->` type($val) (`,` type($valid)^)? attr-dict";
}

def DSA_NextRow : DSA_Op<"next_row"> {
    let summary = "start next row in a table builder";

    let arguments = (ins DSA_ResultTable:$builder);
    let assemblyFormat = "$builder `:` type($builder) attr-dict";
}

def DSA_DownCast : DSA_Op<"down_cast"> {
    let summary = "set result";
    let arguments = (ins AnyType : $state);
    let results = (outs AnyType: $res);
    let assemblyFormat = [{ $state `:` type($state) `->` type($res) attr-dict }];
}
def DSA_SetResultOp : DSA_Op<"set_result"> {
    let summary = "set result";
    let arguments = (ins I32Attr:$result_id, AnyType : $state);
    let assemblyFormat = [{ $result_id $state `:` type($state)  attr-dict }];
}

def DSA_Append : DSA_Op<"ds_append"> {
    let summary = "append column to Vector or last table row in a TableBuilder";

    let arguments = (ins AnyType:$ds, AnyType:$val,Optional<I1>:$valid);
    let assemblyFormat = "$ds `:` type($ds) `,` $val `:` type($val) ( `,` $valid^ )? attr-dict";
        let builders = [OpBuilder<(ins "mlir::Value":$v1, "mlir::Value":$v2), [{
        build($_builder, $_state, v1,v2, mlir::Value());
    }]>];
}



def ForOp : DSA_Op<"for", [SingleBlockImplicitTerminator<"dsa::YieldOp">,
                           RecursiveMemoryEffects]> {
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

def YieldOp : DSA_Op<"yield", [Pure, Terminator,
                               ParentOneOf<["ForOp"]>]> {
    let arguments = (ins Variadic<AnyType>:$results);
    let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
    let assemblyFormat = [{  attr-dict ($results^ `:` type($results))? }];
}


#endif// OPS
