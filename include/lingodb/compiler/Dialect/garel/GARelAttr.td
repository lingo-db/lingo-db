#ifndef GAREL_ATTR
#define GAREL_ATTR

include "mlir/IR/BuiltinAttributeInterfaces.td"

include "GARelDialect.td"

class GARel_Attr<string name, string attrMnemonic, list<Trait> traits = []>
        : AttrDef<GARel_Dialect, name, traits> {
    let mnemonic = attrMnemonic;
}

def JoinPredicate : GARel_Attr<"JoinPredicate", "join_pred"> {
    let summary = "A binary equality join predicate";

    let parameters = (ins
        "std::int32_t":$lhsRelIdx,
        "ColumnIdx":$lhsColIdx,
        "std::int32_t":$rhsRelIdx,
        "ColumnIdx":$rhsColIdx);

    let assemblyFormat = [{
        `<` $lhsRelIdx `[` $lhsColIdx `]` `=` $rhsRelIdx `[` $rhsColIdx `]` `>`
    }];
}

def JoinPredicates : ArrayOfAttr<
        GARel_Dialect,
        "JoinPredicates",
        "join_preds",
        "JoinPredicateAttr">;

def AggregateFunc : I64EnumAttr<
    "AggregateFunc", "",
    [
        I64EnumAttrCase<"SUM", 0>,
        I64EnumAttrCase<"MIN", 1>,
        I64EnumAttrCase<"MAX", 2>,
        I64EnumAttrCase<"LOR", 3>, /* Logical OR (over i1) */
        I64EnumAttrCase<"ARGMIN", 4>,
    ]
> {
    let cppNamespace = "::garel";
}

// NOTE: assumes an aggregator produces exactly one output column.
def Aggregator : GARel_Attr<"Aggregator", "aggregator"> {
    let summary = "Aggregate function with bound input columns";

    let parameters = (ins
        "AggregateFunc":$func,
        ArrayRefParameter<"ColumnIdx">:$inputs);

    let assemblyFormat = [{
        `<` $func $inputs `>`
    }];

    let genVerifyDecl = 1;

    let extraClassDeclaration = [{
        /** Type for values in the output column. */
        mlir::Type getResultType(::mlir::Type relType);
    }];
}

def Aggregators : ArrayOfAttr<
        GARel_Dialect,
        "Aggregators",
        "aggregators",
        "AggregatorAttr">;

#endif // GAREL_ATTR
