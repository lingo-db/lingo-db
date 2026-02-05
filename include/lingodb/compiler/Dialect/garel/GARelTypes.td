
#ifndef GAREL_TYPES
#define GAREL_TYPES

include "mlir/IR/AttrTypeBase.td"

include "GARelDialect.td"
include "GARelAttr.td"

class GARel_Type<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<GARel_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def Relation : GARel_Type<"Relation", "rel"> {
    let summary = "A set of tuples";

    let parameters = (ins OptionalArrayRefParameter<"mlir::Type">:$columns);

    let assemblyFormat = [{
        `<` $columns `>`
    }];
}

def Tuple : GARel_Type<"Tuple", "tuple"> {
    let summary = "A single tuple";

    let parameters = (ins OptionalArrayRefParameter<"mlir::Type">:$columns);

    let assemblyFormat = [{
        `<` $columns `>`
    }];
}

def ColumnType : Type<CPred<"::garel::isColumnType($_self)">, "column type">;

#endif // GAREL_TYPES
