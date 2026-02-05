#ifndef GAREL_TUPLE_OPS
#define GAREL_TUPLE_OPS
/**
 * Tuple-level ops in the GARel dialect.
 *
 * See ./GARelOps.td for the relation-level ops.
 */

include "mlir/Interfaces/InferTypeOpInterface.td"

include "GARelDialect.td"
include "GARelTypes.td"

def ExtractOp : GARel_Op<"extract", [InferTypeOpAdaptor]> {
    let summary = "Extract the value of one column from a tuple";

    let arguments = (ins I32Attr:$column, Tuple:$tuple);

    let results = (outs ColumnType:$result);

    let assemblyFormat = [{
        $column $tuple `:` type($tuple) attr-dict
    }];

    let hasVerifier = 1;
}

#endif // GAREL_TUPLE_OPS
