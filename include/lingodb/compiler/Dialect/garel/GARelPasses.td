#ifndef GAREL_PASSES
#define GAREL_PASSES

include "mlir/Pass/PassBase.td"

def GraphAlgToRel : Pass<"graphalg-to-rel", "::mlir::ModuleOp"> {
    let summary = "Convert graphalg Core IR to relational ops";

    let dependentDialects = [
        "garel::GARelDialect",
        "graphalg::GraphAlgDialect",
    ];
}

#endif // GAREL_PASSES
