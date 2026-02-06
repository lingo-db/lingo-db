#ifndef GRAPHALG_GRAPH_ALG_PASSES
#define GRAPHALG_GRAPH_ALG_PASSES

include "mlir/Pass/PassBase.td"

def GraphAlgPrepareInline : Pass<"graphalg-prepare-inline", "::mlir::ModuleOp"> {
    let summary = "Prepares the IR for function inlining";

    let dependentDialects = [
        "mlir::func::FuncDialect",
    ];
}

def GraphAlgScalarizeApply : Pass<"graphalg-scalarize-apply", "mlir::func::FuncOp"> {
    let summary = "Transforms ApplyInlineOp into ApplyOp with scalar body arguments";

    let dependentDialects = [
        "mlir::arith::ArithDialect",
    ];
}

def GraphAlgToCore : Pass<"graphalg-to-core", "::mlir::func::FuncOp"> {
    let summary = "Desugar graphalg IR into the Core operations";
}

def GraphAlgExplicateSparsity : Pass<"graphalg-explicate-sparsity", "mlir::func::FuncOp"> {
    let summary = "Makes explicit the sparsity of matrices";
}

def GraphAlgSplitAggregate : Pass<"graphalg-split-aggregate", "mlir::func::FuncOp"> {
    let summary = "Splits aggregation into a dedicated op";
}

def GraphAlgLoopAggregate : Pass<"graphalg-loop-aggregate", "mlir::func::FuncOp"> {
    let summary = "Optimizes passing aggregates across loop iterations";
}

def GraphAlgVerifyDimensions : Pass<"graphalg-verify-dimensions", "mlir::func::FuncOp"> {
    let summary = "Verifies that ops only use abstract dimensions defined in the function parameters";
}

def GraphAlgSetDimensions : Pass<"graphalg-set-dimensions", "::mlir::ModuleOp"> {
    let summary = "Sets the top-level function and concrete dimensions for its parameters";

    let options = [
        Option<"functionName", "func", "std::string", /*default*/"\"\"", "Name of the function to call">,
        ListOption<"argDims", "args", "CallArgumentDimensions", "Dimensions of the call arguments">,
    ];
}

def GraphAlgToRelAlgPass : Pass<"convert-graphalg-to-relalg", "::mlir::ModuleOp"> {
    let summary = "Converts GraphAlg operations to LingoDB RelAlg and DB dialects";
    let description = [{
        Lowers GraphAlg specific operations (Matrix Addition, Deferred Reduce, etc.)
        into relational algebra (Aggregation, Select) and database scalar operations
        compatible with the LingoDB backend.
    }];
}
#endif // GRAPHALG_GRAPH_ALG_PASSES
