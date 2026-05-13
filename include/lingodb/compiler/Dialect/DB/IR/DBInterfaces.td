#ifndef DBInterfaces
#define DBInterfaces
include "mlir/IR/OpBase.td"

def DB_NeedsNullWrap : OpInterface<"NeedsNullWrap"> {
    let description = [{
        This interface allows operations to indicate that they cannot handle
        `NULL` values themselves and need a wrapper.
    }];
    let cppNamespace = "::lingodb::compiler::dialect::db";

    let methods = [
        InterfaceMethod<"Returns true if the operator cannot `NULL` itself.",
            "bool", "needsNullWrap", (ins), [{}],
            /*defaultImplementation=*/[{
                return true;
            }]
        >,
    ];
}

def DB_MayNeedNullWrap : DeclareOpInterfaceMethods<DB_NeedsNullWrap,["needsNullWrap"]> {}

def DB_SupportsInvalidValues : OpInterface<"SupportsInvalidValues"> {
    let description = [{
        This interface allows operations to indicate that they can safely handle
        invalid (garbage) values (and therefore can be safely executed on the
        invalid value storage of `NULL` values).
    }];
    let cppNamespace = "::lingodb::compiler::dialect::db";

    let methods = [
        InterfaceMethod<"Returns true if the operator supports invalid values.",
            "bool", "supportsInvalidValues", (ins), [{}],
            /*defaultImplementation=*/[{
                return true;
            }]
        >,
    ];
}

def DB_MaySupportInvalidValues : DeclareOpInterfaceMethods<DB_SupportsInvalidValues,["supportsInvalidValues"]> {}

def DB_SupportsNullabilityChange : OpInterface<"SupportsNullabilityChange"> {
    let cppNamespace = "::lingodb::compiler::dialect::db";

    let methods = [
        InterfaceMethod<"Returns new result type .",
            "mlir::Type", "getChangedResultType", (ins)
        >,
    ];
}

def DB_RefCountedOp : OpInterface<"RefCountedOp"> {
    let description = [{
        Operations that need explicit ref-count adjustments during the memory
        management pass. The pass calls these methods to discover operands and
        results that require `add_use` insertion, or replaces the op entirely
        via `rewriteForRefCount`.

        - `getOwnedOperands` returns operands of which this op consumes /
          captures a reference. The pass emits a matching `add_use` for each,
          *before* the op.
        - `getBorrowedResults` returns results that "borrow" an existing
          reference from somewhere else (e.g. loading from a container). The
          pass emits a matching `add_use` for each, *after* the op.
        - `rewriteForRefCount` is an escape hatch for ops that cannot grow
          ref counts in-place (e.g. `arith.select` -> `scf.if`). It returns
          the replacement op, or `nullptr` if no rewrite is needed.
    }];
    let cppNamespace = "::lingodb::compiler::dialect::db";

    let methods = [
        InterfaceMethod<
            "Operands of which this op consumes / captures a reference (need add_use before).",
            "void", "getOwnedOperands",
            (ins "llvm::SmallVectorImpl<mlir::Value>&":$result),
            /*body=*/[{}],
            /*defaultImplementation=*/[{ (void)result; }]
        >,
        InterfaceMethod<
            "Results that borrow an existing reference (need add_use after).",
            "void", "getBorrowedResults",
            (ins "llvm::SmallVectorImpl<mlir::Value>&":$result),
            /*body=*/[{}],
            /*defaultImplementation=*/[{ (void)result; }]
        >,
        InterfaceMethod<
            "Optional in-place rewrite. Returns the replacement op or nullptr.",
            "mlir::Operation*", "rewriteForRefCount",
            (ins "mlir::OpBuilder&":$builder,
                 "llvm::DenseSet<mlir::Value>&":$returnedValues),
            /*body=*/[{}],
            /*defaultImplementation=*/[{ return nullptr; }]
        >,
    ];
}

def DB_ManagedType : TypeInterface<"ManagedType"> {
    let description = [{
        Types that own managed memory and therefore require explicit ref-count
        tracking in the memory management pass (add_use / cleanup_use /
        promote_to_global). Plain inline types (i32, index, ...) do not
        implement this interface.

        The four emit methods are called by the pass to materialize the
        reference-count adjustments for a value of this type. By default they
        emit the high-level `db.memory.{add_use, cleanup_use, promote_to_global}`
        ops; types whose runtime model differs (e.g. CPython objects with
        Py_INCREF / Py_DECREF) override the defaults to emit their own ops.
    }];
    let cppNamespace = "::lingodb::compiler::dialect::db";

    let methods = [
        InterfaceMethod<"Returns true if values of this type require ref-count management.",
            "bool", "needsManagement"
        >,
        InterfaceMethod<"Emit a ref-count increment for this value at the builder's insertion point.",
            "void", "emitAddUse",
            (ins "mlir::OpBuilder&":$builder, "mlir::Location":$loc, "mlir::Value":$value),
            /*body=*/[{}],
            /*defaultImplementation=*/[{
                ::lingodb::compiler::dialect::db::emitDefaultAddUse(builder, loc, value);
            }]
        >,
        InterfaceMethod<"Emit a ref-count cleanup for this value at the builder's insertion point. `elementFn` is the optional per-element cleanup symbol used for list-of-managed-type values; types that don't need it can ignore it.",
            "void", "emitCleanupUse",
            (ins "mlir::OpBuilder&":$builder, "mlir::Location":$loc, "mlir::Value":$value,
                 "mlir::SymbolRefAttr":$elementFn),
            /*body=*/[{}],
            /*defaultImplementation=*/[{
                ::lingodb::compiler::dialect::db::emitDefaultCleanupUse(builder, loc, value, elementFn);
            }]
        >,
        InterfaceMethod<"Promote a value of this type to a query-global lifetime. Returns the new value to use in place of the original.",
            "mlir::Value", "emitPromoteToGlobal",
            (ins "mlir::OpBuilder&":$builder, "mlir::Location":$loc, "mlir::Value":$value),
            /*body=*/[{}],
            /*defaultImplementation=*/[{
                return ::lingodb::compiler::dialect::db::emitDefaultPromoteToGlobal(builder, loc, value);
            }]
        >,
    ];
}

#endif// DBInterfaces
