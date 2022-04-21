#ifndef DBInterfaces
#define DBInterfaces
include "mlir/IR/OpBase.td"
def DB_HandleNullInterface : OpInterface<"HandleNullInterface"> {
    let description = [{
        This interface describes operations that can handle `NULL` values.
        An operator that can handle `NULL` can do this either by itself
        (perhaps with special semantics) or can delegate this to the default
        handler wrapping `NULL` values with a conditional (see `EliminateNulls`
        pass).

        As an optimization, some operators may even support working on invalid
        (garbage) values when the value is actually `NULL` (e.g., addition),
        allowing to omit the conditional check for `NULL`.
    }];
    let cppNamespace = "::mlir::db";

    let methods = [
        InterfaceMethod<"Returns true if the operator handles `NULL` itself.",
            "bool", "canHandleNulls", (ins), [{}],
            /*defaultImplementation=*/[{
                return false;
            }]
        >,
        InterfaceMethod<[{
                Returns true if operator can handle invalid (garbage) values if
                the value is actually `NULL`. Only considered if the operator
                doesn't handle `NULL` values itself.
            }],
            "bool", "canHandleInvalidValues"
        >,
    ];
}

#endif// DBInterfaces
