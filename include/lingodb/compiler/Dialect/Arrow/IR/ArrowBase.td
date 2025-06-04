#ifndef LINGODB_DIALECT_Arrow_IR_OPS
#define LINGODB_DIALECT_Arrow_IR_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"

//===----------------------------------------------------------------------===//
// Arrow dialect definition.
//===----------------------------------------------------------------------===//

def Arrow_Dialect : Dialect {
    let name = "arrow";
    let description = [{
        The Arrow dialect provides types and operations for working with Apache Arrow data structures.
        It includes types for arrays and builders (for chunked arrays), and the necessary operations to load values from arrays, and append values to builders.

        The operation implemented by this dialect work directly on the physical memory layout, and do not have any knowledge about Apache Arrow's logical types.
        For example, dates are loaded as integers, strings are loaded as ptr + len, and so on.
        Dealing with logical types is the responsibility of higher-level dialects.
    }];
    let cppNamespace = "::lingodb::compiler::dialect::arrow";
    let extraClassDeclaration = [{
        private:
        void registerTypes();

        public:

    }];
        let useDefaultTypePrinterParser=1;
}

class Arrow_Type<string name, string typeMnemonic, list<Trait> traits = []> : TypeDef<Arrow_Dialect, name,traits> {
    let mnemonic = typeMnemonic;
}

def Arrow_Array : Arrow_Type<"Array","array"> {
	let summary = "represents an anonymous Apache Arrow array, without knowledge of the type stored by it";
}

def Arrow_ArrayBuilder : Arrow_Type<"ArrayBuilder","builder"> {
    let summary = "represents an anonymous Apache Arrow builder (building a chunked array), without knowledge of the type stored by it";
}

#endif// LINGODB_DIALECT_Arrow_IR_OPS
