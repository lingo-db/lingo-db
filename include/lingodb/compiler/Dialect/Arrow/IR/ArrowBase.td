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
	let summary = "arrow array type";
}

def Arrow_ArrayBuilder : Arrow_Type<"ArrayBuilder","arraybuilder"> {
    let summary = "arrow array builder type";
    //todo: should we name this chunked array builder?
}

#endif// LINGODB_DIALECT_Arrow_IR_OPS
