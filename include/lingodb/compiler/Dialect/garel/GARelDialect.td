#ifndef GAREL_DIALECT
#define GAREL_DIALECT

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def GARel_Dialect : Dialect {
    let name = "garel";
    let cppNamespace = "::garel";

    let extraClassDeclaration = [{
    private:
        void registerAttributes();
        void registerTypes();
    }];

    let usePropertiesForAttributes = 1;
    let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;
}

// NOTE: GARel ops are always 'Pure'
class GARel_Op<string mnemonic, list<Trait> traits = []> :
        Op<GARel_Dialect, mnemonic, !listconcat(traits, [Pure])>;

#endif // GAREL_DIALECT
