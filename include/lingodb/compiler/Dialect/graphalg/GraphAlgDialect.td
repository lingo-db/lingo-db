#ifndef GRAPHALG_GRAPH_ALG_DIALECT
#define GRAPHALG_GRAPH_ALG_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

include "GraphAlgInterfaces.td"

def GraphAlg_Dialect : Dialect {
    let name = "graphalg";
    let cppNamespace = "::graphalg";

    let extraClassDeclaration = [{
    private:
        GraphAlgCast _cast;

        void registerAttributes();
        void registerTypes();

    public:
        bool isCastLegal(mlir::Type from, mlir::Type to) const;

        mlir::TypedAttr castAttribute(
                mlir::TypedAttr attr,
                mlir::Type to) const;
    }];

    let usePropertiesForAttributes = 1;
    let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;

    let hasConstantMaterializer = 1;
}

class GraphAlg_Op<string mnemonic, list<Trait> traits = []> :
        Op<GraphAlg_Dialect, mnemonic, traits>;

/** Base class for ops that are part of graphalg Core. */
class Core_Op<string mnemonic, list<Trait> traits = []> :
        Op<GraphAlg_Dialect, mnemonic, !listconcat([IsCore], traits)>;

#endif // GRAPHALG_GRAPH_ALG_DIALECT
