#ifndef GRAPHALG_GRAPH_ALG_ATTR
#define GRAPHALG_GRAPH_ALG_ATTR

include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

include "GraphAlgDialect.td"

def BinaryOp : I64EnumAttr<
    "BinaryOp", "",
    [
        I64EnumAttrCase<"ADD", 0>,
        I64EnumAttrCase<"SUB", 1>,
        I64EnumAttrCase<"MUL", 2>,
        I64EnumAttrCase<"DIV", 3>,
        // Comparison ops
        I64EnumAttrCase<"EQ", 4>,
        I64EnumAttrCase<"NE", 5>,
        I64EnumAttrCase<"LT", 6>,
        I64EnumAttrCase<"GT", 7>,
        I64EnumAttrCase<"LE", 8>,
        I64EnumAttrCase<"GE", 9>,
    ]> {
  let cppNamespace = "::graphalg";
}

class GraphAlg_Attr<string name, string attrMnemonic, list<Trait> traits = []>
        : AttrDef<GraphAlg_Dialect, name, traits> {
    let mnemonic = attrMnemonic;
}

def TropInfAttr : GraphAlg_Attr<"TropInf", "trop_inf", [TypedAttrInterface]> {
    let summary = "An Attribute representing infinity in tropical semirings";

    let parameters = (ins AttributeSelfTypeParameter<"">:$type);

    // Note: an assembly format is required because GraphAlg_Attr sets
    // 'mnemonic'.
    let assemblyFormat = "";

    let genVerifyDecl = 1;
}

// Built-in attribute such as mlir::IntegerAttr are restricted to built-in
// types, so we wrap them to for the tropical semirings.
def TropIntAttr : GraphAlg_Attr<"TropInt", "trop_int", [TypedAttrInterface]> {
    let summary = "Represents integer values in tropical semirings";

    let parameters = (ins AttributeSelfTypeParameter<"">:$type, "mlir::IntegerAttr":$value);

    let assemblyFormat = [{
        `<` $value `>`
    }];

    let genVerifyDecl = 1;
}

def TropFloatAttr : GraphAlg_Attr<"TropFloat", "trop_float", [TypedAttrInterface]> {
    let summary = "Represents floating-point values in tropical semirings";

    let parameters = (ins AttributeSelfTypeParameter<"">:$type, "mlir::FloatAttr":$value);

    let assemblyFormat = [{
        `<` $value `>`
    }];

    let genVerifyDecl = 1;
}

def DimAttr : GraphAlg_Attr<"Dim", "dim"> {
    let summary = "Matrix dimension";

    let description = [{
        Represents the number of rows or columns of a matrix type.

        May be an abstract identifier rather than a concrete integer to allow
        static analysis of graphalg programs without having to give concrete
        matrices for function parameters.
        This is sufficient to verify that matrix operands to linear algebra
        operations have compatible shapes, under the assumption that distinct
        symbolic dimensions represent distinct concrete dimensions.

        Alternatively, it is a concrete unsigned integer number of row or
        columns. The dimension '1' is particularly useful because it is used to
        define scalars and row and column vectors. It can be retrieved using
        `getOne`.

        graphalg IR typically uses only abstract symbols and the '1' concrete
        dimension. It is only after the `SetDimensions` pass that abstract
        symbols are eliminated and replaced with concrete dimensions.
    }];

    let parameters = (ins
            OptionalParameter<"mlir::DistinctAttr">:$abstractId,
            "std::uint64_t":$concreteDim);
    let skipDefaultBuilders = 1;

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        static DimAttr getOne(mlir::MLIRContext* context);
        static DimAttr newAbstract(mlir::MLIRContext* context);
        static DimAttr getConcrete(
                mlir::MLIRContext* context,
                std::uint64_t dim);

        bool isAbstract() const;
        bool isConcrete() const;
        bool isOne() const;

        static DimAttr parseBare(mlir::AsmParser& parser, mlir::Type);
        void printBare(mlir::AsmPrinter& printer) const;
    }];
}

def MatrixAttr : GraphAlg_Attr<"Matrix", "mat"> {
    let summary = "Constant value matrix";

    let parameters = (ins
        AttributeSelfTypeParameter<"">:$type,
        "mlir::ArrayAttr":$elems);

    let assemblyFormat = [{
        `<` $elems `>`
    }];

    let genVerifyDecl = 1;
}

#endif // GRAPHALG_GRAPH_ALG_ATTR
