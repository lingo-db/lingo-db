#ifndef GRAPHALG_GRAPH_ALG_TYPES
#define GRAPHALG_GRAPH_ALG_TYPES

include "mlir/IR/AttrTypeBase.td"

include "GraphAlgDialect.td"
include "GraphAlgInterfaces.td"

class GraphAlg_Type<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<GraphAlg_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def  Matrix : GraphAlg_Type<"Matrix", "mat"> {
    let summary = "Matrix with possibly-abstract dimensions";

    let description = [{
        Dense matrix type where all entries are in the given semiring.
        The dimensions of the matrix are defined as `DimAttr`, which allows for
        statically checking that arguments to common linear algebra operations
        are compatible, without knowing what concrete matrices they will be
        applied to.
    }];

    let parameters = (ins
        DimAttr:$rows,
        DimAttr:$cols,
        TypeParameter<"::mlir::Type", "Element semiring">:$semiring);

    let assemblyFormat = [{
        `<` custom<BareAttr>($rows) `x` custom<BareAttr>($cols) `x` $semiring `>`
    }];

    let extraClassDeclaration = [{
        static MatrixType scalarOf(mlir::Type semiring);

        /** Whether the matrix has one row and one column. */
        bool isScalar() const;

        /** Whether the matrix has one row. */
        bool isRowVector() const;

        /** Whether the matrix has one column. */
        bool isColumnVector() const;

        /** Whether the matrix semiring is boolean. */
        bool isBoolean() const;

        std::pair<DimAttr, DimAttr> getDims() const;

        MatrixType asScalar();

        /**
         * Creates a \c MatrixType with the same shape, but a different
         * semiring.
         */
        MatrixType withSemiring(mlir::Type semiring);
    }];

    let genVerifyDecl = 1;
}

def Dim : GraphAlg_Type<"Dim", "dim"> {
    let summary = "Matrix dimension";
}

def TropI64 : GraphAlg_Type<"TropI64", "trop_i64", [
        DeclareTypeInterfaceMethods<SemiringTypeInterface>]> {
    let summary = "64-bit integer + infinity in the tropical min-plus semiring";
}

def TropF64 : GraphAlg_Type<"TropF64", "trop_f64", [
        DeclareTypeInterfaceMethods<SemiringTypeInterface>]> {
    let summary = "64-bit float + infinity in the tropical min-plus semiring";
}

def TropMaxI64 : GraphAlg_Type<"TropMaxI64", "trop_max_i64", [
        DeclareTypeInterfaceMethods<SemiringTypeInterface>]> {
    let summary = "64-bit integer + infinity in the tropical max-plus semiring";
}

// Type Constraints
def ElemType : AnyTypeOf<[I1, I64, F64, TropI64, TropF64]>;

def IsScalar : CPred<"::llvm::cast<::graphalg::MatrixType>($_self).isScalar()">;

class IsSemiringOf<list<Type> allowedTypes> : Concat<
    "[](::mlir::Type semiring) { return ",
    SubstLeaves<"$_self", "semiring", AnyTypeOf<allowedTypes>.predicate>,
    "; }(::llvm::cast<::graphalg::MatrixType>($_self).getSemiring())">;

class MatrixOf<list<Type> allowedTypes, list<Pred> otherPreds = []>
    : ConfinedType<
        Matrix,
        !listconcat([IsSemiringOf<allowedTypes>], otherPreds),
        "Matrix of " # AnyTypeOf<allowedTypes>.summary # " values",
        "::graphalg::MatrixType">;

class ScalarOf<list<Type> allowedTypes>
    : MatrixOf<allowedTypes, [IsScalar]> {
    let summary = "Scalar of " # AnyTypeOf<allowedTypes>.summary;
}

class BuildableScalar<Type type> : ScalarOf<[type]>, BuildableType<
    "graphalg::MatrixType::scalarOf(" # type.builderCall # ")">;

def RowVector : ConfinedType<Matrix, [
    CPred<"::llvm::cast<::graphalg::MatrixType>($_self).isRowVector()">],
    "Row vector",
    "::graphalg::MatrixType">;
def ColVector : ConfinedType<Matrix, [
    CPred<"::llvm::cast<::graphalg::MatrixType>($_self).isColumnVector()">],
    "Column vector",
    "::graphalg::MatrixType">;

def I1Matrix : MatrixOf<[I1]>;

def AnyScalar : ScalarOf<[AnyType]>;
def I1Scalar: BuildableScalar<I1>;
def I64Scalar : BuildableScalar<I64>;
def F64Scalar : BuildableScalar<F64>;
def ScalarNumber : ScalarOf<[I64, F64]>;

def AnySemiring : Type<
    CPred<"::llvm::isa<SemiringTypeInterface>($_self)">,
    "semiring">;

class AllSemiringsMatch<list<string> names> : AllMatchSameOperatorTrait<
    names,
    "llvm::cast<graphalg::MatrixType>($_self.getType()).getSemiring()",
    "semiring">;

class AllDimsMatch<list<string> names> : AllMatchSameOperatorTrait<
    names,
    "llvm::cast<graphalg::MatrixType>($_self.getType()).getDims()",
    "dimensions">;

#endif // GRAPHALG_GRAPH_ALG_TYPES
