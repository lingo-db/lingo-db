#ifndef GRAPHALG_TYPE_INTERFACES
#define GRAPHALG_TYPE_INTERFACES

include "mlir/IR/OpBase.td"

def SemiringTypeInterface : TypeInterface<"SemiringTypeInterface"> {
    let cppNamespace = "::graphalg";
    let description = [{
        Indicates that this type is a semiring.

        A semiring is required to have:

        - An addition operation
        - A multiplication operation
        - Identity elements for both operations.

        See also: https://en.wikipedia.org/wiki/Semiring
    }];

    let methods = [
        InterfaceMethod<
            "Get the additive identity",
            "mlir::TypedAttr",
            "addIdentity">,

        InterfaceMethod<
            "Apply the addition operator",
            "mlir::TypedAttr",
            "add",
            (ins "mlir::TypedAttr":$lhs, "mlir::TypedAttr":$rhs)>,

        InterfaceMethod<
            "Get the multiplicative identity",
            "mlir::TypedAttr",
            "mulIdentity">,

        InterfaceMethod<
            "Apply the multiplication operator",
            "mlir::TypedAttr",
            "mul",
            (ins "mlir::TypedAttr":$lhs, "mlir::TypedAttr":$rhs)>,
    ];
}

def InferDensityInterface
        : OpInterface<"InferDensityInterface"> {
    let description = [{
        Infers if matrices are dense or sparse
    }];

    let methods = [
        InterfaceMethod<
            "Infer density of op results",
            "void",
            "inferDensity",
            (ins
                "llvm::ArrayRef<const DensityLattice*>":$operands,
                "llvm::ArrayRef<DensityLattice*>":$results,
                "llvm::MutableArrayRef<mlir::ChangeResult>":$changedResults)>,
    ];
}

def DenseResultTrait : NativeOpTrait<"DenseResult"> {
    let cppNamespace = "::graphalg";
}

def DenseResult : TraitList<[DenseResultTrait, InferDensityInterface]>;

def PropagatesDenseTrait : NativeOpTrait<"PropagatesDense"> {
    let cppNamespace = "::graphalg";
}

def PropagatesDense : TraitList<[PropagatesDenseTrait, InferDensityInterface]>;

def IsCore : NativeOpTrait<"IsCore"> {
    let cppNamespace = "::graphalg";
}

def SameOperandsAndResultSemiring : NativeOpTrait<"SameOperandsAndResultSemiring"> {
    let cppNamespace = "::graphalg";
}

#endif
