#ifndef GRAPHALG_GRAPH_ALG_OPS
#define GRAPHALG_GRAPH_ALG_OPS

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

include "GraphAlgAttr.td"
include "GraphAlgDialect.td"
include "GraphAlgTypes.td"

def TransposeOp : Core_Op<"transpose", [
        Pure,
        InferTypeOpAdaptor,
        PropagatesDense]> {
    let summary = "Transposes the input matrix";

    let arguments = (ins Matrix:$input);

    let results = (outs Matrix:$result);

    let assemblyFormat = [{
        $input `:` type($input) attr-dict
    }];

    let hasFolder = 1;
}

def DiagOp : Core_Op<"diag", [Pure, InferTypeOpAdaptor]> {
    let summary = "Create a diagonal matrix";

    let description = [{
        Creates a square diagonal matrix with the elements of the input vector
        on the main diagonal.
        All other entries of the matrix are set to the additive identity.
    }];

    let arguments = (ins AnyTypeOf<[ColVector, RowVector]>:$input);

    let results = (outs Matrix:$result);

    let assemblyFormat = [{
        $input `:` type($input) attr-dict
    }];

    let hasFolder = 1;
}

def ApplyUnaryOp : GraphAlg_Op<"apply_unary", [
        Pure,
        DeclareOpInterfaceMethods<CallOpInterface>,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>,
        AllDimsMatch<["input", "result"]>]> {
    let summary = "Apply a unary function to each element";

    let arguments = (ins FlatSymbolRefAttr:$func, Matrix:$input);

    let builders = [
        OpBuilder<(ins "mlir::func::FuncOp":$func, "mlir::Value":$input)>
    ];

    let results = (outs Matrix:$result);

    let assemblyFormat = [{
        $func $input `:` type($input) `->` type($result) attr-dict
    }];
}

def ApplyBinaryOp : GraphAlg_Op<"apply_binary", [
        Pure,
        DeclareOpInterfaceMethods<CallOpInterface>,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>,
        AllDimsMatch<["lhs", "result"]>]> {
    let summary = "Apply a binary function to each element";

    let arguments = (ins FlatSymbolRefAttr:$func, Matrix:$lhs, AnyScalar:$rhs);

    let builders = [
        OpBuilder<(ins
            "mlir::func::FuncOp":$func,
            "mlir::Value":$lhs,
            "mlir::Value":$rhs)>
    ];

    let results = (outs Matrix:$result);

    let assemblyFormat = [{
        $func $lhs `,` $rhs `:` functional-type(operands, $result) attr-dict
    }];
}

def ApplyElementWiseOp : GraphAlg_Op<"apply_elementwise", [
        Pure,
        DeclareOpInterfaceMethods<CallOpInterface>,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>,
        AllDimsMatch<["lhs", "rhs", "result"]>]> {
    let summary = "Apply a binary function element-wise to two matrices";

    let description = [{
        Applies a binary function element-wise to two matrices with the same
        dimensions. This operation handles the case where an apply operation has
        two inputs with matching dimensions, unlike ApplyBinaryOp which handles
        one matrix and one scalar.
    }];

    let arguments = (ins FlatSymbolRefAttr:$func, Matrix:$lhs, Matrix:$rhs);

    let builders = [
        OpBuilder<(ins
            "mlir::func::FuncOp":$func,
            "mlir::Value":$lhs,
            "mlir::Value":$rhs)>
    ];

    let results = (outs Matrix:$result);

    let assemblyFormat = [{
        $func $lhs `,` $rhs `:` functional-type(operands, $result) attr-dict
    }];
}

def SelectUnaryOp : GraphAlg_Op<"select_unary", [
        Pure,
        SameOperandsAndResultType,
        DeclareOpInterfaceMethods<CallOpInterface>,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Filter elements based on a unary predicate";

    let description = [{
        Evaluates the predicate function for every element of the input.
        If the function returns true, the value of the input at that index is
        preserved in the output. Values that fail the predicate are set to the
        additive identity.
    }];

    let arguments = (ins FlatSymbolRefAttr:$func, Matrix:$input);

    let results = (outs Matrix:$result);

    let assemblyFormat = [{
        $func $input `:` type($input) attr-dict
    }];
}

def SelectBinaryOp : GraphAlg_Op<"select_binary", [
        Pure,
        AllTypesMatch<["lhs", "result"]>,
        DeclareOpInterfaceMethods<CallOpInterface>,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Filter elements based on a binary predicate";

    let description = [{
        Evaluates the predicate function for every element of the input.
        A second scalar argument is passed on as the second value for the
        predicate function, and will be the same for every call.
        If the function returns true, the value of the input at that index is
        preserved in the output. Values that fail the predicate are set to the
        additive identity.
    }];

    let arguments = (ins FlatSymbolRefAttr:$func, Matrix:$lhs, AnyScalar:$rhs);

    let results = (outs Matrix:$result);

    let assemblyFormat = [{
        $func $lhs `,` $rhs `:` type($lhs) `,` type($rhs) attr-dict
    }];
}

def MatMulOp : Core_Op<"mxm", [
        Pure,
        InferTypeOpAdaptor,
        SameOperandsAndResultSemiring,
        PropagatesDense]> {
    let summary = "Matrix Multiplication";

    let arguments = (ins Matrix:$lhs, Matrix:$rhs);

    let results = (outs Matrix:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs `:` type($lhs) `,` type($rhs) attr-dict
    }];

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def VecMatMulOp : GraphAlg_Op<"vxm", [
        Pure,
        InferTypeOpAdaptor,
        SameOperandsAndResultSemiring]> {
    let summary = "Vector-matrix multiplication with implicit transpose";

    let description = [{
        Proper matrix multiplication (as in MatMulOp) requires that the number
        of columns in lhs matches the number of rows in rhs.
        This op deals with the case where we want to multiply an n-by-1 column
        vector with an n-by-n matrix.
        It represents the computation of (lhs.T * rhs).T.
    }];

    let arguments = (ins ColVector:$lhs, Matrix:$rhs);

    let results = (outs Matrix:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs `:` type($lhs) `,` type($rhs) attr-dict
    }];

    let hasVerifier = 1;
}

def ElementWiseOp : GraphAlg_Op<"ewise", [
        Pure,
        AllTypesMatch<["lhs", "rhs"]>,
        AllDimsMatch<["lhs", "rhs", "result"]>,
        InferTypeOpAdaptor]> {
    let summary = "Element-wise operator application";

    let arguments = (ins Matrix:$lhs, BinaryOp:$op, Matrix:$rhs);

    let results = (outs Matrix:$result);

    let assemblyFormat = [{
        $lhs $op $rhs `:` type($lhs) attr-dict
    }];
}

def ReduceOp : Core_Op<"reduce", [
        Pure,
        SameOperandsAndResultSemiring,
        PropagatesDense]> {
    let summary = "Reduction along one or both dimensions";

    let description = [{
        Reduces the dimensionality of the input by aggregating values using the
        addition operator for the input semiring.
        The input matrix is reduced to a row vector, column vector, or scalar
        value, depending on the result type.
        If the rows and columns of the result both match those of the input,
        the input is returned unchanged.
    }];

    let arguments = (ins Matrix:$input);

    let results = (outs Matrix:$result);

    let assemblyFormat = [{
        $input `:` type($input) `->` type($result) attr-dict
    }];

    let hasVerifier = 1;

    let hasFolder = 1;
}

def CastOp : GraphAlg_Op<"cast", [Pure, AllDimsMatch<["input", "result"]>]> {
    let summary = "Cast matrix to a different semiring";

    let arguments = (ins Matrix:$input);

    let results = (outs Matrix:$result);

    let assemblyFormat = [{
        $input `:` type($input) `->` type($result) attr-dict
    }];

    let hasVerifier = 1;
}

def CastDimOp : Core_Op<"cast_dim", [Pure, DenseResult]> {
    let summary = "Cast a matrix dimension to an integer";

    let arguments = (ins DimAttr:$input);

    let results = (outs I64Scalar:$result);

    let assemblyFormat = [{
        custom<BareAttr>($input) attr-dict
    }];

    let hasFolder = 1;
}

def DimOp : GraphAlg_Op<"dim", [Pure]> {
    let summary = "Matrix dimension value";

    let description = [{
        This pseudo-op is used to represent dimension symbols as values.
        It not needed anymore after parsing is completed, after which all
        previous users should refer to DimAttr directly.
    }];

    let arguments = (ins DimAttr:$dim);

    let results = (outs Dim);
}

def NValsOp : GraphAlg_Op<"nvals", [Pure]> {
    let summary = "Number of non-zero elements in the matrix";

    let arguments = (ins Matrix:$input);

    let results = (outs I64Scalar:$result);

    let assemblyFormat = [{
        $input `:` type($input) attr-dict
    }];
}

def LiteralOp : GraphAlg_Op<"literal", [
        ConstantLike,
        Pure,
        InferTypeOpAdaptor]> {
    let summary = "Scalar literal";

    let arguments = (ins TypedAttrInterface:$value);

    let results = (outs AnyScalar:$result);

    let assemblyFormat = [{
        $value attr-dict
    }];

    let hasFolder = 1;
}

def NotOp : GraphAlg_Op<"not", [Pure, SameOperandsAndResultType]> {
    let summary = "Logical not";

    let arguments = (ins I1Scalar:$input);

    let results = (outs I1Scalar:$result);

    let assemblyFormat = [{
        $input attr-dict
    }];
}

def NegOp : GraphAlg_Op<"neg", [Pure, SameOperandsAndResultType]> {
    let summary = "Unary negation";

    let arguments = (ins ScalarNumber:$input);

    let results = (outs ScalarNumber:$result);

    let assemblyFormat = [{
        $input `:` type($input) attr-dict
    }];
}

def MaskOp : GraphAlg_Op<"mask", [
        Pure,
        AllTypesMatch<["base", "input", "result"]>,
        AllDimsMatch<["base", "mask"]>]> {
    let summary = "Masked assignment";

    let description = [{
        Creates a matrix by blending the `base` and `input` matrices together
        according to `mask`.
        For a given position, the value is copied from `input` if the value in
        the mask is `true`, or taken from `base` otherwise.
        If `complement` is set, the mask is inverted first (this is equivalent
        to swapping `base` and `input`).
    }];

    let arguments = (ins
        Matrix:$base,
        I1Matrix:$mask,
        Matrix:$input,
        BoolAttr:$complement);

    let results = (outs Matrix:$result);

    let assemblyFormat = [{
        $base `<` $mask `:` type($mask) `>` `=`
        $input `:` type($input)
        attr-dict
    }];
}

// Not in core according to spec, but very handy to keep around.
// We use it instead of multiplication with the 1-vector.
def BroadcastOp : Core_Op<"broadcast", [
        Pure,
        SameOperandsAndResultSemiring,
        PropagatesDense]> {
    let summary = "Broadcast the input to a matrix of higher rank";

    let arguments = (ins Matrix:$input);

    let results = (outs Matrix:$result);

    let assemblyFormat = [{
        $input `:` type($input) `->` type($result) attr-dict
    }];

    let hasFolder = 1;
    let hasVerifier = 1;
}

// Not core according to spec, but we don't want to unroll in the general case.
def ForConstOp : Core_Op<"for_const", [
        Pure,
        AllTypesMatch<["rangeBegin", "rangeEnd"]>,
        DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getEntrySuccessorOperands"]>]> {
    let summary = "For loop with constant bounds";

    let description = [{
        A loop iterating over the integer range starting at `rangeBegin`
        (inclusive) and ending at `rangeEnd` (exclusive).
        The `body` region is executed once for every value in the integer range
        (that value is passed as the first block argument).
        At the first iteration of the loop, the other block arguments take the
        values of `initArgs`. For subsequent iterations, results from the
        previous iteration (produced by `YieldOp`) are taken instead.
        The `until` region, if present, is executed after `body`, and produces a
        single boolean scalar indicating whether the loop should terminate
        early.

        In more imperative terms, `initArgs` can be seen as the set of variables
        that are updated in the loop body.
        Within the loop body, those variables can be accessed through the block
        arguments, and their updated values are set through `YieldOp`.
        Finally, `results` represents the new state of those variables after the
        loop terminates.
    }];

    let arguments = (ins
        Variadic<Matrix>:$initArgs,
        I64Scalar:$rangeBegin,
        I64Scalar:$rangeEnd);

    let results = (outs Variadic<Matrix>:$results);

    let regions = (region SizedRegion<1>:$body, MaxSizedRegion<1>:$until);

    let assemblyFormat = [{
        `range` `(`
            $rangeBegin `,`
            $rangeEnd
        `)` `:` type($rangeEnd)
        `init` `(` $initArgs `)` `:` type($initArgs) `->` type($results) attr-dict
        `body` $body
        `until` $until
    }];

    let hasRegionVerifier = 1;
}

def ForDimOp : Core_Op<"for_dim", [
        Pure,
        DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getEntrySuccessorOperands"]>]> {
    let summary = "For loop over a matrix dimension";

    let description = [{
        A loop iterating over the half-open range [0..`dim`).

        This op is otherwise equivalent to `ForConstOp`.
    }];

    let arguments = (ins Variadic<Matrix>:$initArgs, DimAttr:$dim);

    let results = (outs Variadic<Matrix>:$results);

    let regions = (region SizedRegion<1>:$body, MaxSizedRegion<1>:$until);

    let assemblyFormat = [{
        `range` `(` custom<BareAttr>($dim) `)`
        `init` `(` $initArgs `)` `:` type($initArgs) `->` type($results) attr-dict
        `body` $body
        `until` $until
    }];

    let hasRegionVerifier = 1;
    let hasCanonicalizer = 1;
}

def YieldOp : Core_Op<"yield", [
        Pure,
        Terminator,
        ParentOneOf<["ForConstOp", "ForDimOp"]>,
        DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>]> {
    let summary = "Yield from a loop body";

    let description = [{
        Terminates a loop body, specifying the values to assign to the loop
        variables.

        It is also used to terminate `until` blocks.
        In this case, it returns a single boolean scalar matrix to indicate
        whether the loop should be terminated.
    }];

    let arguments = (ins Variadic<Matrix>:$inputs);

    let assemblyFormat = [{
        $inputs `:` type($inputs) attr-dict
    }];
}

def PickAnyOp : Core_Op<"pick_any", [
        Pure,
        SameOperandsAndResultType]> {
    let summary = "Picks one value per row";

    let description = [{
        Keeps one non-zero value per row in the input, setting all other entries
        to zero, where zero is the additive identity.
    }];

    let arguments = (ins Matrix:$input);

    let results = (outs Matrix:$result);

    let assemblyFormat = [{
        $input `:` type($input) attr-dict
    }];

    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

// === Out-of-specification ops ===

def TrilOp : Core_Op<"tril", [Pure, SameOperandsAndResultType]> {
    let summary = "Lower triangular";

    let description = [{
        Extracts the lower triangular portion of the input matrix.
        All entries above the main diagonal are set to the additive identity.
    }];

    let arguments = (ins Matrix:$input);

    let results = (outs Matrix:$result);

    let assemblyFormat = [{
        $input `:` type($input) attr-dict
    }];

    let hasFolder = 1;
}

def TriuOp : GraphAlg_Op<"triu", [Pure, SameOperandsAndResultType]> {
    let summary = "Upper triangular";

    let description = [{
        Extracts the upper triangular portion of the input matrix.
        All entries below the main diagonal are set to the additive identity.

        This op is equivalent to transposing the input, applying `TrilOp`, then
        transposing again.
    }];

    let arguments = (ins Matrix:$input);

    let results = (outs Matrix:$result);

    let assemblyFormat = [{
        $input `:` type($input) attr-dict
    }];

    let hasFolder = 1;
}

// === Ops for inlining ===
def ApplyInlineOp : GraphAlg_Op<"apply_inline", [Pure, IsolatedFromAbove]>{
    let summary = "Applies an element-wise function to matrices";

    let description = [{
        The body is executed once for every position `(r, c)` to produce the
        value at `output(r, c)`, with the values of the block arguments set to
        `input[0](r, c), input[1](r, c), ...`.

        This op is used to transform the different `apply_*` and `select_*`
        into a canonical representation that facilitates function inlining.
    }];

    let arguments = (ins Variadic<Matrix>:$inputs);
    let builders = [
        // Note: creates body block and block arguments
        OpBuilder<(ins
                "mlir::ValueRange":$inputs,
                "mlir::Type":$outputSemiring)>,
    ];

    let regions = (region SizedRegion<1>:$body);

    let results = (outs Matrix:$result);

    let assemblyFormat = [{
        $inputs `:` type($inputs) `->` type($result) $body attr-dict
    }];

    let hasVerifier = 1;
    let hasRegionVerifier = 1;
}

def ApplyInlineReturnOp : GraphAlg_Op<"apply_inline.return", [
        Pure,
        Terminator,
        HasParent<"ApplyInlineOp">]> {
    let arguments = (ins AnyScalar:$value);

    let assemblyFormat = [{
        $value `:` type($value) attr-dict
    }];

    // Note: Parent ApplyInlineOp performs validation of the return value.
}

def ApplyOp : Core_Op<"apply", [Pure, IsolatedFromAbove, PropagatesDense]> {
    let arguments = (ins Variadic<Matrix>:$inputs);

    let regions = (region SizedRegion<1>:$body);

    let results = (outs Matrix:$result);

    let assemblyFormat = [{
        ($inputs^ `:` type($inputs))? `->` type($result) $body attr-dict
    }];

    let hasVerifier = 1;
    let hasRegionVerifier = 1;

    let hasFolder = 1;
    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        /** Initialize the body block and block arguments. */
        mlir::Block& createBody();
    }];
}

def ApplyReturnOp : Core_Op<"apply.return", [
        Pure,
        Terminator,
        HasParent<"ApplyOp">]> {
    let arguments = (ins AnySemiring:$value);

    let assemblyFormat = [{
        $value `:` type($value) attr-dict
    }];

    // Note: Parent ApplyInlineOp performs validation of the return value.
}

def ConstantOp : Core_Op<"const", [
        ConstantLike,
        Pure,
        AllTypesMatch<["value", "result"]>,
        DenseResult]> {
    let summary = "Constant scalar value";

    let arguments = (ins TypedAttrInterface:$value);

    let results = (outs AnySemiring:$result);

    let assemblyFormat = [{
        $value attr-dict
    }];

    let hasFolder = 1;
}

def ConstantMatrixOp : Core_Op<"const_mat", [
        ConstantLike,
        Pure,
        DenseResult]> {
    let summary = "Matrix filled with a constant value";

    let description = [{
        Defines a matrix where all elements are filled with the same constant
        value. The `value` property stores the (scalar) value attribute that is
        broadcast to every matrix cell.
    }];

    let arguments = (ins TypedAttrInterface:$value);

    let results = (outs Matrix:$result);

    let assemblyFormat = [{
        $value `->` type($result) attr-dict
    }];

    let hasFolder = 1;

    let hasVerifier = 1;
}

def CastScalarOp : Core_Op<"cast_scalar", [Pure]>{
    let summary = "Cast scalar to a different semiring";

    let arguments = (ins AnySemiring:$input);

    let results = (outs AnySemiring:$result);

    let assemblyFormat = [{
        $input `:` type($input) `->` type($result) attr-dict
    }];

    let hasFolder = 1;
    let hasVerifier = 1;
}

def AddOp : Core_Op<"add", [Pure, SameOperandsAndResultType]> {
    let summary = "Applies the addition operator defined for the semiring";

    let arguments = (ins AnySemiring:$lhs, AnySemiring:$rhs);

    let results = (outs AnySemiring:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs `:` type($lhs) attr-dict
    }];

    let hasFolder = 1;
}

def MulOp : Core_Op<"mul", [Pure, SameOperandsAndResultType]> {
    let summary = "Applies the multiplication operator defined for the semiring";

    let arguments = (ins AnySemiring:$lhs, AnySemiring:$rhs);

    let results = (outs AnySemiring:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs `:` type($lhs) attr-dict
    }];

    let hasFolder = 1;
}

def EqOp : Core_Op<"eq", [Pure, AllTypesMatch<["lhs", "rhs"]>]> {
    let summary = "Scalar equality comparison";

    let arguments = (ins AnySemiring:$lhs, AnySemiring:$rhs);

    let results = (outs I1:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs `:` type($lhs) attr-dict
    }];

    let hasFolder = 1;
}

// Note: In spec but not used in implementation: OneVector.
// Its functionality is subsumed by BroadcastOp.

// =============================================================================
// ===============================  SPARSE OPS  ================================
// =============================================================================

def MakeDenseOp : GraphAlg_Op<"make_dense", [
        Pure,
        IsolatedFromAbove,
        SameOperandsAndResultType,
        DenseResult]> {
    let summary = "Makes the input matrix dense";

    let description = [{
        This op can be inserted before ops that require their input matrices to
        be dense, and may have different semantics if given a sparse matrix.

        This op will be very expensive on systems that operate on large sparse
        matrices, so it should be optimized away if at all possible.

        To this end, the op contains a region that determines if making the
        input dense is strictly necessary. If the body returns additive identity
        unconditionally after canonicalization, the entire op can be removed.
        Block arguments represent free variables for which the value is not
        statically known. The region exists only to check at compile time if
        this folding and/or canonicalization to zero is possible. No runtime
        code should be generated from it. It can be safely dropped in later
        transformations.
    }];

    let arguments = (ins Matrix:$input);

    let regions = (region SizedRegion<1>:$body);

    let results = (outs Matrix:$result);

    let assemblyFormat = [{
        $input `:` type($input) ($body^)? attr-dict
    }];

    let hasFolder = 1;

    let hasRegionVerifier = 1;
}

def MakeDenseReturnOp : GraphAlg_Op<"make_dense.return", [
        Pure,
        Terminator,
        HasParent<"MakeDenseOp">]> {
    let arguments = (ins AnySemiring:$value);

    let assemblyFormat = [{
        $value `:` type($value) attr-dict
    }];
}

def ElementWiseAddOp : GraphAlg_Op<"ewise_add", [
        Pure,
        SameOperandsAndResultType,
        DeclareOpInterfaceMethods<InferDensityInterface>]> {
    let summary = "Element-wise addition of two matrices";

    let description = [{
        This op is equivalent to `ElementWiseOp` with `op = ADD`.

        It is used to handle sparsity more efficiently.
    }];

    let arguments = (ins Matrix:$lhs, Matrix:$rhs);

    let results = (outs Matrix:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs `:` type($rhs) attr-dict
    }];
}

// =============================================================================
// ==========================  SPLIT AGGREGATE OPS  ============================
// =============================================================================

def MatMulJoinOp : GraphAlg_Op<"mxm_join", [
        Pure,
        InferTypeOpAdaptor,
        SameOperandsAndResultSemiring]> {
    let summary = "Matrix multiplication without aggregation";

    let description = [{
        Represents matrix multiplication without aggregating multiple values for
        the same row and column into one. This op on its own therefore does not
        produce a well-formed matrix value in the general case: Its output must
        be aggregated with a `DeferredReduceOp` or similar.

        The aggregation may be omitted if the inner dimension (cols of lhs, rows
        of rhs) is one, in which case the join will never produce more than one
        value per matrix cell.
    }];

    let arguments = (ins Matrix:$lhs, Matrix:$rhs);

    let results = (outs Matrix:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs `:` type($lhs) `,` type($rhs) attr-dict
    }];
}

def UnionOp : GraphAlg_Op<"union", [
        Pure,
        SameOperandsAndResultSemiring]> {
    let summary = "Union of matrices";

    let description = [{
        Combines the cells of the input matrices.

        This may result in a matrix with multiple values per cell, which is not
        well-formed. In this case, it should be followed with a
        `DeferredReduceOp` to perform the aggregation.

        If an input has higher dimensionality than the result, it is implicitly
        flattened.
    }];

    let arguments = (ins Variadic<Matrix>:$inputs);

    let results = (outs Matrix:$result);

    let assemblyFormat = [{
        $inputs `:` type($inputs) `->` type($result) attr-dict
    }];

    let hasFolder = 1;
    let hasVerifier = 1;
}

def DeferredReduceOp : GraphAlg_Op<"deferred_reduce", [
        Pure,
        SameOperandsAndResultSemiring]> {
    let summary = "Deferred aggregation";

    let description = [{
        This op is equivalent to `ReduceOp`, with the exception that it also
        aggregates if the input and output types match (i.e. it will not be
        folded away in this case). It supports input that has multiple values
        per matrix cell, such as produced by `MatMulJoinOp`. The output is
        always a proper matrix with at most one value per cell.
    }];

    let arguments = (ins Variadic<Matrix>:$inputs);

    let results = (outs Matrix:$result);

    let assemblyFormat = [{
        $inputs `:` type($inputs) `->` type($result) attr-dict
    }];

    let hasFolder = 1;
    let hasVerifier = 1;
}

#endif // GRAPHALG_GRAPH_ALG_OPS
