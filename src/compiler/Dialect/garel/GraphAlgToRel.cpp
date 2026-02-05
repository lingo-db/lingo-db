#include <array>
#include <numeric>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "lingodb/compiler/Dialect/garel/GARelAttr.h"
#include "lingodb/compiler/Dialect/garel/GARelDialect.h"
#include "lingodb/compiler/Dialect/garel/GARelOps.h"
#include "lingodb/compiler/Dialect/garel/GARelTypes.h"
#include "lingodb/compiler/Dialect/graphalg/GraphAlgAttr.h"
#include "lingodb/compiler/Dialect/graphalg/GraphAlgCast.h"
#include "lingodb/compiler/Dialect/graphalg/GraphAlgDialect.h"
#include "lingodb/compiler/Dialect/graphalg/GraphAlgOps.h"
#include "lingodb/compiler/Dialect/graphalg/GraphAlgTypes.h"
#include "lingodb/compiler/Dialect/graphalg/SemiringTypes.h"

namespace garel {

#define GEN_PASS_DEF_GRAPHALGTOREL
#include "lingodb/compiler/Dialect/garel/GARelPasses.h.inc"

namespace {

/**
 * Converts all graphalg IR ops into relation ops from the GARel dialect.
 *
 * Matrices, vectors and scalars are converted into relations:
 * - Matrices => (row, column, value) tuples
 * - Vectors => (row, value) tuples
 * - Scalars => a single (value) tuple. For consistency, they are still
 * relations.
 *
 * Top-level operations are converted into relational ops such as \c ProjectOp,
 * \c JoinOp and \c AggregateOp.
 *
 * Scalar operations inside of \c ApplyOp are converted into ops from the arith
 * dialect.
 */
class GraphAlgToRel : public impl::GraphAlgToRelBase<GraphAlgToRel> {
public:
  using impl::GraphAlgToRelBase<GraphAlgToRel>::GraphAlgToRelBase;

  void runOnOperation() final;
};

/** Converts semiring types into their relational equivalents. */
class SemiringTypeConverter : public mlir::TypeConverter {
private:
  static mlir::Type convertSemiringType(graphalg::SemiringTypeInterface type);

public:
  SemiringTypeConverter();
};

/** Converts matrix types into relations. */
class MatrixTypeConverter : public mlir::TypeConverter {
private:
  SemiringTypeConverter _semiringConverter;

  mlir::FunctionType convertFunctionType(mlir::FunctionType type) const;
  RelationType convertMatrixType(graphalg::MatrixType type) const;

public:
  MatrixTypeConverter(mlir::MLIRContext *ctx,
                      const SemiringTypeConverter &semiringConverter);
};

/**
 * Convenient wrapper around a matrix value and its relation equivalent
 * after type conversion.
 *
 * This class is particularly useful for retrieving the relation column for
 * the rows, columns or values of the matrix.
 */
class MatrixAdaptor {
private:
  mlir::TypedValue<graphalg::MatrixType> _matrix;

  RelationType _relType;
  // May be null for outputs, in which case only the relation type is available.
  mlir::TypedValue<RelationType> _relation;

public:
  // For output matrices, where we only have the desired output type.
  MatrixAdaptor(mlir::Value matrix, mlir::Type relType)
      : _matrix(llvm::cast<mlir::TypedValue<graphalg::MatrixType>>(matrix)),
        _relType(llvm::cast<RelationType>(relType)) {}

  // For input matrices, where the OpAdaptor provides the relation value.
  MatrixAdaptor(mlir::Value matrix, mlir::Value relation)
      : MatrixAdaptor(matrix, relation.getType()) {
    this->_relation = llvm::cast<mlir::TypedValue<RelationType>>(relation);
  }

  graphalg::MatrixType matrixType() { return _matrix.getType(); }

  RelationType relType() { return _relType; }

  mlir::TypedValue<RelationType> relation() {
    assert(!!_relation && "No relation value (only type)");
    return _relation;
  }

  auto columns() const { return _relType.getColumns(); }

  bool isScalar() const { return _matrix.getType().isScalar(); }

  bool hasRowColumn() const { return !_matrix.getType().getRows().isOne(); }

  bool hasColColumn() const { return !_matrix.getType().getCols().isOne(); }

  ColumnIdx rowColumn() const {
    assert(hasRowColumn());
    return 0;
  }

  ColumnIdx colColumn() const {
    assert(hasColColumn());
    // Follow row column, if there is one.
    return hasRowColumn() ? 1 : 0;
  }

  ColumnIdx valColumn() const {
    // Last column in the relation.
    return columns().size() - 1;
  }

  graphalg::SemiringTypeInterface semiring() {
    return llvm::cast<graphalg::SemiringTypeInterface>(
        _matrix.getType().getSemiring());
  }
};

template <typename T> class OpConversion : public mlir::OpConversionPattern<T> {
  using mlir::OpConversionPattern<T>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(T op,
                  typename mlir::OpConversionPattern<T>::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

class ApplyOpConversion : public mlir::OpConversionPattern<graphalg::ApplyOp> {
private:
  const SemiringTypeConverter &_bodyArgConverter;

  mlir::LogicalResult
  matchAndRewrite(graphalg::ApplyOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;

public:
  ApplyOpConversion(const SemiringTypeConverter &bodyArgConverter,
                    const MatrixTypeConverter &typeConverter,
                    mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<graphalg::ApplyOp>(typeConverter, ctx),
        _bodyArgConverter(bodyArgConverter) {}
};

struct InputColumnRef {
  std::size_t relIdx;
  ColumnIdx colIdx;
  ColumnIdx outIdx;
};

} // namespace

// =============================================================================
// =============================== Class Methods ===============================
// =============================================================================

mlir::Type
SemiringTypeConverter::convertSemiringType(graphalg::SemiringTypeInterface t) {
  auto *ctx = t.getContext();
  // To i1
  if (t == graphalg::SemiringTypes::forBool(ctx)) {
    return mlir::IntegerType::get(ctx, 1);
  }

  // To i64
  if (t == graphalg::SemiringTypes::forInt(ctx) ||
      t == graphalg::SemiringTypes::forTropInt(ctx) ||
      t == graphalg::SemiringTypes::forTropMaxInt(ctx)) {
    return mlir::IntegerType::get(ctx, 64);
  }

  // To f64
  if (t == graphalg::SemiringTypes::forReal(ctx) ||
      t == graphalg::SemiringTypes::forTropReal(ctx)) {
    return mlir::Float64Type::get(ctx);
  }

  return nullptr;
}

SemiringTypeConverter::SemiringTypeConverter() {
  addConversion(convertSemiringType);
}

mlir::FunctionType
MatrixTypeConverter::convertFunctionType(mlir::FunctionType type) const {
  llvm::SmallVector<mlir::Type> inputs;
  if (mlir::failed(convertTypes(type.getInputs(), inputs))) {
    return {};
  }

  llvm::SmallVector<mlir::Type> results;
  if (mlir::failed(convertTypes(type.getResults(), results))) {
    return {};
  }

  return mlir::FunctionType::get(type.getContext(), inputs, results);
}

RelationType
MatrixTypeConverter::convertMatrixType(graphalg::MatrixType type) const {
  llvm::SmallVector<mlir::Type> columns;
  auto *ctx = type.getContext();
  if (!type.getRows().isOne()) {
    columns.push_back(mlir::IndexType::get(ctx));
  }

  if (!type.getCols().isOne()) {
    columns.push_back(mlir::IndexType::get(ctx));
  }

  auto valueType = _semiringConverter.convertType(type.getSemiring());
  if (!valueType) {
    return {};
  }

  columns.push_back(valueType);
  return RelationType::get(ctx, columns);
}

MatrixTypeConverter::MatrixTypeConverter(
    mlir::MLIRContext *ctx, const SemiringTypeConverter &semiringConverter)
    : _semiringConverter(semiringConverter) {
  addConversion(
      [this](mlir::FunctionType t) { return convertFunctionType(t); });

  addConversion(
      [this](graphalg::MatrixType t) { return convertMatrixType(t); });
}

// =============================================================================
// ============================== Helper Methods ===============================
// =============================================================================

/**
 * Create a relation with all indices for a matrix dimension.
 *
 * Used to broadcast scalar values to a larger matrix.
 */
static RangeOp createDimRead(mlir::Location loc, graphalg::DimAttr dim,
                             mlir::OpBuilder &builder) {
  return builder.create<RangeOp>(loc, dim.getConcreteDim());
}

static void
buildApplyJoinPredicates(mlir::MLIRContext *ctx,
                         llvm::SmallVectorImpl<JoinPredicateAttr> &predicates,
                         llvm::ArrayRef<InputColumnRef> columnsToJoin) {
  if (columnsToJoin.size() < 2) {
    return;
  }

  auto first = columnsToJoin.front();
  for (auto other : columnsToJoin.drop_front()) {
    predicates.push_back(JoinPredicateAttr::get(ctx, first.relIdx, first.colIdx,
                                                other.relIdx, other.colIdx));
  }
}

static mlir::FailureOr<mlir::TypedAttr> convertConstant(mlir::Operation *op,
                                                        mlir::TypedAttr attr) {
  auto *ctx = attr.getContext();
  auto type = attr.getType();
  if (type == graphalg::SemiringTypes::forBool(ctx)) {
    return attr;
  } else if (type == graphalg::SemiringTypes::forInt(ctx)) {
    return attr;
  } else if (type == graphalg::SemiringTypes::forReal(ctx)) {
    return attr;
  } else if (type == graphalg::SemiringTypes::forTropInt(ctx)) {
    std::int64_t value;
    if (llvm::isa<graphalg::TropInfAttr>(attr)) {
      // Positive infinity, kind of.
      value = std::numeric_limits<std::int64_t>::max();
    } else {
      auto intAttr = llvm::cast<graphalg::TropIntAttr>(attr);
      value = intAttr.getValue().getValue().getSExtValue();
    }

    return mlir::TypedAttr(
        mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), value));
  } else if (type == graphalg::SemiringTypes::forTropReal(ctx)) {
    double value;
    if (llvm::isa<graphalg::TropInfAttr>(attr)) {
      // Has a proper positive infinity value
      value = std::numeric_limits<double>::infinity();
    } else {
      auto floatAttr = llvm::cast<graphalg::TropFloatAttr>(attr);
      value = floatAttr.getValue().getValueAsDouble();
    }

    return mlir::TypedAttr(
        mlir::FloatAttr::get(mlir::Float64Type::get(ctx), value));
  } else if (type == graphalg::SemiringTypes::forTropMaxInt(ctx)) {
    std::int64_t value;
    if (llvm::isa<graphalg::TropInfAttr>(attr)) {
      // Negative infinity, kind of.
      value = std::numeric_limits<std::int64_t>::min();
    } else {
      auto intAttr = llvm::cast<graphalg::TropIntAttr>(attr);
      value = intAttr.getValue().getValue().getSExtValue();
    }

    return mlir::TypedAttr(
        mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), value));
  }

  return op->emitOpError("cannot convert constant ") << attr;
}

static bool isTropicalnessCast(graphalg::SemiringTypeInterface inRing,
                               graphalg::SemiringTypeInterface outRing) {
  assert(inRing != outRing && "No-op cast");
  // If the relational types match, it is purely a 'tropicalness' cast such as
  // i64 -> !graphalg.trop_i64.
  SemiringTypeConverter conv;
  return conv.convertType(inRing) == conv.convertType(outRing);
}

static mlir::Value preserveAdditiveIdentity(graphalg::CastScalarOp op,
                                            mlir::Value input,
                                            mlir::Value defaultOutput,
                                            mlir::OpBuilder &builder) {
  // Return defaultOutput, except when input is the additive identity, which
  // we need to remap to the additive identity of the target type.
  auto inRing =
      llvm::cast<graphalg::SemiringTypeInterface>(op.getInput().getType());
  auto outRing = llvm::cast<graphalg::SemiringTypeInterface>(op.getType());

  auto inIdent = convertConstant(op, inRing.addIdentity());
  assert(mlir::succeeded(inIdent));
  auto outIdent = convertConstant(op, outRing.addIdentity());
  assert(mlir::succeeded(outIdent));

  auto inIdentOp =
      builder.create<mlir::arith::ConstantOp>(input.getLoc(), *inIdent);
  auto outIdentOp =
      builder.create<mlir::arith::ConstantOp>(input.getLoc(), *outIdent);

  // Compare input == inIdent
  mlir::Value identCompare;
  if (input.getType().isF64()) {
    assert(inIdentOp.getType().isF64());
    identCompare = builder.create<mlir::arith::CmpFOp>(
        op.getLoc(), mlir::arith::CmpFPredicate::OEQ, input, inIdentOp);
  } else {
    assert(input.getType().isSignlessInteger(64));
    assert(inIdentOp.getType().isSignlessInteger(64));
    identCompare = builder.create<mlir::arith::CmpIOp>(
        op.getLoc(), mlir::arith::CmpIPredicate::eq, input, inIdentOp);
  }

  return builder.create<mlir::arith::SelectOp>(op.getLoc(), identCompare,
                                               outIdentOp, defaultOutput);
}

static mlir::FailureOr<AggregatorAttr>
createAggregator(mlir::Operation *op, graphalg::SemiringTypeInterface sring,
                 ColumnIdx input, mlir::OpBuilder &builder) {
  auto *ctx = builder.getContext();
  AggregateFunc func;
  if (sring == graphalg::SemiringTypes::forBool(ctx)) {
    func = AggregateFunc::LOR;
  } else if (sring.isIntOrFloat()) {
    func = AggregateFunc::SUM;
  } else if (llvm::isa<graphalg::TropI64Type, graphalg::TropF64Type>(sring)) {
    func = AggregateFunc::MIN;
  } else if (llvm::isa<graphalg::TropMaxI64Type>(sring)) {
    func = AggregateFunc::MAX;
  } else {
    return op->emitOpError("aggregation with semiring ")
           << sring << " is not supported";
  }

  std::array<ColumnIdx, 1> inputs{input};
  return AggregatorAttr::get(ctx, func, inputs);
}

static mlir::IntegerAttr tryGetConstantInt(mlir::Value v) {
  mlir::Attribute attr;
  if (!mlir::matchPattern(v, mlir::m_Constant(&attr))) {
    return nullptr;
  }

  return llvm::cast<mlir::IntegerAttr>(attr);
}

static mlir::FailureOr<mlir::Value>
createMul(mlir::Operation *op, graphalg::SemiringTypeInterface sring,
          mlir::Value lhs, mlir::Value rhs, mlir::OpBuilder &builder) {
  auto *ctx = builder.getContext();
  if (sring == graphalg::SemiringTypes::forBool(ctx)) {
    return mlir::Value(
        builder.create<mlir::arith::AndIOp>(op->getLoc(), lhs, rhs));
  } else if (sring == graphalg::SemiringTypes::forInt(ctx)) {
    return mlir::Value(
        builder.create<mlir::arith::MulIOp>(op->getLoc(), lhs, rhs));
  } else if (sring == graphalg::SemiringTypes::forReal(ctx)) {
    return mlir::Value(
        builder.create<mlir::arith::MulFOp>(op->getLoc(), lhs, rhs));
  } else if (sring == graphalg::SemiringTypes::forTropInt(ctx) ||
             sring == graphalg::SemiringTypes::forTropMaxInt(ctx)) {
    return mlir::Value(
        builder.create<mlir::arith::AddIOp>(op->getLoc(), lhs, rhs));
  } else if (sring == graphalg::SemiringTypes::forTropReal(ctx)) {
    return mlir::Value(
        builder.create<mlir::arith::AddFOp>(op->getLoc(), lhs, rhs));
  }

  return op->emitOpError("multiplication with semiring ")
         << sring << " is not supported";
}

// =============================================================================
// =============================== Op Conversion ===============================
// =============================================================================

template <>
mlir::LogicalResult OpConversion<mlir::func::FuncOp>::matchAndRewrite(
    mlir::func::FuncOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto funcType = llvm::cast_if_present<mlir::FunctionType>(
      typeConverter->convertType(op.getFunctionType()));
  if (!funcType) {
    return op->emitOpError("function type ")
           << op.getFunctionType() << " cannot be converted";
  }

  rewriter.modifyOpInPlace(op, [&]() {
    // Update function type.
    op.setFunctionType(funcType);
  });

  // Convert block args.
  mlir::TypeConverter::SignatureConversion newSig(funcType.getNumInputs());
  if (mlir::failed(
          rewriter.convertRegionTypes(&op.getFunctionBody(), *typeConverter))) {
    return mlir::failure();
  }

  return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<mlir::func::ReturnOp>::matchAndRewrite(
    mlir::func::ReturnOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.modifyOpInPlace(op,
                           [&]() { op->setOperands(adaptor.getOperands()); });

  return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<graphalg::TransposeOp>::matchAndRewrite(
    graphalg::TransposeOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  MatrixAdaptor input(op.getInput(), adaptor.getInput());
  MatrixAdaptor output(op, typeConverter->convertType(op.getType()));

  auto projectOp = rewriter.replaceOpWithNewOp<ProjectOp>(op, output.relType(),
                                                          input.relation());

  auto &body = projectOp.createProjectionsBlock();
  rewriter.setInsertionPointToStart(&body);

  llvm::SmallVector<ColumnIdx, 3> columns(input.columns().size());
  std::iota(columns.begin(), columns.end(), 0);
  assert(columns.size() <= 3);
  // Transpose is a no-op if there are fewer than 3 columns.
  if (columns.size() == 3) {
    // Swap row and column
    std::swap(columns[0], columns[1]);
  }

  // Return the input columns (after row and column have been swapped)
  llvm::SmallVector<mlir::Value, 3> results;
  for (auto col : columns) {
    results.push_back(
        rewriter.create<ExtractOp>(op.getLoc(), col, body.getArgument(0)));
  }

  rewriter.create<ProjectReturnOp>(op.getLoc(), results);
  return mlir::success();
}

static constexpr llvm::StringLiteral APPLY_ROW_IDX_ATTR_KEY =
    "garel.apply.row_idx";
static constexpr llvm::StringLiteral APPLY_COL_IDX_ATTR_KEY =
    "garel.apply.col_idx";

mlir::LogicalResult ApplyOpConversion::matchAndRewrite(
    graphalg::ApplyOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  llvm::SmallVector<MatrixAdaptor> inputs;
  for (auto [matrix, relation] :
       llvm::zip_equal(op.getInputs(), adaptor.getInputs())) {
    auto &input = inputs.emplace_back(matrix, relation);
  }

  llvm::SmallVector<mlir::Value> joinChildren;
  llvm::SmallVector<InputColumnRef> rowColumns;
  llvm::SmallVector<InputColumnRef> colColumns;
  llvm::SmallVector<ColumnIdx> valColumns;
  ColumnIdx nextColumnIdx = 0;
  for (const auto &[idx, input] : llvm::enumerate(inputs)) {
    joinChildren.push_back(input.relation());

    if (input.hasRowColumn()) {
      rowColumns.push_back(InputColumnRef{
          .relIdx = idx,
          .colIdx = input.rowColumn(),
          .outIdx = nextColumnIdx + input.rowColumn(),
      });
    }

    if (input.hasColColumn()) {
      colColumns.push_back(InputColumnRef{
          .relIdx = idx,
          .colIdx = input.colColumn(),
          .outIdx = nextColumnIdx + input.colColumn(),
      });
    }

    valColumns.push_back(nextColumnIdx + input.valColumn());
    nextColumnIdx += input.columns().size();
  }

  auto outputType = typeConverter->convertType(op.getType());
  MatrixAdaptor output(op.getResult(), outputType);
  if (rowColumns.empty() && output.hasRowColumn()) {
    // None of the inputs have a row column, but we need it in the output.
    // Broadcast to all rows.
    auto rowsOp =
        createDimRead(op.getLoc(), output.matrixType().getRows(), rewriter);
    joinChildren.push_back(rowsOp);
    rowColumns.push_back(InputColumnRef{
        .relIdx = joinChildren.size() - 1,
        .colIdx = 0,
        .outIdx = nextColumnIdx++,
    });
  }

  if (colColumns.empty() && output.hasColColumn()) {
    // None of the inputs have a col column, but we need it in the output.
    // Broadcast to all columns.
    auto colsOp =
        createDimRead(op.getLoc(), output.matrixType().getCols(), rewriter);
    joinChildren.push_back(colsOp);
    colColumns.push_back(InputColumnRef{
        .relIdx = joinChildren.size() - 1,
        .colIdx = 0,
        .outIdx = nextColumnIdx++,
    });
  }

  mlir::Value joined;
  if (joinChildren.size() == 1) {
    joined = joinChildren.front();
  } else {
    llvm::SmallVector<JoinPredicateAttr> predicates;
    buildApplyJoinPredicates(rewriter.getContext(), predicates, rowColumns);
    buildApplyJoinPredicates(rewriter.getContext(), predicates, colColumns);
    joined = rewriter.create<JoinOp>(op.getLoc(), joinChildren, predicates);
  }

  auto projectOp = rewriter.create<ProjectOp>(op->getLoc(), outputType, joined);

  // Convert old body
  if (mlir::failed(
          rewriter.convertRegionTypes(&op.getBody(), _bodyArgConverter))) {
    return op->emitOpError("failed to convert body argument types");
  }

  // Read value columns, to be used as arg replacements for the old body.
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  auto &body = projectOp.createProjectionsBlock();
  rewriter.setInsertionPointToStart(&body);

  llvm::SmallVector<mlir::Value> columnReads;
  for (auto col : valColumns) {
    columnReads.push_back(
        rewriter.create<ExtractOp>(op->getLoc(), col, body.getArgument(0)));
  }

  // Inline into new body
  rewriter.inlineBlockBefore(&op.getBody().front(), &body, body.end(),
                             columnReads);

  rewriter.replaceOp(op, projectOp);

  // Attach the row and column indexes to the return op.
  auto returnOp = llvm::cast<graphalg::ApplyReturnOp>(body.getTerminator());
  if (!rowColumns.empty()) {
    returnOp->setAttr(APPLY_ROW_IDX_ATTR_KEY,
                      rewriter.getI32IntegerAttr(rowColumns[0].outIdx));
  }

  if (!colColumns.empty()) {
    returnOp->setAttr(APPLY_COL_IDX_ATTR_KEY,
                      rewriter.getI32IntegerAttr(colColumns[0].outIdx));
  }

  return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<graphalg::ApplyReturnOp>::matchAndRewrite(
    graphalg::ApplyReturnOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  llvm::SmallVector<mlir::Value> results;

  // Note: conversion is done top-down, so the ApplyOp is converted to
  // ProjectOp before we reach this op in its body.
  auto inputTuple = op->getBlock()->getArgument(0);

  if (auto idx = op->getAttrOfType<mlir::IntegerAttr>(APPLY_ROW_IDX_ATTR_KEY)) {
    results.push_back(
        rewriter.create<ExtractOp>(op->getLoc(), idx, inputTuple));
  }

  if (auto idx = op->getAttrOfType<mlir::IntegerAttr>(APPLY_COL_IDX_ATTR_KEY)) {
    results.push_back(
        rewriter.create<ExtractOp>(op->getLoc(), idx, inputTuple));
  }

  // The value column
  results.push_back(adaptor.getValue());

  rewriter.replaceOpWithNewOp<ProjectReturnOp>(op, results);
  return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<graphalg::BroadcastOp>::matchAndRewrite(
    graphalg::BroadcastOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  MatrixAdaptor input(op.getInput(), adaptor.getInput());
  MatrixAdaptor output(op, typeConverter->convertType(op.getType()));

  llvm::SmallVector<mlir::Value> joinChildren;
  ColumnIdx currentColIdx = 0;

  std::optional<ColumnIdx> rowColumnIdx;
  std::optional<ColumnIdx> colColumnIdx;

  if (input.hasRowColumn()) {
    // Already have a row column.
    rowColumnIdx = input.rowColumn();
  } else if (output.hasRowColumn()) {
    // Broadcast over all rows.
    joinChildren.push_back(
        createDimRead(op.getLoc(), output.matrixType().getRows(), rewriter));
    rowColumnIdx = currentColIdx++;
  }

  if (input.hasColColumn()) {
    // Already have a col column.
    colColumnIdx = input.colColumn();
  } else if (output.hasColColumn()) {
    // Broadcast over all columns.
    joinChildren.push_back(
        createDimRead(op.getLoc(), output.matrixType().getCols(), rewriter));
    colColumnIdx = currentColIdx++;
  }

  joinChildren.push_back(input.relation());
  auto valColumnIdx = currentColIdx + input.valColumn();

  auto joinOp =
      rewriter.create<JoinOp>(op.getLoc(), joinChildren,
                              // on join predicates (cartesian product)
                              llvm::ArrayRef<JoinPredicateAttr>{});

  // Remap to correctly order as (row, col, val).
  llvm::SmallVector<ColumnIdx, 3> outputColumns;
  if (rowColumnIdx) {
    outputColumns.push_back(*rowColumnIdx);
  }

  if (colColumnIdx) {
    outputColumns.push_back(*colColumnIdx);
  }

  outputColumns.push_back(valColumnIdx);

  // NOTE: folds if the remapping is unncessary.
  auto remapped =
      rewriter.createOrFold<RemapOp>(op.getLoc(), joinOp, outputColumns);
  rewriter.replaceOp(op, remapped);
  return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<graphalg::ConstantMatrixOp>::matchAndRewrite(
    graphalg::ConstantMatrixOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  MatrixAdaptor output(op, typeConverter->convertType(op.getType()));

  auto constantValue = convertConstant(op, op.getValue());
  if (mlir::failed(constantValue)) {
    return mlir::failure();
  }

  auto constantOp = rewriter.create<ConstantOp>(op.getLoc(), *constantValue);

  // Broadcast to rows/columns if needed.
  llvm::SmallVector<mlir::Value> joinChildren;
  if (!output.matrixType().getRows().isOne()) {
    // Broadcast over all rows.
    joinChildren.push_back(
        createDimRead(op.getLoc(), output.matrixType().getRows(), rewriter));
  }

  if (!output.matrixType().getCols().isOne()) {
    // Broadcast over all columns.
    joinChildren.push_back(
        createDimRead(op.getLoc(), output.matrixType().getCols(), rewriter));
  }

  joinChildren.push_back(constantOp);
  auto joinOp = rewriter.createOrFold<JoinOp>(
      op.getLoc(), joinChildren, llvm::ArrayRef<JoinPredicateAttr>{});

  rewriter.replaceOp(op, joinOp);
  return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<graphalg::DeferredReduceOp>::matchAndRewrite(
    graphalg::DeferredReduceOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  MatrixAdaptor input(op.getInputs()[0], adaptor.getInputs()[0]);
  MatrixAdaptor output(op, typeConverter->convertType(op.getType()));

  // Group by keys
  llvm::SmallVector<ColumnIdx, 2> groupBy;
  if (output.hasRowColumn()) {
    groupBy.push_back(input.rowColumn());
  }

  if (output.hasColColumn()) {
    groupBy.push_back(input.colColumn());
  }

  // Aggregators
  auto aggregator =
      createAggregator(op, input.semiring(), input.valColumn(), rewriter);
  if (mlir::failed(aggregator)) {
    return mlir::failure();
  }

  std::array<AggregatorAttr, 1> aggregators{*aggregator};

  // union the inputs and then aggregate.
  auto unionOp =
      rewriter.createOrFold<UnionOp>(op.getLoc(), adaptor.getInputs());
  rewriter.replaceOpWithNewOp<AggregateOp>(op, unionOp, groupBy, aggregators);
  return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<graphalg::DiagOp>::matchAndRewrite(
    graphalg::DiagOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  MatrixAdaptor input(op.getInput(), adaptor.getInput());
  MatrixAdaptor output(op, typeConverter->convertType(op.getType()));

  std::array<ColumnIdx, 3> mapping{0, 0, 1};
  rewriter.replaceOpWithNewOp<RemapOp>(op, adaptor.getInput(), mapping);

  return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<graphalg::ForConstOp>::matchAndRewrite(
    graphalg::ForConstOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto rangeBegin = tryGetConstantInt(op.getRangeBegin());
  auto rangeEnd = tryGetConstantInt(op.getRangeEnd());
  if (!rangeBegin || !rangeEnd) {
    return op->emitOpError("iter range is not constant");
  }

  auto iters = rangeEnd.getInt() - rangeBegin.getInt();

  llvm::SmallVector<mlir::Value> initArgs{adaptor.getRangeBegin()};
  initArgs.append(adaptor.getInitArgs().begin(), adaptor.getInitArgs().end());

  auto blockSignature =
      typeConverter->convertBlockSignature(&op.getBody().front());
  if (!blockSignature) {
    return op->emitOpError("Failed to convert iter args");
  }

  // The relational version of this op can only have a single output value.
  // For loops with multiple results, duplicate.
  llvm::SmallVector<mlir::Value> resultValues;
  for (auto i : llvm::seq(op->getNumResults())) {
    auto result = op->getResult(i);
    if (result.use_empty()) {
      // Not used. Take init arg as a dummy value.
      resultValues.push_back(adaptor.getInitArgs()[i]);
      continue;
    }

    // We are adding the iteration count variable as a first argument, so offset
    // the result index accordingly.
    std::int64_t resultIdx = i + 1;
    auto resultType = adaptor.getInitArgs()[i].getType();
    auto forOp = rewriter.create<ForOp>(op.getLoc(), resultType, initArgs,
                                        iters, resultIdx);
    // body block
    rewriter.cloneRegionBefore(op.getBody(), forOp.getBody(),
                               forOp.getBody().begin());
    rewriter.applySignatureConversion(&forOp.getBody().front(),
                                      *blockSignature);

    // until block
    if (!op.getUntil().empty()) {
      rewriter.cloneRegionBefore(op.getUntil(), forOp.getUntil(),
                                 forOp.getUntil().begin());
      rewriter.applySignatureConversion(&forOp.getUntil().front(),
                                        *blockSignature);
    }

    resultValues.push_back(forOp);
  }

  rewriter.replaceOp(op, resultValues);
  return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<graphalg::YieldOp>::matchAndRewrite(
    graphalg::YieldOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  llvm::SmallVector<mlir::Value> inputs;

  auto *block = op->getBlock();
  auto forOp = llvm::cast<ForOp>(block->getParentOp());
  if (block == &forOp.getBody().front()) {
    // Main body
    auto iterVar = op->getBlock()->getArgument(0);
    // Increment the iteration counter using a garel.project op.
    auto loc = forOp.getLoc();
    auto projOp = rewriter.create<ProjectOp>(loc, iterVar.getType(), iterVar);
    auto &projBlock = projOp.createProjectionsBlock();
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&projBlock);
    auto iterOp = rewriter.create<ExtractOp>(loc, 0, projBlock.getArgument(0));
    auto oneOp = rewriter.create<mlir::arith::ConstantIntOp>(
        loc, 1, rewriter.getI64Type());
    auto addOp = rewriter.create<mlir::arith::AddIOp>(loc, iterOp, oneOp);
    rewriter.create<ProjectReturnOp>(loc, mlir::ValueRange{addOp});

    inputs.push_back(projOp);
  } else {
    // No changes needed for 'until' block.
  }

  inputs.append(adaptor.getInputs().begin(), adaptor.getInputs().end());

  rewriter.replaceOpWithNewOp<ForYieldOp>(op, inputs);
  return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<graphalg::MatMulJoinOp>::matchAndRewrite(
    graphalg::MatMulJoinOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  MatrixAdaptor lhs(op.getLhs(), adaptor.getLhs());
  MatrixAdaptor rhs(op.getRhs(), adaptor.getRhs());
  MatrixAdaptor result(op, typeConverter->convertType(op.getType()));

  // Join matrices.
  llvm::SmallVector<JoinPredicateAttr, 1> predicates;
  if (lhs.hasColColumn() && rhs.hasRowColumn()) {
    predicates.push_back(rewriter.getAttr<JoinPredicateAttr>(
        /*lhsRelIdx=*/0, lhs.colColumn(), /*rhsRelIdx=*/1, rhs.rowColumn()));
  }

  auto joinOp = rewriter.create<JoinOp>(
      op->getLoc(), mlir::ValueRange{lhs.relation(), rhs.relation()},
      predicates);

  // Project the multiplied values.
  auto projOp =
      rewriter.create<ProjectOp>(op.getLoc(), result.relType(), joinOp);
  {
    auto &body = projOp.createProjectionsBlock();
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&body);

    llvm::SmallVector<mlir::Value> projections;

    if (lhs.hasRowColumn()) {
      projections.push_back(rewriter.create<ExtractOp>(
          op.getLoc(), lhs.rowColumn(), body.getArgument(0)));
    }

    if (rhs.hasColColumn()) {
      // In the join output, rhs columns come after lhs columns.
      auto colIdx = lhs.columns().size() + rhs.colColumn();
      projections.push_back(
          rewriter.create<ExtractOp>(op.getLoc(), colIdx, body.getArgument(0)));
    }

    // Get the value columns.
    auto lhsVal = rewriter.create<ExtractOp>(op.getLoc(), lhs.valColumn(),
                                             body.getArgument(0));
    auto rhsVal = rewriter.create<ExtractOp>(
        op.getLoc(), lhs.columns().size() + rhs.valColumn(),
        body.getArgument(0));

    // Perform the multiplication
    auto mulOp = createMul(
        op,
        llvm::cast<graphalg::SemiringTypeInterface>(op.getType().getSemiring()),
        lhsVal, rhsVal, rewriter);
    if (mlir::failed(mulOp)) {
      return mlir::failure();
    }

    projections.push_back(*mulOp);
    rewriter.create<ProjectReturnOp>(op.getLoc(), projections);
  }

  rewriter.replaceOp(op, projOp);
  return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<graphalg::PickAnyOp>::matchAndRewrite(
    graphalg::PickAnyOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  MatrixAdaptor input(op.getInput(), adaptor.getInput());
  MatrixAdaptor output(op, typeConverter->convertType(op.getType()));

  auto *ctx = rewriter.getContext();

  // Remove rows where value is the additive identity.
  auto selectOp = rewriter.create<SelectOp>(op.getLoc(), input.relation());
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    auto &body = selectOp.createPredicatesBlock();
    rewriter.setInsertionPointToStart(&body);

    auto valOp = rewriter.create<ExtractOp>(op.getLoc(), input.valColumn(),
                                            body.getArgument(0));
    auto addIdent = convertConstant(op, input.semiring().addIdentity());
    if (mlir::failed(addIdent)) {
      return mlir::failure();
    }

    auto addIdentOp =
        rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), *addIdent);
    mlir::Value cmpOp;
    if (addIdentOp.getType().isF64()) {
      cmpOp = rewriter.create<mlir::arith::CmpFOp>(
          op.getLoc(), mlir::arith::CmpFPredicate::ONE, valOp, addIdentOp);
    } else {
      cmpOp = rewriter.create<mlir::arith::CmpIOp>(
          op.getLoc(), mlir::arith::CmpIPredicate::ne, valOp, addIdentOp);
    }

    rewriter.create<SelectReturnOp>(op.getLoc(), mlir::ValueRange{cmpOp});
  }

  llvm::SmallVector<ColumnIdx, 1> groupBy;
  if (input.hasRowColumn()) {
    groupBy.push_back(input.rowColumn());
  }

  assert(input.hasColColumn());
  std::array<AggregatorAttr, 2> aggregators{
      // Minimum column
      rewriter.getAttr<AggregatorAttr>(
          AggregateFunc::MIN, std::array<ColumnIdx, 1>{input.colColumn()}),
      // Value for minimum column
      rewriter.getAttr<AggregatorAttr>(
          AggregateFunc::ARGMIN,
          std::array<ColumnIdx, 2>{input.valColumn(), input.colColumn()}),
  };
  rewriter.replaceOpWithNewOp<AggregateOp>(op, selectOp, groupBy, aggregators);
  return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<graphalg::TrilOp>::matchAndRewrite(
    graphalg::TrilOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  MatrixAdaptor input(op.getInput(), adaptor.getInput());

  if (!input.hasRowColumn() || !input.hasColColumn()) {
    return op->emitOpError(
        "only works on full matrices (not scalars or vector)");
  }

  auto selectOp = rewriter.replaceOpWithNewOp<SelectOp>(op, input.relation());

  auto &body = selectOp.createPredicatesBlock();
  rewriter.setInsertionPointToStart(&body);

  auto row = rewriter.create<ExtractOp>(op.getLoc(), input.rowColumn(),
                                        body.getArgument(0));
  auto col = rewriter.create<ExtractOp>(op.getLoc(), input.colColumn(),
                                        body.getArgument(0));
  // col < row
  auto cmpOp = rewriter.create<mlir::arith::CmpIOp>(
      op.getLoc(), mlir::arith::CmpIPredicate::ult, col, row);
  rewriter.create<SelectReturnOp>(op.getLoc(), mlir::ValueRange{cmpOp});

  return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<graphalg::UnionOp>::matchAndRewrite(
    graphalg::UnionOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto targetType =
      llvm::cast<RelationType>(typeConverter->convertType(op.getType()));
  MatrixAdaptor target(op, targetType);

  llvm::SmallVector<mlir::Value> inputs;
  for (auto [matrix, rel] :
       llvm::zip_equal(op.getInputs(), adaptor.getInputs())) {
    MatrixAdaptor input(matrix, rel);

    // Drop columns we don't want in the output.
    llvm::SmallVector<ColumnIdx, 3> remap;
    if (target.hasRowColumn()) {
      remap.push_back(input.rowColumn());
    }

    if (target.hasColColumn()) {
      remap.push_back(input.colColumn());
    }

    remap.push_back(input.valColumn());
    inputs.push_back(
        rewriter.createOrFold<RemapOp>(op.getLoc(), input.relation(), remap));
  }

  auto newOp = rewriter.createOrFold<UnionOp>(op.getLoc(), targetType, inputs);
  rewriter.replaceOp(op, newOp);

  return mlir::success();
}

// =============================================================================
// ============================ Tuple Op Conversion ============================
// =============================================================================

template <>
mlir::LogicalResult OpConversion<graphalg::ConstantOp>::matchAndRewrite(
    graphalg::ConstantOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto value = convertConstant(op, op.getValue());
  if (mlir::failed(value)) {
    return mlir::failure();
  }

  rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, *value);
  return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<graphalg::AddOp>::matchAndRewrite(
    graphalg::AddOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto sring = op.getType();
  auto *ctx = rewriter.getContext();
  if (sring == graphalg::SemiringTypes::forBool(ctx)) {
    rewriter.replaceOpWithNewOp<mlir::arith::OrIOp>(op, adaptor.getLhs(),
                                                    adaptor.getRhs());
  } else if (sring == graphalg::SemiringTypes::forInt(ctx)) {
    rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(op, adaptor.getLhs(),
                                                     adaptor.getRhs());
  } else if (sring == graphalg::SemiringTypes::forReal(ctx)) {
    rewriter.replaceOpWithNewOp<mlir::arith::AddFOp>(op, adaptor.getLhs(),
                                                     adaptor.getRhs());
  } else if (sring == graphalg::SemiringTypes::forTropInt(ctx)) {
    rewriter.replaceOpWithNewOp<mlir::arith::MinSIOp>(op, adaptor.getLhs(),
                                                      adaptor.getRhs());
  } else if (sring == graphalg::SemiringTypes::forTropReal(ctx)) {
    rewriter.replaceOpWithNewOp<mlir::arith::MinimumFOp>(op, adaptor.getLhs(),
                                                         adaptor.getRhs());
  } else if (sring == graphalg::SemiringTypes::forTropMaxInt(ctx)) {
    rewriter.replaceOpWithNewOp<mlir::arith::MaxSIOp>(op, adaptor.getLhs(),
                                                      adaptor.getRhs());
  } else {
    return op->emitOpError("conversion not supported for semiring ") << sring;
  }

  return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<graphalg::CastScalarOp>::matchAndRewrite(
    graphalg::CastScalarOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto *ctx = rewriter.getContext();
  auto inRing =
      llvm::cast<graphalg::SemiringTypeInterface>(op.getInput().getType());
  auto outRing = llvm::cast<graphalg::SemiringTypeInterface>(op.getType());
  assert(inRing != outRing && "Identity cast not removed by fold()");

  if (outRing == graphalg::SemiringTypes::forBool(ctx)) {
    // Rewrite to: input != zero(inRing)
    auto addIdent = convertConstant(op, inRing.addIdentity());
    if (mlir::failed(addIdent)) {
      return mlir::failure();
    }

    auto addIdentOp =
        rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), *addIdent);

    if (addIdentOp.getType().isF64()) {
      rewriter.replaceOpWithNewOp<mlir::arith::CmpFOp>(
          op, mlir::arith::CmpFPredicate::ONE, adaptor.getInput(), addIdentOp);
    } else {
      rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(
          op, mlir::arith::CmpIPredicate::ne, adaptor.getInput(), addIdentOp);
    }

    return mlir::success();
  } else if (inRing == graphalg::SemiringTypes::forBool(ctx)) {
    // Mapping:
    // true -> multiplicative identity
    // false -> additive identity
    auto trueValue = convertConstant(op, outRing.mulIdentity());
    if (mlir::failed(trueValue)) {
      return mlir::failure();
    }

    auto falseValue = convertConstant(op, outRing.addIdentity());
    if (mlir::failed(falseValue)) {
      return mlir::failure();
    }

    auto trueOp =
        rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), *trueValue);
    auto falseOp =
        rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), *falseValue);
    rewriter.replaceOpWithNewOp<mlir::arith::SelectOp>(op, adaptor.getInput(),
                                                       trueOp, falseOp);
    return mlir::success();
  } else if (inRing == graphalg::SemiringTypes::forInt(ctx) &&
             outRing == graphalg::SemiringTypes::forReal(ctx)) {
    // Promote to i64 -> f64
    rewriter.replaceOpWithNewOp<mlir::arith::SIToFPOp>(op, outRing,
                                                       adaptor.getInput());
    return mlir::success();
  } else if (inRing == graphalg::SemiringTypes::forReal(ctx) &&
             outRing == graphalg::SemiringTypes::forInt(ctx)) {
    // Truncate to int
    rewriter.replaceOpWithNewOp<mlir::arith::FPToSIOp>(op, outRing,
                                                       adaptor.getInput());
    return mlir::success();
  } else if (isTropicalnessCast(inRing, outRing)) {
    // Only cast the 'tropicalness' of the type. The underlying relational type
    // does not change. Preserve the value unless it is the additive identity,
    // in which case we remap it to the additive identity of the output ring.
    auto selectOp = preserveAdditiveIdentity(op, adaptor.getInput(),
                                             adaptor.getInput(), rewriter);
    rewriter.replaceOp(op, selectOp);
    return mlir::success();
  } else if (inRing == graphalg::SemiringTypes::forTropInt(ctx) &&
             outRing == graphalg::SemiringTypes::forTropReal(ctx)) {
    // trop_i64 to trop_f64
    // Cast the underlying relational type, but preserve the additive identity.
    auto castOp = rewriter.create<mlir::arith::SIToFPOp>(
        op.getLoc(), rewriter.getF64Type(), adaptor.getInput());
    auto selectOp =
        preserveAdditiveIdentity(op, adaptor.getInput(), castOp, rewriter);
    rewriter.replaceOp(op, selectOp);
    return mlir::success();
  } else if (inRing == graphalg::SemiringTypes::forTropReal(ctx) &&
             outRing == graphalg::SemiringTypes::forTropInt(ctx)) {
    // trop_f64 to trop_i64
    // Cast the underlying relational type, but preserve the additive identity.
    auto castOp = rewriter.create<mlir::arith::FPToSIOp>(
        op.getLoc(), rewriter.getI64Type(), adaptor.getInput());
    auto selectOp =
        preserveAdditiveIdentity(op, adaptor.getInput(), castOp, rewriter);
    rewriter.replaceOp(op, selectOp);
    return mlir::success();
  }

  return op->emitOpError("cast from ")
         << op.getInput().getType() << " to " << op.getType()
         << " not yet supported in " << GARelDialect::getDialectNamespace()
         << " dialect";
}

template <>
mlir::LogicalResult OpConversion<graphalg::EqOp>::matchAndRewrite(
    graphalg::EqOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();
  if (lhs.getType().isF64()) {
    assert(rhs.getType().isF64());
    rewriter.replaceOpWithNewOp<mlir::arith::CmpFOp>(
        op, mlir::arith::CmpFPredicate::OEQ, lhs, rhs);
  } else {
    assert(lhs.getType().isSignlessInteger());
    assert(rhs.getType().isSignlessInteger());
    rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(
        op, mlir::arith::CmpIPredicate::eq, lhs, rhs);
  }

  return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<graphalg::MulOp>::matchAndRewrite(
    graphalg::MulOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto sring = op.getType();
  auto *ctx = rewriter.getContext();
  auto mulOp = createMul(op, llvm::cast<graphalg::SemiringTypeInterface>(sring),
                         adaptor.getLhs(), adaptor.getRhs(), rewriter);
  if (mlir::failed(mulOp)) {
    return mlir::failure();
  }

  rewriter.replaceOp(op, *mulOp);
  return mlir::success();
}

static bool hasRelationSignature(mlir::func::FuncOp op) {
  // All inputs should be relations
  auto funcType = op.getFunctionType();
  for (auto input : funcType.getInputs()) {
    if (!llvm::isa<RelationType>(input)) {
      return false;
    }
  }

  // There should be exactly one relation result
  return funcType.getNumResults() == 1 &&
         llvm::isa<RelationType>(funcType.getResult(0));
}

static bool hasRelationOperands(mlir::Operation *op) {
  return llvm::all_of(op->getOperandTypes(),
                      [](auto t) { return llvm::isa<RelationType>(t); });
}

void GraphAlgToRel::runOnOperation() {
  mlir::ConversionTarget target(getContext());
  // Eliminate all graphalg ops
  target.addIllegalDialect<graphalg::GraphAlgDialect>();
  // Turn them into relational ops.
  target.addLegalDialect<GARelDialect>();
  // and arith ops for the scalar operations.
  target.addLegalDialect<mlir::arith::ArithDialect>();
  // Keep container module.
  target.addLegalOp<mlir::ModuleOp>();
  // Keep functions, but change their signature.
  target.addDynamicallyLegalOp<mlir::func::FuncOp>(hasRelationSignature);
  target.addDynamicallyLegalOp<mlir::func::ReturnOp>(hasRelationOperands);

  SemiringTypeConverter semiringTypeConverter;
  MatrixTypeConverter matrixTypeConverter(&getContext(), semiringTypeConverter);

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<
      OpConversion<mlir::func::FuncOp>, OpConversion<mlir::func::ReturnOp>,
      OpConversion<graphalg::TransposeOp>, OpConversion<graphalg::BroadcastOp>,
      OpConversion<graphalg::ConstantMatrixOp>,
      OpConversion<graphalg::DeferredReduceOp>, OpConversion<graphalg::DiagOp>,
      OpConversion<graphalg::ForConstOp>, OpConversion<graphalg::YieldOp>,
      OpConversion<graphalg::MatMulJoinOp>, OpConversion<graphalg::PickAnyOp>,
      OpConversion<graphalg::TrilOp>, OpConversion<graphalg::UnionOp>>(
      matrixTypeConverter, &getContext());
  patterns.add<ApplyOpConversion>(semiringTypeConverter, matrixTypeConverter,
                                  &getContext());

  // Scalar patterns.
  patterns
      .add<OpConversion<graphalg::ApplyReturnOp>,
           OpConversion<graphalg::ConstantOp>, OpConversion<graphalg::AddOp>,
           OpConversion<graphalg::CastScalarOp>, OpConversion<graphalg::EqOp>,
           OpConversion<graphalg::MulOp>>(semiringTypeConverter, &getContext());

  if (mlir::failed(mlir::applyFullConversion(getOperation(), target,
                                             std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace garel
