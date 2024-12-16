#include "lingodb/execution/BackendPasses.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/DecomposeCallGraphTypes.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
using namespace mlir;

namespace {
template <typename SourceOp>
class DecomposeCallGraphTypesOpConversionPattern
   : public OpConversionPattern<SourceOp> {
   public:
   DecomposeCallGraphTypesOpConversionPattern(TypeConverter& typeConverter,
                                              MLIRContext* context,
                                              ValueDecomposer& decomposer,
                                              PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(typeConverter, context, benefit),
        decomposer(decomposer) {}

   protected:
   ValueDecomposer& decomposer;
};
struct DecomposeCallGraphTypesForBranchOp
   : public DecomposeCallGraphTypesOpConversionPattern<mlir::cf::BranchOp> {
   using DecomposeCallGraphTypesOpConversionPattern::
      DecomposeCallGraphTypesOpConversionPattern;
   LogicalResult
   matchAndRewrite(cf::BranchOp op, OpAdaptor adaptor,
                   ConversionPatternRewriter& rewriter) const final {
      SmallVector<Value, 2> newOperands;
      for (Value operand : adaptor.getDestOperands())
         decomposer.decomposeValue(rewriter, op.getLoc(), operand.getType(),
                                   operand, newOperands);
      rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, op.getDest(), newOperands);
      return success();
   }
};
struct DecomposeCallGraphTypesForCondBranchOp
   : public DecomposeCallGraphTypesOpConversionPattern<mlir::cf::CondBranchOp> {
   using DecomposeCallGraphTypesOpConversionPattern::
      DecomposeCallGraphTypesOpConversionPattern;
   LogicalResult
   matchAndRewrite(cf::CondBranchOp op, OpAdaptor adaptor,
                   ConversionPatternRewriter& rewriter) const final {
      SmallVector<Value, 2> newTrueOperands;
      SmallVector<Value, 2> newFalseOperands;
      for (Value operand : adaptor.getTrueDestOperands())
         decomposer.decomposeValue(rewriter, op.getLoc(), operand.getType(), operand, newTrueOperands);
      for (Value operand : adaptor.getFalseDestOperands())
         decomposer.decomposeValue(rewriter, op.getLoc(), operand.getType(), operand, newFalseOperands);
      rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(op, op.getCondition(), op.getTrueDest(),newTrueOperands,op.getFalseDest(),newFalseOperands);
      return success();
   }
};
/// A pass for testing call graph type decomposition.
///
/// This instantiates the patterns with a TypeConverter and ValueDecomposer
/// that splits tuple types into their respective element types.
/// For example, `tuple<T1, T2, T3> --> T1, T2, T3`.
struct DecomposeTuplesPass
   : public PassWrapper<DecomposeTuplesPass, OperationPass<ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DecomposeTuplesPass)

   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<util::UtilDialect>();
   }
   StringRef getArgument() const final {
      return "util-decompose-tuples";
   }
   StringRef getDescription() const final {
      return "Decomposes types at call graph boundaries.";
   }
   void runOnOperation() override {
      ModuleOp module = getOperation();
      auto* context = &getContext();
      TypeConverter typeConverter;
      ConversionTarget target(*context);
      ValueDecomposer decomposer;
      RewritePatternSet patterns(context);

      target.addLegalDialect<util::UtilDialect>();

      target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
         return typeConverter.isLegal(op.getOperandTypes());
      });
      target.addDynamicallyLegalOp<cf::BranchOp>([&](cf::BranchOp op) {
         return typeConverter.isLegal(op.getOperandTypes());
      });
      target.addDynamicallyLegalOp<cf::CondBranchOp>([&](cf::CondBranchOp op) {
         return typeConverter.isLegal(op.getOperandTypes());
      });
      target.addDynamicallyLegalOp<func::CallOp>(
         [&](func::CallOp op) { return typeConverter.isLegal(op); });
      target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
         if (!typeConverter.isSignatureLegal(op.getFunctionType())) {
            return false;
         }
         for (auto& block : op.getBody().getBlocks()) {
            if (!typeConverter.isLegal(block.getArgumentTypes())) {
               return false;
            }
         }
         return true;
      });

      typeConverter.addConversion([](Type type) { return type; });
      typeConverter.addConversion(
         [](TupleType tupleType, SmallVectorImpl<Type>& types) {
            tupleType.getFlattenedTypes(types);
            return success();
         });

      decomposer.addDecomposeValueConversion([](OpBuilder& builder, Location loc,
                                                TupleType resultType, Value value,
                                                SmallVectorImpl<Value>& values) {
         for (unsigned i = 0, e = resultType.size(); i < e; ++i) {
            Value res = builder.create<util::GetTupleOp>(
               loc, resultType.getType(i), value, builder.getI32IntegerAttr(i));
            values.push_back(res);
         }
         return success();
      });

      typeConverter.addArgumentMaterialization(
         [](OpBuilder& builder, TupleType resultType, ValueRange inputs,
            Location loc) -> std::optional<Value> {
            if (inputs.size() == 1)
               return std::nullopt;
            TupleType tuple = builder.getTupleType(inputs.getTypes());
            Value value = builder.create<util::PackOp>(loc, tuple, inputs);
            return value;
         });

      populateDecomposeCallGraphTypesPatterns(context, typeConverter, decomposer,
                                              patterns);
      patterns.insert<DecomposeCallGraphTypesForBranchOp>(typeConverter, patterns.getContext(), decomposer);
      patterns.insert<DecomposeCallGraphTypesForCondBranchOp>(typeConverter, patterns.getContext(), decomposer);

      if (failed(applyPartialConversion(module, target, std::move(patterns))))
         return signalPassFailure();
   }
};

} // namespace
std::unique_ptr<mlir::Pass> execution::createDecomposeTuplePass() {
   return std::make_unique<DecomposeTuplesPass>();
}