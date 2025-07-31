#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/Transforms/OneToNTypeConversion.h>

#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "lingodb/execution/BackendPasses.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
SmallVector<Value> buildDecomposeTuple(OpBuilder& builder,
                                       TypeRange resultTypes,
                                       ValueRange inputs, Location loc) {
   if (inputs.size() != 1)
      return {};
   Value tuple = inputs.front();
   auto tupleType = dyn_cast<TupleType>(tuple.getType());
   if (!tupleType)
      return {};
   SmallVector<Type> flattenedTypes;
   tupleType.getFlattenedTypes(flattenedTypes);
   if (TypeRange(resultTypes) != TypeRange(flattenedTypes))
      return {};
   // Recursively decompose the tuple.
   SmallVector<Value> result;
   std::function<void(Value)> decompose = [&](Value tuple) {
      auto tupleType = dyn_cast<TupleType>(tuple.getType());
      if (!tupleType) {
         result.push_back(tuple);
         return;
      }
      for (unsigned i = 0, e = tupleType.size(); i < e; ++i) {
         Type elementType = tupleType.getType(i);
         Value element = builder.create<lingodb::compiler::dialect::util::GetTupleOp>(
            loc, elementType, tuple, builder.getI32IntegerAttr(i));
         decompose(element);
      }
   };
   decompose(tuple);
   return result;
}

Value buildMakeTupleOp(OpBuilder& builder, TupleType resultType,
                       ValueRange inputs, Location loc) {
   SmallVector<Value> elements;
   elements.reserve(resultType.getTypes().size());
   ValueRange::iterator inputIt = inputs.begin();
   for (Type elementType : resultType.getTypes()) {
      if (auto nestedTupleType = dyn_cast<TupleType>(elementType)) {
         SmallVector<Type> nestedFlattenedTypes;
         nestedTupleType.getFlattenedTypes(nestedFlattenedTypes);
         size_t numNestedFlattenedTypes = nestedFlattenedTypes.size();
         ValueRange nestedFlattenedelements(inputIt,
                                            inputIt + numNestedFlattenedTypes);
         inputIt += numNestedFlattenedTypes;

         // Recurse on the values for the nested TupleType.
         Value res = buildMakeTupleOp(builder, nestedTupleType,
                                      nestedFlattenedelements, loc);
         if (!res)
            return Value();
         elements.push_back(res);
      } else {
         elements.push_back(*inputIt++);
      }
   }
   return builder.create<lingodb::compiler::dialect::util::PackOp>(loc, resultType, elements);
}

SmallVector<Value> flattenValues(ArrayRef<ValueRange> values) {
   SmallVector<Value> result;
   for (const auto& vals : values)
      llvm::append_range(result, vals);
   return result;
}

class BranchOpTypeConversion : public OpConversionPattern<cf::BranchOp> {
   public:
   using OpConversionPattern::OpConversionPattern;

   LogicalResult
   matchAndRewrite(cf::BranchOp op, OneToNOpAdaptor adaptor,
                   ConversionPatternRewriter& rewriter) const final {
      rewriter.replaceOpWithNewOp<cf::BranchOp>(op, op.getDest(),
                                                flattenValues(adaptor.getOperands()));
      return success();
   }
};

class CondBranchOpTypeConversion : public OpConversionPattern<cf::CondBranchOp> {
   public:
   using OpConversionPattern::OpConversionPattern;

   LogicalResult
   matchAndRewrite(cf::CondBranchOp op, OneToNOpAdaptor adaptor,
                   ConversionPatternRewriter& rewriter) const final {
      rewriter.replaceOpWithNewOp<cf::CondBranchOp>(op, op.getCondition(), op.getTrueDest(),
                                                    flattenValues(adaptor.getTrueDestOperands()),
                                                    op.getFalseDest(),
                                                    flattenValues(adaptor.getFalseDestOperands()));
      return success();
   }
};

struct DecomposeTuplesPass
   : public PassWrapper<DecomposeTuplesPass, OperationPass<ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DecomposeTuplesPass)

   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<lingodb::compiler::dialect::util::UtilDialect>();
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
      RewritePatternSet patterns(context);

      target.addLegalDialect<lingodb::compiler::dialect::util::UtilDialect>();

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

      typeConverter.addSourceMaterialization(buildMakeTupleOp);
      typeConverter.addTargetMaterialization(buildDecomposeTuple);

      populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
      populateReturnOpTypeConversionPattern(patterns, typeConverter);
      populateCallOpTypeConversionPattern(patterns, typeConverter);
      patterns.add<BranchOpTypeConversion>(typeConverter, patterns.getContext());
      patterns.add<CondBranchOpTypeConversion>(typeConverter, patterns.getContext());

      if (failed(applyPartialConversion(module, target, std::move(patterns))))
         return signalPassFailure();
   }
};
} // namespace
std::unique_ptr<Pass> lingodb::execution::createDecomposeTuplePass() {
   return std::make_unique<DecomposeTuplesPass>();
}
