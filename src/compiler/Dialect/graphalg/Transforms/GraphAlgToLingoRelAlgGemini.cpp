#include <array>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
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

// LingoDB Includes
#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/DB/IR/DBTypes.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/TupleStream/Column.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

#include "lingodb/compiler/Conversion/RelAlgToSubOp/OrderedAttributes.h"
#include "lingodb/compiler/Dialect/Arrow/IR/ArrowDialect.h"
#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Utils.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/Dialect/util/FunctionHelper.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "lingodb/runtime/ExternalDataSourceProperty.h"


#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

// GraphAlg Includes
#include "lingodb/compiler/Dialect/graphalg/GraphAlgAttr.h"
#include "lingodb/compiler/Dialect/graphalg/GraphAlgCast.h"
#include "lingodb/compiler/Dialect/graphalg/GraphAlgDialect.h"
#include "lingodb/compiler/Dialect/graphalg/GraphAlgOps.h"
#include "lingodb/compiler/Dialect/graphalg/GraphAlgTypes.h"
#include "lingodb/compiler/Dialect/graphalg/SemiringTypes.h"

#include <mlir/Dialect/SCF/IR/SCFOps.h.inc>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>

namespace graphalg {

#define GEN_PASS_DEF_GRAPHALGTORELALG
#include "lingodb/compiler/Dialect/graphalg/GraphAlgPasses.h.inc"

using namespace mlir;
using namespace lingodb::compiler::dialect;

/**
 * Helper to generate unique attribute names for RelAlg.
 */
struct AttributeGenerator {
   static std::string nextName(const std::string& prefix = "attr") {
      static size_t counter = 0;
      return prefix + "_" + std::to_string(counter++);
   }
};

/**
 * Metadata attached to RelAlg operations to track which attributes correspond
 * to the Matrix dimensions (Row, Col) and Value.
 */
struct MatrixMeta {
   tuples::ColumnRefAttr row;
   tuples::ColumnRefAttr col;
   tuples::ColumnRefAttr val;

   bool hasRow() const { return row != nullptr; }
   bool hasCol() const { return col != nullptr; }

   // Serialization to DictionaryAttr for attachment to Ops
   DictionaryAttr toDict(Builder& builder) const {
      SmallVector<NamedAttribute, 3> attrs;
      if (row) attrs.push_back(builder.getNamedAttr("row", row));
      if (col) attrs.push_back(builder.getNamedAttr("col", col));
      if (val) attrs.push_back(builder.getNamedAttr("val", val));
      return builder.getDictionaryAttr(attrs);
   }

   // Deserialization
   static MatrixMeta fromOp(Operation* op) {
      if (!op) return {};
      auto dict = op->getAttrOfType<DictionaryAttr>("ga_meta");
      if (!dict) return {};
      MatrixMeta meta;
      if (auto attr = dict.get("row")) meta.row = llvm::cast<tuples::ColumnRefAttr>(attr);
      if (auto attr = dict.get("col")) meta.col = llvm::cast<tuples::ColumnRefAttr>(attr);
      if (auto attr = dict.get("val")) meta.val = llvm::cast<tuples::ColumnRefAttr>(attr);
      return meta;
   }
   static void set(Value val, const MatrixMeta& meta) {
      if (auto* op = val.getDefiningOp()) {
         OpBuilder builder(op);
         op->setAttr("ga_meta", meta.toDict(builder));
      }
   }
};

/**
 * Type Converter: GraphAlg -> TupleStream / DB Types
 */
class GraphAlgTypeConverter : public TypeConverter {
   public:
   GraphAlgTypeConverter(MLIRContext* ctx) {
      // Matrix -> TupleStream
      addConversion([&](graphalg::MatrixType type) -> Type {
         return tuples::TupleStreamType::get(ctx);
      });

      // Scalar/Semiring types -> DB Types
      addConversion([&](Type type) -> std::optional<Type> {
         if (auto st = llvm::dyn_cast<graphalg::SemiringTypeInterface>(type)) {
            return convertSemiringType(st);
         }
         return std::nullopt;
      });

      // Pass-through
      addConversion([](Type type) { return type; });
   }

   static Type convertSemiringType(graphalg::SemiringTypeInterface t) {
      auto* ctx = t.getContext();

      // Boolean
      if (t == graphalg::SemiringTypes::forBool(ctx)) {
         return IntegerType::get(ctx, 1);
      }
      // Integer (64-bit)
      if (t == graphalg::SemiringTypes::forInt(ctx) ||
          t == graphalg::SemiringTypes::forTropInt(ctx) ||
          t == graphalg::SemiringTypes::forTropMaxInt(ctx)) {
         return IntegerType::get(ctx, 64);
      }
      // Float (64-bit)
      if (t == graphalg::SemiringTypes::forReal(ctx) ||
          t == graphalg::SemiringTypes::forTropReal(ctx)) {
         return Float64Type::get(ctx);
      }
      return Type();
   }
};

/**
 * Pass Definition
 */
class GraphAlgToRelAlgPass : public PassWrapper<GraphAlgToRelAlgPass, OperationPass<ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GraphAlgToRelAlgPass)

   void runOnOperation() final;
};

// =============================================================================
// =============================== Helpers =====================================
// =============================================================================

static bool isBool(Type t) {
   return t == graphalg::SemiringTypes::forBool(t.getContext());
}

static bool isTropical(Type t) {
   return llvm::isa<graphalg::TropI64Type, graphalg::TropF64Type>(t);
}

static bool isTropicalMax(Type t) {
   return t == graphalg::SemiringTypes::forTropMaxInt(t.getContext());
}

// Create a column definition attribute properly constructing the Column object
static tuples::ColumnDefAttr createColumnDef(MLIRContext* ctx, StringRef name, Type type) {
   auto column = std::make_shared<tuples::Column>();
   column->type = type;

   auto symName = SymbolRefAttr::get(ctx, name);
   return tuples::ColumnDefAttr::get(ctx, symName, column, Attribute());
}

// Create a reference to a column definition
static tuples::ColumnRefAttr createColumnRef(tuples::ColumnDefAttr def) {
   return tuples::ColumnRefAttr::get(def.getContext(), def.getName(), def.getColumnPtr());
}

// =============================================================================
// =============================== Patterns ====================================
// =============================================================================

class ConstantMatrixConversion : public OpConversionPattern<graphalg::ConstantMatrixOp> {
   public:
   using OpConversionPattern::OpConversionPattern;

   LogicalResult matchAndRewrite(graphalg::ConstantMatrixOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter& rewriter) const override {
      auto* ctx = rewriter.getContext();
      auto matType = op.getType();

      // Explicit cast to interface
      auto semiringType = llvm::cast<graphalg::SemiringTypeInterface>(matType.getSemiring());
      Type valType = GraphAlgTypeConverter::convertSemiringType(semiringType);

      auto valDef = createColumnDef(ctx, AttributeGenerator::nextName("const_val"), valType);
      auto valRef = createColumnRef(valDef);

      auto valConst = op.getValue();
      Attribute rawVal = valConst;
      if (auto tropInt = llvm::dyn_cast<graphalg::TropIntAttr>(valConst)) rawVal = tropInt.getValue();
      if (auto tropF = llvm::dyn_cast<graphalg::TropFloatAttr>(valConst)) rawVal = tropF.getValue();

      auto constRel = rewriter.create<relalg::ConstRelationOp>(
         op.getLoc(),
         ArrayAttr::get(ctx, {valDef}),
         ArrayAttr::get(ctx, {ArrayAttr::get(ctx, {rawVal})}));

      MatrixMeta meta;
      meta.val = valRef;

      constRel->setAttr("ga_meta", meta.toDict(rewriter));
      rewriter.replaceOp(op, constRel.getResult());
      return success();
   }
};

class MatMulJoinConversion : public OpConversionPattern<graphalg::MatMulJoinOp> {
   public:
   using OpConversionPattern::OpConversionPattern;

   LogicalResult matchAndRewrite(graphalg::MatMulJoinOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter& rewriter) const override {
      Value lhsRel = adaptor.getLhs();
      Value rhsRel = adaptor.getRhs();

      auto lhsMeta = MatrixMeta::fromOp(lhsRel.getDefiningOp());
      auto rhsMeta = MatrixMeta::fromOp(rhsRel.getDefiningOp());

      if (!lhsMeta.hasCol() || !rhsMeta.hasRow()) {
         return op.emitError("MatMulJoin requires LHS col and RHS row");
      }

      auto joinOp = rewriter.create<relalg::InnerJoinOp>(op.getLoc(), lhsRel, rhsRel);
      {
         Block* predBlock = new Block;
         joinOp.getPredicate().push_back(predBlock);
         auto tupleArg = predBlock->addArgument(tuples::TupleType::get(rewriter.getContext()), op.getLoc());

         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(predBlock);

         auto lhsColVal = rewriter.create<tuples::GetColumnOp>(op.getLoc(), lhsMeta.col.getColumn().type, lhsMeta.col, tupleArg);
         auto rhsRowVal = rewriter.create<tuples::GetColumnOp>(op.getLoc(), rhsMeta.row.getColumn().type, rhsMeta.row, tupleArg);

         Value cmp = rewriter.create<db::CmpOp>(op.getLoc(), db::DBCmpPredicate::eq, lhsColVal, rhsRowVal);
         rewriter.create<tuples::ReturnOp>(op.getLoc(), cmp);
      }

      auto mapOp = rewriter.create<relalg::MapOp>(op.getLoc(), joinOp.getResult());

      auto semiringType = llvm::cast<graphalg::SemiringTypeInterface>(op.getType().getSemiring());
      Type resType = GraphAlgTypeConverter::convertSemiringType(semiringType);

      auto newValDef = createColumnDef(rewriter.getContext(), AttributeGenerator::nextName("mul_res"), resType);
      auto newValRef = createColumnRef(newValDef);

      mapOp.setComputedColsAttr(ArrayAttr::get(rewriter.getContext(), {newValDef}));

      {
         Block* mapBlock = new Block;
         mapOp.getPredicate().push_back(mapBlock);
         auto tupleArg = mapBlock->addArgument(tuples::TupleType::get(rewriter.getContext()), op.getLoc());

         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(mapBlock);

         auto lhsVal = rewriter.create<tuples::GetColumnOp>(op.getLoc(), lhsMeta.val.getColumn().type, lhsMeta.val, tupleArg);
         auto rhsVal = rewriter.create<tuples::GetColumnOp>(op.getLoc(), rhsMeta.val.getColumn().type, rhsMeta.val, tupleArg);

         Value res;
         if (isTropical(op.getType().getSemiring())) {
            res = rewriter.create<db::AddOp>(op.getLoc(), resType, lhsVal, rhsVal);
         } else {
            res = rewriter.create<db::MulOp>(op.getLoc(), resType, lhsVal, rhsVal);
         }

         rewriter.create<tuples::ReturnOp>(op.getLoc(), res);
      }

      MatrixMeta resMeta;
      if (lhsMeta.hasRow()) resMeta.row = lhsMeta.row;
      if (rhsMeta.hasCol()) resMeta.col = rhsMeta.col;
      resMeta.val = newValRef;

      mapOp->setAttr("ga_meta", resMeta.toDict(rewriter));
      rewriter.replaceOp(op, mapOp.getResult());
      return success();
   }
};

class ApplyOpConversion : public OpConversionPattern<graphalg::ApplyOp> {
   public:
   using OpConversionPattern::OpConversionPattern;

   LogicalResult matchAndRewrite(graphalg::ApplyOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter& rewriter) const override {
      auto inputs = adaptor.getInputs();
      if (inputs.empty()) return failure();

      Value currentRel = inputs[0];

      auto mapOp = rewriter.create<relalg::MapOp>(op.getLoc(), currentRel);

      auto semiringType = llvm::cast<graphalg::SemiringTypeInterface>(op.getType().getSemiring());
      Type resType = GraphAlgTypeConverter::convertSemiringType(semiringType);

      auto resDef = createColumnDef(rewriter.getContext(), AttributeGenerator::nextName("apply_res"), resType);
      auto resRef = createColumnRef(resDef);

      mapOp.setComputedColsAttr(ArrayAttr::get(rewriter.getContext(), {resDef}));

      {
         Block* mapBlock = new Block;
         mapOp.getPredicate().push_back(mapBlock);
         auto tupleArg = mapBlock->addArgument(tuples::TupleType::get(rewriter.getContext()), op.getLoc());

         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(mapBlock);

         SmallVector<Value> newArgs;
         for (size_t i = 0; i < op.getInputs().size(); ++i) {
            auto inputMeta = MatrixMeta::fromOp(adaptor.getInputs()[i].getDefiningOp());
            if (inputMeta.val) {
               newArgs.push_back(rewriter.create<tuples::GetColumnOp>(
                  op.getLoc(), inputMeta.val.getColumn().type, inputMeta.val, tupleArg));
            } else {
               // Error handling: missing metadata for input
               std::cerr << "Not implemented" << std::endl;
               return failure();
            }
         }

         rewriter.inlineBlockBefore(&op.getBody().front(), mapBlock, mapBlock->end(), newArgs);

         Operation* term = mapBlock->getTerminator();
         if (auto ret = llvm::dyn_cast<graphalg::ApplyReturnOp>(term)) {
            rewriter.setInsertionPoint(term);
            rewriter.replaceOpWithNewOp<tuples::ReturnOp>(term, ret.getValue());
         }
      }

      MatrixMeta resMeta = MatrixMeta::fromOp(inputs[0].getDefiningOp());
      resMeta.val = resRef;
      mapOp->setAttr("ga_meta", resMeta.toDict(rewriter));

      rewriter.replaceOp(op, mapOp.getResult());
      return success();
   }
};

class DeferredReduceConversion : public OpConversionPattern<graphalg::DeferredReduceOp> {
   public:
   using OpConversionPattern::OpConversionPattern;

   static relalg::AggrFunc getAggrFuncForSemiring(Type semiringType) {
      // Assuming you have helpers like isTropical, isTropicalMax, etc.
      // You can also check against specific singleton types if available.

      if (isTropical(semiringType)) {
         return relalg::AggrFunc::min;
      }
      if (isTropicalMax(semiringType)) {
         return relalg::AggrFunc::max;
      }
      if (isBool(semiringType)) {
         // Boolean "Addition" is OR. In aggregation, MAX(0, 1, 1) = 1 (True).
         // SUM would also work but might overflow if not careful, MAX is safer for Bool.
         return relalg::AggrFunc::max;
      }
      // Default for Int/Real is Arithmetic Addition
      return relalg::AggrFunc::sum;
   }

   LogicalResult matchAndRewrite(graphalg::DeferredReduceOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter& rewriter) const override {
      // 1. Prepare Input
      // (If you have multiple inputs, you would insert the Union logic here)
      Value inputRel = adaptor.getInputs()[0];
      auto inputOp = op.getInputs()[0].getDefiningOp();
      auto inputMeta = MatrixMeta::fromOp(inputOp);

      auto resMatrixType = op.getType();
      auto semiringType = llvm::cast<graphalg::SemiringTypeInterface>(resMatrixType.getSemiring());

      // 2. Determine GroupBy Keys (Row/Col)
      SmallVector<Attribute> groupByAttrs;
      MatrixMeta resMeta;

      if (!resMatrixType.getRows().isOne() && inputMeta.hasRow()) {
         groupByAttrs.push_back(inputMeta.row);
         resMeta.row = inputMeta.row;
      }
      if (!resMatrixType.getCols().isOne() && inputMeta.hasCol()) {
         groupByAttrs.push_back(inputMeta.col);
         resMeta.col = inputMeta.col;
      }

      // 3. Prepare Aggregation Column
      Type valType = GraphAlgTypeConverter::convertSemiringType(semiringType);

      // Define the computed column
      auto aggDefAttr = createColumnDef(rewriter.getContext(), AttributeGenerator::nextName("agg_val"), valType);

      // Store the reference in metadata
      resMeta.val = createColumnRef(aggDefAttr);

      // 4. Create AggregationOp
      auto aggOp = rewriter.create<relalg::AggregationOp>(
         op.getLoc(),
         tuples::TupleStreamType::get(rewriter.getContext()),
         inputRel,
         ArrayAttr::get(rewriter.getContext(), groupByAttrs),
         ArrayAttr::get(rewriter.getContext(), {aggDefAttr}));

      // 5. Populate Aggregation Region (The "Aggregator" logic)
      Block* aggBlock = rewriter.createBlock(&aggOp.getAggrFunc());
      Value groupStream = aggBlock->addArgument(tuples::TupleStreamType::get(rewriter.getContext()), op.getLoc());

      {
         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(aggBlock);

         // Dynamic check for correct aggregation function (Min, Max, Sum)
         relalg::AggrFunc func = getAggrFuncForSemiring(semiringType);

         Value res = rewriter.create<relalg::AggrFuncOp>(
            op.getLoc(),
            valType,
            relalg::AggrFuncAttr::get(rewriter.getContext(), func),
            groupStream,
            inputMeta.val);

         rewriter.create<tuples::ReturnOp>(op.getLoc(), res);
      }

      // 6. Finish
      MatrixMeta::set(aggOp.getResult(), resMeta);
      rewriter.replaceOp(op, aggOp.getResult());

      return success();
   }
};

class AddConversion : public OpConversionPattern<graphalg::AddOp> {
   using OpConversionPattern::OpConversionPattern;
   LogicalResult matchAndRewrite(graphalg::AddOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter& rewriter) const override {
      auto sring = op.getType();
      auto* ctx = rewriter.getContext();
      if (sring == graphalg::SemiringTypes::forBool(ctx)) {
         rewriter.replaceOpWithNewOp<arith::OrIOp>(op, adaptor.getLhs(),
                                                   adaptor.getRhs());
      } else if (sring == graphalg::SemiringTypes::forInt(ctx)) {
         rewriter.replaceOpWithNewOp<arith::AddIOp>(op, adaptor.getLhs(),
                                                    adaptor.getRhs());
      } else if (sring == graphalg::SemiringTypes::forReal(ctx)) {
         rewriter.replaceOpWithNewOp<arith::AddFOp>(op, adaptor.getLhs(),
                                                    adaptor.getRhs());
      } else if (sring == graphalg::SemiringTypes::forTropInt(ctx)) {
         rewriter.replaceOpWithNewOp<arith::MinSIOp>(op, adaptor.getLhs(),
                                                     adaptor.getRhs());
      } else if (sring == graphalg::SemiringTypes::forTropReal(ctx)) {
         rewriter.replaceOpWithNewOp<arith::MinimumFOp>(op, adaptor.getLhs(),
                                                        adaptor.getRhs());
      } else if (sring == graphalg::SemiringTypes::forTropMaxInt(ctx)) {
         rewriter.replaceOpWithNewOp<arith::MaxSIOp>(op, adaptor.getLhs(),
                                                     adaptor.getRhs());
      } else {
         return op->emitOpError("conversion not supported for semiring ") << sring;
      }

      return success();
   }
};

class MulConversion : public OpConversionPattern<graphalg::MulOp> {
   using OpConversionPattern::OpConversionPattern;
   LogicalResult matchAndRewrite(graphalg::MulOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter& rewriter) const override {
      if (isTropical(op.getType())) {
         rewriter.replaceOpWithNewOp<db::AddOp>(op, adaptor.getLhs().getType(), adaptor.getLhs(), adaptor.getRhs());
      } else {
         rewriter.replaceOpWithNewOp<db::MulOp>(op, adaptor.getLhs().getType(), adaptor.getLhs(), adaptor.getRhs());
      }
      return success();
   }
};

void GraphAlgToRelAlgPass::runOnOperation() {
   MLIRContext* context = &getContext();
   GraphAlgTypeConverter typeConverter(context);

   ConversionTarget target(*context);
   target.addLegalOp<ModuleOp>();
   target.addLegalDialect<BuiltinDialect>();

   target.addLegalDialect<relalg::RelAlgDialect>();
   target.addLegalDialect<tuples::TupleStreamDialect>();
   target.addLegalDialect<db::DBDialect>();

   using namespace lingodb::compiler::dialect;
   using namespace mlir;
   target.addLegalDialect<gpu::GPUDialect>();
   target.addLegalDialect<async::AsyncDialect>();
   target.addIllegalDialect<relalg::RelAlgDialect>();
   target.addLegalDialect<subop::SubOperatorDialect>();
   target.addLegalDialect<db::DBDialect>();
   target.addLegalDialect<lingodb::compiler::dialect::arrow::ArrowDialect>();

   target.addLegalDialect<tuples::TupleStreamDialect>();
   target.addLegalDialect<func::FuncDialect>();
   target.addLegalDialect<memref::MemRefDialect>();
   target.addLegalDialect<arith::ArithDialect>();
   target.addLegalDialect<cf::ControlFlowDialect>();
   target.addLegalDialect<scf::SCFDialect>();
   target.addLegalDialect<util::UtilDialect>();

   target.addLegalDialect<arith::ArithDialect>();

   target.addIllegalDialect<graphalg::GraphAlgDialect>();

   target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
   });
   target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return typeConverter.isLegal(op.getOperandTypes());
   });

   RewritePatternSet patterns(context);
   patterns.add<
      ConstantMatrixConversion,
      MatMulJoinConversion,
      ApplyOpConversion,
      DeferredReduceConversion,
      AddConversion,
      MulConversion>(typeConverter, context);

   populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
   populateReturnOpTypeConversionPattern(patterns, typeConverter);

   if (failed(applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
   }
}
std::unique_ptr<OperationPass<ModuleOp>> createGraphAlgToRelAlgPass() {
   return std::make_unique<GraphAlgToRelAlgPass>();
}
} // namespace