#include <array>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
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

#include "lingodb/compiler/Dialect/Arrow/IR/ArrowDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

// GraphAlg Includes
#include "lingodb/compiler/Dialect/graphalg/GraphAlgAttr.h"
#include "lingodb/compiler/Dialect/graphalg/GraphAlgCast.h"
#include "lingodb/compiler/Dialect/graphalg/GraphAlgDialect.h"
#include "lingodb/compiler/Dialect/graphalg/GraphAlgOps.h"
#include "lingodb/compiler/Dialect/graphalg/GraphAlgTypes.h"
#include "lingodb/compiler/Dialect/graphalg/SemiringTypes.h"

namespace graphalg {

#define GEN_PASS_DEF_GRAPHALGTORELALG
#include "lingodb/compiler/Dialect/graphalg/GraphAlgPasses.h.inc"

using namespace mlir;
using namespace lingodb::compiler::dialect;

// =============================================================================
// ================================= Helpers ===================================
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

static tuples::ColumnDefAttr createColumnDef(MLIRContext* ctx, StringRef name, Type type) {
   auto column = std::make_shared<tuples::Column>();
   column->type = type;
   auto symName = SymbolRefAttr::get(ctx, name);
   return tuples::ColumnDefAttr::get(ctx, symName, column, Attribute());
}

static tuples::ColumnRefAttr createColumnRef(tuples::ColumnDefAttr def) {
   return tuples::ColumnRefAttr::get(def.getContext(), def.getName(), def.getColumnPtr());
}

// =============================================================================
// ========================= State & Metadata Management =======================
// =============================================================================

struct AttributeGenerator {
   static std::string nextName(const std::string& prefix = "attr") {
      static size_t counter = 0;
      return prefix + "_" + std::to_string(counter++);
   }
};

struct MatrixMeta {
   tuples::ColumnRefAttr row;
   tuples::ColumnRefAttr col;
   tuples::ColumnRefAttr val;

   tuples::ColumnDefAttr rowDef;
   tuples::ColumnDefAttr colDef;
   tuples::ColumnDefAttr valDef;

   bool hasRow() const { return row != nullptr; }
   bool hasCol() const { return col != nullptr; }
};

// =============================================================================
// ============================== Type Converters ==============================
// =============================================================================

class GraphAlgTypeConverter : public TypeConverter {
   public:
   GraphAlgTypeConverter(MLIRContext* ctx) {
      addConversion([](Type type) { return type; });

      addConversion([](MatrixType type) -> Type {
         return tuples::TupleStreamType::get(type.getContext());
      });

      addConversion([](Type type) -> std::optional<Type> {
         if (auto st = llvm::dyn_cast<SemiringTypeInterface>(type)) {
            if (Type converted = convertSemiringType(st)) {
               return converted;
            }
         }
         return std::nullopt;
      });

      addConversion([this](FunctionType type) -> Type {
         SmallVector<Type> inputs, results;
         if (failed(convertTypes(type.getInputs(), inputs)) ||
             failed(convertTypes(type.getResults(), results))) {
            return nullptr;
         }
         return FunctionType::get(type.getContext(), inputs, results);
      });

      addSourceMaterialization([](OpBuilder& builder, Type resultType, ValueRange inputs, Location loc) -> Value {
         return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
      });
      addTargetMaterialization([](OpBuilder& builder, Type resultType, ValueRange inputs, Location loc) -> Value {
         return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
      });
   }

   static Type convertSemiringType(graphalg::SemiringTypeInterface t) {
      auto* ctx = t.getContext();
      if (t == graphalg::SemiringTypes::forBool(ctx)) return IntegerType::get(ctx, 1);
      if (t == graphalg::SemiringTypes::forInt(ctx) ||
          t == graphalg::SemiringTypes::forTropInt(ctx) ||
          t == graphalg::SemiringTypes::forTropMaxInt(ctx)) return IntegerType::get(ctx, 64);
      if (t == graphalg::SemiringTypes::forReal(ctx) ||
          t == graphalg::SemiringTypes::forTropReal(ctx)) return Float64Type::get(ctx);
      return Type();
   }
};

// =============================================================================
// ============================== Conversion State =============================
// =============================================================================

class ConversionState {
   private:
   llvm::DenseMap<Value, MatrixMeta> metaMap;

   public:
   void set(Value origVal, const MatrixMeta& meta) { metaMap[origVal] = meta; }

   MatrixMeta get(Value origVal, MLIRContext* ctx) {
      auto it = metaMap.find(origVal);
      if (it != metaMap.end()) return it->second;

      if (auto matType = llvm::dyn_cast<graphalg::MatrixType>(origVal.getType())) {
         MatrixMeta meta;
         std::string prefix = AttributeGenerator::nextName("stream");

         if (!matType.getRows().isOne()) {
            auto rowDef = createColumnDef(ctx, prefix + "_row", IntegerType::get(ctx, 64));
            meta.rowDef = rowDef;
            meta.row = createColumnRef(rowDef);
         }
         if (!matType.getCols().isOne()) {
            auto colDef = createColumnDef(ctx, prefix + "_col", IntegerType::get(ctx, 64));
            meta.colDef = colDef;
            meta.col = createColumnRef(colDef);
         }

         auto semiringType = llvm::cast<graphalg::SemiringTypeInterface>(matType.getSemiring());
         Type valType = GraphAlgTypeConverter::convertSemiringType(semiringType);
         auto valDef = createColumnDef(ctx, prefix + "_val", valType);
         meta.valDef = valDef;
         meta.val = createColumnRef(valDef);

         metaMap[origVal] = meta;
         return meta;
      }
      return {};
   }
};

// =============================================================================
// ================================= Patterns ==================================
// =============================================================================

template <typename T>
class StatefulConversion : public OpConversionPattern<T> {
   protected:
   ConversionState& state;

   public:
   StatefulConversion(TypeConverter& tc, MLIRContext* ctx, ConversionState& state)
      : OpConversionPattern<T>(tc, ctx), state(state) {}
};

class FuncOpConversion : public StatefulConversion<func::FuncOp> {
   public:
   using StatefulConversion::StatefulConversion;
   LogicalResult matchAndRewrite(func::FuncOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto funcType = op.getFunctionType();
      auto newFuncType = llvm::cast_if_present<FunctionType>(typeConverter->convertType(funcType));
      if (!newFuncType) return failure();

      if (!op.getBody().empty()) {
         for (BlockArgument arg : op.getBody().front().getArguments()) {
            state.get(arg, rewriter.getContext());
         }
      }

      rewriter.modifyOpInPlace(op, [&] { op.setFunctionType(newFuncType); });

      TypeConverter::SignatureConversion signatureConversion(funcType.getNumInputs());
      for (const auto& arg : llvm::enumerate(funcType.getInputs())) {
         signatureConversion.addInputs(arg.index(), typeConverter->convertType(arg.value()));
      }

      if (failed(rewriter.convertRegionTypes(&op.getBody(), *typeConverter, &signatureConversion))) {
         return failure();
      }
      return success();
   }
};

class ReturnOpConversion : public OpConversionPattern<func::ReturnOp> {
   public:
   using OpConversionPattern::OpConversionPattern;
   LogicalResult matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
      return success();
   }
};

class ConstantMatrixConversion : public StatefulConversion<graphalg::ConstantMatrixOp> {
   public:
   using StatefulConversion::StatefulConversion;
   LogicalResult matchAndRewrite(graphalg::ConstantMatrixOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto* ctx = rewriter.getContext();
      auto semiringType = llvm::cast<graphalg::SemiringTypeInterface>(op.getType().getSemiring());
      Type valType = GraphAlgTypeConverter::convertSemiringType(semiringType);

      auto valDef = createColumnDef(ctx, AttributeGenerator::nextName("const_val"), valType);

      Attribute rawVal = op.getValue();
      if (auto tropInt = llvm::dyn_cast<graphalg::TropIntAttr>(rawVal)) rawVal = tropInt.getValue();
      if (auto tropF = llvm::dyn_cast<graphalg::TropFloatAttr>(rawVal)) rawVal = tropF.getValue();

      auto constRel = rewriter.create<relalg::ConstRelationOp>(
         op.getLoc(),
         ArrayAttr::get(ctx, {valDef}),
         ArrayAttr::get(ctx, {ArrayAttr::get(ctx, {rawVal})}));

      MatrixMeta meta;
      meta.valDef = valDef;
      meta.val = createColumnRef(valDef);

      state.set(op.getResult(), meta);
      rewriter.replaceOp(op, constRel.getResult());
      return success();
   }
};

class ApplyOpConversion : public StatefulConversion<graphalg::ApplyOp> {
   public:
   using StatefulConversion::StatefulConversion;
   LogicalResult matchAndRewrite(graphalg::ApplyOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (op.getInputs().empty()) return failure();

      auto loc = op.getLoc();
      MatrixMeta currentMeta = state.get(op.getInputs()[0], rewriter.getContext());
      if (!currentMeta.val) return op.emitError("Missing metadata for ApplyOp operand 0"), failure();

      SmallVector<MatrixMeta> inputMetas = {currentMeta};
      Value currentNewRel = adaptor.getInputs()[0];

      for (size_t i = 1; i < op.getInputs().size(); ++i) {
         MatrixMeta nextMeta = state.get(op.getInputs()[i], rewriter.getContext());
         if (!nextMeta.val) return op.emitError("Missing metadata for ApplyOp operand ") << i, failure();

         Value nextNewRel = adaptor.getInputs()[i];
         inputMetas.push_back(nextMeta);

         auto joinOp = rewriter.create<relalg::InnerJoinOp>(loc, currentNewRel, nextNewRel);
         Block* joinBlock = new Block;
         joinOp.getPredicate().push_back(joinBlock);
         auto tupleArg = joinBlock->addArgument(tuples::TupleType::get(rewriter.getContext()), loc);

         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(joinBlock);

         Value cmpAcc;
         auto appendCmp = [&](tuples::ColumnRefAttr a, tuples::ColumnRefAttr b) {
            auto valA = rewriter.create<tuples::GetColumnOp>(loc, a.getColumn().type, a, tupleArg);
            auto valB = rewriter.create<tuples::GetColumnOp>(loc, b.getColumn().type, b, tupleArg);
            Value cmp;
            if (valA.getType().isF64())
               cmp = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, valA, valB);
            else
               cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, valA, valB);

            if (cmpAcc)
               cmpAcc = rewriter.create<arith::AndIOp>(loc, cmpAcc, cmp);
            else
               cmpAcc = cmp;
         };

         if (currentMeta.hasRow() && nextMeta.hasRow()) appendCmp(currentMeta.row, nextMeta.row);
         if (currentMeta.hasCol() && nextMeta.hasCol()) appendCmp(currentMeta.col, nextMeta.col);

         if (!cmpAcc) cmpAcc = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
         rewriter.create<tuples::ReturnOp>(loc, ValueRange{cmpAcc});

         currentNewRel = joinOp.getResult();
         if (!currentMeta.hasRow() && nextMeta.hasRow()) currentMeta.row = nextMeta.row;
         if (!currentMeta.hasCol() && nextMeta.hasCol()) currentMeta.col = nextMeta.col;
      }

      auto mapOp = rewriter.create<relalg::MapOp>(loc, currentNewRel);
      auto semiringType = llvm::cast<graphalg::SemiringTypeInterface>(op.getType().getSemiring());
      Type resType = GraphAlgTypeConverter::convertSemiringType(semiringType);

      auto resDef = createColumnDef(rewriter.getContext(), AttributeGenerator::nextName("apply_res"), resType);
      mapOp.setComputedColsAttr(ArrayAttr::get(rewriter.getContext(), {resDef}));

      {
         Block* mapBlock = new Block;
         mapOp.getPredicate().push_back(mapBlock);
         auto tupleArg = mapBlock->addArgument(tuples::TupleType::get(rewriter.getContext()), loc);

         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(mapBlock);

         SmallVector<Value> newArgs;
         for (size_t i = 0; i < inputMetas.size(); ++i) {
            newArgs.push_back(rewriter.create<tuples::GetColumnOp>(
               loc, inputMetas[i].val.getColumn().type, inputMetas[i].val, tupleArg));
         }

         rewriter.mergeBlocks(&op.getBody().front(), mapBlock, newArgs);
      }

      MatrixMeta resMeta = currentMeta;
      resMeta.valDef = resDef;
      resMeta.val = createColumnRef(resDef);

      state.set(op.getResult(), resMeta);
      rewriter.replaceOp(op, mapOp.getResult());
      return success();
   }
};

class ApplyReturnOpConversion : public OpConversionPattern<graphalg::ApplyReturnOp> {
   public:
   using OpConversionPattern::OpConversionPattern;
   LogicalResult matchAndRewrite(graphalg::ApplyReturnOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<tuples::ReturnOp>(op, ValueRange{adaptor.getValue()});
      return success();
   }
};

class MxmOpConversion : public ConversionPattern {
   ConversionState& state;

   public:
   MxmOpConversion(TypeConverter& tc, MLIRContext* ctx, ConversionState& state)
      : ConversionPattern(tc, "graphalg.mxm", 1, ctx), state(state) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto lhsMeta = state.get(op->getOperand(0), rewriter.getContext());
      auto rhsMeta = state.get(op->getOperand(1), rewriter.getContext());

      if (!lhsMeta.val || !rhsMeta.val) return op->emitError("Missing metadata for MxmOp");
      if (!lhsMeta.hasCol() || !rhsMeta.hasRow()) return op->emitError("MxmOp requires LHS col and RHS row");

      auto joinOp = rewriter.create<relalg::InnerJoinOp>(op->getLoc(), operands[0], operands[1]);
      {
         Block* predBlock = new Block;
         joinOp.getPredicate().push_back(predBlock);
         auto tupleArg = predBlock->addArgument(tuples::TupleType::get(rewriter.getContext()), op->getLoc());

         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(predBlock);

         auto lhsColVal = rewriter.create<tuples::GetColumnOp>(op->getLoc(), lhsMeta.col.getColumn().type, lhsMeta.col, tupleArg);
         auto rhsRowVal = rewriter.create<tuples::GetColumnOp>(op->getLoc(), rhsMeta.row.getColumn().type, rhsMeta.row, tupleArg);

         Value cmp = rewriter.create<arith::CmpIOp>(op->getLoc(), arith::CmpIPredicate::eq, lhsColVal, rhsRowVal);
         rewriter.create<tuples::ReturnOp>(op->getLoc(), ValueRange{cmp});
      }

      auto mapOp = rewriter.create<relalg::MapOp>(op->getLoc(), joinOp.getResult());
      auto matrixType = llvm::cast<graphalg::MatrixType>(op->getResultTypes()[0]);
      auto semiringType = llvm::cast<graphalg::SemiringTypeInterface>(matrixType.getSemiring());
      Type resType = GraphAlgTypeConverter::convertSemiringType(semiringType);

      auto newValDef = createColumnDef(rewriter.getContext(), AttributeGenerator::nextName("mul_res"), resType);
      mapOp.setComputedColsAttr(ArrayAttr::get(rewriter.getContext(), {newValDef}));

      {
         Block* mapBlock = new Block;
         mapOp.getPredicate().push_back(mapBlock);
         auto tupleArg = mapBlock->addArgument(tuples::TupleType::get(rewriter.getContext()), op->getLoc());

         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(mapBlock);

         auto lhsVal = rewriter.create<tuples::GetColumnOp>(op->getLoc(), lhsMeta.val.getColumn().type, lhsMeta.val, tupleArg);
         auto rhsVal = rewriter.create<tuples::GetColumnOp>(op->getLoc(), rhsMeta.val.getColumn().type, rhsMeta.val, tupleArg);

         Value res;
         if (isTropical(semiringType)) {
            if (resType.isF64())
               res = rewriter.create<arith::AddFOp>(op->getLoc(), lhsVal, rhsVal);
            else
               res = rewriter.create<arith::AddIOp>(op->getLoc(), lhsVal, rhsVal);
         } else {
            if (resType.isF64())
               res = rewriter.create<arith::MulFOp>(op->getLoc(), lhsVal, rhsVal);
            else
               res = rewriter.create<arith::MulIOp>(op->getLoc(), lhsVal, rhsVal);
         }

         rewriter.create<tuples::ReturnOp>(op->getLoc(), ValueRange{res});
      }

      MatrixMeta resMeta;
      if (lhsMeta.hasRow()) resMeta.row = lhsMeta.row;
      if (rhsMeta.hasCol()) resMeta.col = rhsMeta.col;
      resMeta.valDef = newValDef;
      resMeta.val = createColumnRef(newValDef);

      state.set(op->getResult(0), resMeta);
      rewriter.replaceOp(op, mapOp.getResult());
      return success();
   }
};

class ForDimOpConversion : public ConversionPattern {
   ConversionState& state;

   public:
   ForDimOpConversion(TypeConverter& tc, MLIRContext* ctx, ConversionState& state)
      : ConversionPattern(tc, "graphalg.for_dim", 1, ctx), state(state) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      auto ctx = rewriter.getContext();

      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

      Block* oldBody = &op->getRegion(0).front();

      SmallVector<MatrixMeta> blockArgMetas;
      SmallVector<Value> renamedInitArgs;

      for (size_t i = 0; i < op->getNumOperands(); ++i) {
         Value oldInit = op->getOperand(i);
         Value newInit = operands[i];
         MatrixMeta initMeta = state.get(oldInit, ctx);

         Value oldBlockArg = oldBody->getArgument(i + 1);
         MatrixMeta blockArgMeta = state.get(oldBlockArg, ctx);
         blockArgMetas.push_back(blockArgMeta);

         SmallVector<Attribute> renameDefs;
         if (blockArgMeta.hasRow() && initMeta.hasRow()) {
            renameDefs.push_back(tuples::ColumnDefAttr::get(
               ctx, blockArgMeta.rowDef.getName(), blockArgMeta.rowDef.getColumnPtr(),
               ArrayAttr::get(ctx, {initMeta.row})));
         }
         if (blockArgMeta.hasCol() && initMeta.hasCol()) {
            renameDefs.push_back(tuples::ColumnDefAttr::get(
               ctx, blockArgMeta.colDef.getName(), blockArgMeta.colDef.getColumnPtr(),
               ArrayAttr::get(ctx, {initMeta.col})));
         }
         if (blockArgMeta.val && initMeta.val) {
            renameDefs.push_back(tuples::ColumnDefAttr::get(
               ctx, blockArgMeta.valDef.getName(), blockArgMeta.valDef.getColumnPtr(),
               ArrayAttr::get(ctx, {initMeta.val})));
         }

         if (!renameDefs.empty()) {
            auto renameOp = rewriter.create<relalg::RenamingOp>(
               loc, tuples::TupleStreamType::get(ctx), newInit,
               ArrayAttr::get(ctx, renameDefs));
            renamedInitArgs.push_back(renameOp.getResult());
         } else {
            renamedInitArgs.push_back(newInit);
         }

         state.set(op->getResult(i), blockArgMeta);
      }

      auto forOp = rewriter.create<scf::ForOp>(loc, zero, one, step, renamedInitArgs);
      Block* forBody = forOp.getBody();

      auto dummyDef = createColumnDef(ctx, AttributeGenerator::nextName("dummy_idx"), rewriter.getI64Type());
      auto constRel = rewriter.create<relalg::ConstRelationOp>(
         loc,
         ArrayAttr::get(ctx, {dummyDef}),
         ArrayAttr::get(ctx, {ArrayAttr::get(ctx, {rewriter.getI64IntegerAttr(0)})}));

      MatrixMeta dummyMeta;
      dummyMeta.valDef = dummyDef;
      dummyMeta.val = createColumnRef(dummyDef);
      state.set(oldBody->getArgument(0), dummyMeta);

      SmallVector<Value> replacedArgs;
      replacedArgs.push_back(constRel.getResult());
      for (size_t i = 0; i < renamedInitArgs.size(); ++i) {
         replacedArgs.push_back(forBody->getArgument(i + 1));
      }

      rewriter.mergeBlocks(oldBody, forBody, replacedArgs);

      for (size_t i = 0; i < renamedInitArgs.size(); ++i) {
         state.set(forBody->getArgument(i + 1), blockArgMetas[i]);
      }

      rewriter.replaceOp(op, forOp.getResults());
      return success();
   }
};

class YieldOpConversion : public ConversionPattern {
   ConversionState& state;

   public:
   YieldOpConversion(TypeConverter& tc, MLIRContext* ctx, ConversionState& state)
      : ConversionPattern(tc, "graphalg.yield", 1, ctx), state(state) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      auto ctx = rewriter.getContext();

      SmallVector<Value> renamedYields;
      for (size_t i = 0; i < operands.size(); ++i) {
         Value newYield = operands[i];
         Value oldYield = op->getOperand(i);
         MatrixMeta yieldMeta = state.get(oldYield, ctx);

         Value newBlockArg = op->getBlock()->getArgument(i + 1);
         MatrixMeta blockArgMeta = state.get(newBlockArg, ctx);

         SmallVector<Attribute> renameDefs;
         if (blockArgMeta.hasRow() && yieldMeta.hasRow()) {
            renameDefs.push_back(tuples::ColumnDefAttr::get(
               ctx, blockArgMeta.rowDef.getName(), blockArgMeta.rowDef.getColumnPtr(),
               ArrayAttr::get(ctx, {yieldMeta.row})));
         }
         if (blockArgMeta.hasCol() && yieldMeta.hasCol()) {
            renameDefs.push_back(tuples::ColumnDefAttr::get(
               ctx, blockArgMeta.colDef.getName(), blockArgMeta.colDef.getColumnPtr(),
               ArrayAttr::get(ctx, {yieldMeta.col})));
         }
         if (blockArgMeta.val && yieldMeta.val) {
            renameDefs.push_back(tuples::ColumnDefAttr::get(
               ctx, blockArgMeta.valDef.getName(), blockArgMeta.valDef.getColumnPtr(),
               ArrayAttr::get(ctx, {yieldMeta.val})));
         }

         if (!renameDefs.empty()) {
            auto renameOp = rewriter.create<relalg::RenamingOp>(
               loc, tuples::TupleStreamType::get(ctx), newYield,
               ArrayAttr::get(ctx, renameDefs));
            renamedYields.push_back(renameOp.getResult());
         } else {
            renamedYields.push_back(newYield);
         }
      }

      rewriter.replaceOpWithNewOp<scf::YieldOp>(op, renamedYields);
      return success();
   }
};

class DeferredReduceConversion : public StatefulConversion<graphalg::DeferredReduceOp> {
   public:
   using StatefulConversion::StatefulConversion;

   static relalg::AggrFunc getAggrFuncForSemiring(Type semiringType) {
      if (isTropical(semiringType)) return relalg::AggrFunc::min;
      if (isTropicalMax(semiringType)) return relalg::AggrFunc::max;
      if (isBool(semiringType)) return relalg::AggrFunc::max;
      return relalg::AggrFunc::sum;
   }

   LogicalResult matchAndRewrite(graphalg::DeferredReduceOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value inputRel = adaptor.getInputs()[0];
      auto inputMeta = state.get(op.getInputs()[0], rewriter.getContext());

      auto resMatrixType = op.getType();
      auto semiringType = llvm::cast<graphalg::SemiringTypeInterface>(resMatrixType.getSemiring());

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

      Type valType = GraphAlgTypeConverter::convertSemiringType(semiringType);
      auto aggDefAttr = createColumnDef(rewriter.getContext(), AttributeGenerator::nextName("agg_val"), valType);
      resMeta.valDef = aggDefAttr;
      resMeta.val = createColumnRef(aggDefAttr);

      auto aggOp = rewriter.create<relalg::AggregationOp>(
         op.getLoc(),
         tuples::TupleStreamType::get(rewriter.getContext()),
         inputRel,
         ArrayAttr::get(rewriter.getContext(), groupByAttrs),
         ArrayAttr::get(rewriter.getContext(), {aggDefAttr}));

      Block* aggBlock = rewriter.createBlock(&aggOp.getAggrFunc());
      Value groupStream = aggBlock->addArgument(tuples::TupleStreamType::get(rewriter.getContext()), op.getLoc());

      {
         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(aggBlock);

         relalg::AggrFunc func = getAggrFuncForSemiring(semiringType);
         Value res = rewriter.create<relalg::AggrFuncOp>(
            op.getLoc(), valType, relalg::AggrFuncAttr::get(rewriter.getContext(), func), groupStream, inputMeta.val);

         rewriter.create<tuples::ReturnOp>(op.getLoc(), ValueRange{res});
      }

      state.set(op.getResult(), resMeta);
      rewriter.replaceOp(op, aggOp.getResult());
      return success();
   }
};

class TransposeOpConversion : public StatefulConversion<graphalg::TransposeOp> {
   public:
   using StatefulConversion::StatefulConversion;
   LogicalResult matchAndRewrite(graphalg::TransposeOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto info = state.get(op.getInput(), rewriter.getContext());
      if (!info.hasRow() || !info.hasCol()) {
         state.set(op.getResult(), info);
         rewriter.replaceOp(op, adaptor.getInput());
         return success();
      }

      auto ctx = rewriter.getContext();
      auto newRowDef = createColumnDef(ctx, AttributeGenerator::nextName("trans_row"), IntegerType::get(ctx, 64));
      auto newColDef = createColumnDef(ctx, AttributeGenerator::nextName("trans_col"), IntegerType::get(ctx, 64));
      auto newValDef = createColumnDef(ctx, AttributeGenerator::nextName("trans_val"), info.val.getColumn().type);

      auto mapOp = rewriter.create<relalg::MapOp>(op.getLoc(), adaptor.getInput());
      mapOp.setComputedColsAttr(ArrayAttr::get(ctx, {newRowDef, newColDef, newValDef}));

      Block* mapBlock = new Block;
      mapOp.getPredicate().push_back(mapBlock);
      auto tupleArg = mapBlock->addArgument(tuples::TupleType::get(ctx), op.getLoc());

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mapBlock);

      auto colVal = rewriter.create<tuples::GetColumnOp>(op.getLoc(), IntegerType::get(ctx, 64), info.col, tupleArg);
      auto rowVal = rewriter.create<tuples::GetColumnOp>(op.getLoc(), IntegerType::get(ctx, 64), info.row, tupleArg);
      auto valVal = rewriter.create<tuples::GetColumnOp>(op.getLoc(), info.val.getColumn().type, info.val, tupleArg);

      rewriter.create<tuples::ReturnOp>(op.getLoc(), ValueRange{colVal.getResult(), rowVal.getResult(), valVal.getResult()});

      MatrixMeta resMeta;
      resMeta.rowDef = newRowDef;
      resMeta.colDef = newColDef;
      resMeta.valDef = newValDef;
      resMeta.row = createColumnRef(newRowDef);
      resMeta.col = createColumnRef(newColDef);
      resMeta.val = createColumnRef(newValDef);

      state.set(op.getResult(), resMeta);
      rewriter.replaceOp(op, mapOp.getResult());
      return success();
   }
};

class CastOpConversion : public ConversionPattern {
   ConversionState& state;

   public:
   CastOpConversion(TypeConverter& tc, MLIRContext* ctx, ConversionState& state)
      : ConversionPattern(tc, "graphalg.cast", 1, ctx), state(state) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      auto ctx = rewriter.getContext();
      auto inputMeta = state.get(op->getOperand(0), ctx);

      if (!inputMeta.val) return op->emitError("Missing metadata for cast"), failure();

      auto matrixType = llvm::cast<graphalg::MatrixType>(op->getResultTypes()[0]);
      auto semiringType = llvm::cast<graphalg::SemiringTypeInterface>(matrixType.getSemiring());
      Type resType = GraphAlgTypeConverter::convertSemiringType(semiringType);

      auto newValDef = createColumnDef(ctx, AttributeGenerator::nextName("cast_val"), resType);

      auto mapOp = rewriter.create<relalg::MapOp>(loc, operands[0]);
      mapOp.setComputedColsAttr(ArrayAttr::get(ctx, {newValDef}));

      Block* mapBlock = new Block;
      mapOp.getPredicate().push_back(mapBlock);
      auto tupleArg = mapBlock->addArgument(tuples::TupleType::get(ctx), loc);

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mapBlock);

      auto val = rewriter.create<tuples::GetColumnOp>(loc, inputMeta.val.getColumn().type, inputMeta.val, tupleArg);
      Value castedVal;

      Type inType = val.getType();
      if (inType == resType) {
         castedVal = val;
      } else if (inType.isInteger(1)) {
         if (resType.isInteger(64)) {
            castedVal = rewriter.create<arith::ExtUIOp>(loc, resType, val);
         } else if (resType.isF64()) {
            auto ext = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), val);
            castedVal = rewriter.create<arith::UIToFPOp>(loc, resType, ext);
         }
      } else if (resType.isInteger(1)) {
         auto zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(inType));
         if (inType.isF64()) {
            castedVal = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ONE, val, zero);
         } else {
            castedVal = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, val, zero);
         }
      } else if (inType.isInteger(64) && resType.isF64()) {
         castedVal = rewriter.create<arith::SIToFPOp>(loc, resType, val);
      } else if (inType.isF64() && resType.isInteger(64)) {
         castedVal = rewriter.create<arith::FPToSIOp>(loc, resType, val);
      } else {
         return op->emitError("Unsupported cast"), failure();
      }

      rewriter.create<tuples::ReturnOp>(loc, ValueRange{castedVal});

      MatrixMeta resMeta = inputMeta;
      resMeta.valDef = newValDef;
      resMeta.val = createColumnRef(newValDef);
      state.set(op->getResult(0), resMeta);

      rewriter.replaceOp(op, mapOp.getResult());
      return success();
   }
};

class AddConversion : public OpConversionPattern<graphalg::AddOp> {
   public:
   using OpConversionPattern::OpConversionPattern;
   LogicalResult matchAndRewrite(graphalg::AddOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto sring = op.getType();
      auto* ctx = rewriter.getContext();
      if (sring == graphalg::SemiringTypes::forBool(ctx)) {
         rewriter.replaceOpWithNewOp<arith::OrIOp>(op, adaptor.getLhs(), adaptor.getRhs());
      } else if (sring == graphalg::SemiringTypes::forInt(ctx)) {
         rewriter.replaceOpWithNewOp<arith::AddIOp>(op, adaptor.getLhs(), adaptor.getRhs());
      } else if (sring == graphalg::SemiringTypes::forReal(ctx)) {
         rewriter.replaceOpWithNewOp<arith::AddFOp>(op, adaptor.getLhs(), adaptor.getRhs());
      } else if (sring == graphalg::SemiringTypes::forTropInt(ctx)) {
         rewriter.replaceOpWithNewOp<arith::MinSIOp>(op, adaptor.getLhs(), adaptor.getRhs());
      } else if (sring == graphalg::SemiringTypes::forTropReal(ctx)) {
         rewriter.replaceOpWithNewOp<arith::MinimumFOp>(op, adaptor.getLhs(), adaptor.getRhs());
      } else if (sring == graphalg::SemiringTypes::forTropMaxInt(ctx)) {
         rewriter.replaceOpWithNewOp<arith::MaxSIOp>(op, adaptor.getLhs(), adaptor.getRhs());
      }
      return success();
   }
};

class CastScalarOpConversion : public OpConversionPattern<graphalg::CastScalarOp> {
   public:
   using OpConversionPattern::OpConversionPattern;
   LogicalResult matchAndRewrite(graphalg::CastScalarOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto outType = GraphAlgTypeConverter::convertSemiringType(llvm::cast<graphalg::SemiringTypeInterface>(op.getType()));
      auto inType = adaptor.getInput().getType();

      if (inType == outType) {
         rewriter.replaceOp(op, adaptor.getInput());
         return success();
      }

      if (inType.isInteger(1)) {
         if (outType.isInteger(64)) {
            rewriter.replaceOpWithNewOp<arith::ExtUIOp>(op, outType, adaptor.getInput());
         } else if (outType.isF64()) {
            auto ext = rewriter.create<arith::ExtUIOp>(op.getLoc(), rewriter.getI64Type(), adaptor.getInput());
            rewriter.replaceOpWithNewOp<arith::UIToFPOp>(op, outType, ext);
         }
         return success();
      }

      return failure();
   }
};

// =============================================================================
// =============================== Pass Definition =============================
// =============================================================================

class GraphAlgToRelAlgPass : public PassWrapper<GraphAlgToRelAlgPass, OperationPass<ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GraphAlgToRelAlgPass)

   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<
         relalg::RelAlgDialect,
         tuples::TupleStreamDialect,
         db::DBDialect,
         scf::SCFDialect,
         arith::ArithDialect>();
   }

   void runOnOperation() final;

   virtual llvm::StringRef getArgument() const override { return "graphalg-core-to-relalg"; }
};

void GraphAlgToRelAlgPass::runOnOperation() {
   MLIRContext* context = &getContext();
   GraphAlgTypeConverter typeConverter(context);
   ConversionState state;

   ConversionTarget target(*context);
   target.addLegalOp<ModuleOp>();
   target.addLegalDialect<BuiltinDialect>();
   target.addLegalDialect<gpu::GPUDialect>();
   target.addLegalDialect<async::AsyncDialect>();
   target.addLegalDialect<relalg::RelAlgDialect>();
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

   target.addIllegalDialect<GraphAlgDialect>();

   target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
   });
   target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return typeConverter.isLegal(op.getOperandTypes());
   });

   RewritePatternSet patterns(context);

   patterns.add<ReturnOpConversion>(typeConverter, context);

   patterns.add<
      FuncOpConversion,
      ConstantMatrixConversion,
      ApplyOpConversion,
      DeferredReduceConversion,
      TransposeOpConversion>(typeConverter, context, state);

   patterns.add<
      CastOpConversion,
      MxmOpConversion,
      ForDimOpConversion,
      YieldOpConversion>(typeConverter, context, state);

   patterns.add<
      ApplyReturnOpConversion,
      AddConversion,
      CastScalarOpConversion>(typeConverter, context);

   if (failed(applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
   }
}

std::unique_ptr<OperationPass<ModuleOp>> createGraphAlgToRelAlgPass() {
   return std::make_unique<GraphAlgToRelAlgPass>();
}

} // namespace graphalg