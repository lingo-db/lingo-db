#include <array>
#include <atomic>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "lingodb/compiler/Dialect/Arrow/IR/ArrowDialect.h"
#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/TupleStream/Column.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"

#include "lingodb/compiler/Dialect/graphalg/GraphAlgAttr.h"
#include "lingodb/compiler/Dialect/graphalg/GraphAlgDialect.h"
#include "lingodb/compiler/Dialect/graphalg/GraphAlgOps.h"
#include "lingodb/compiler/Dialect/graphalg/GraphAlgTypes.h"
#include "lingodb/compiler/Dialect/graphalg/SemiringTypes.h"

namespace graphalg {

#define GEN_PASS_DEF_GRAPHALGTORELALG
#include "lingodb/compiler/Dialect/graphalg/GraphAlgPasses.h.inc"

using namespace mlir;
using namespace lingodb::compiler::dialect;

static bool isBool(Type t) {
   return t.isInteger(1) || t == SemiringTypes::forBool(t.getContext());
}

static bool isTropicalNonMax(Type t) {
   return llvm::isa<TropI64Type, TropF64Type>(t);
}

static bool isTropicalMax(Type t) {
   return t == SemiringTypes::forTropMaxInt(t.getContext());
}

static bool isTropical(Type t) {
   return isTropicalNonMax(t) || isTropicalMax(t);
}

static tuples::ColumnDefAttr createColumnDef(MLIRContext* ctx, StringRef name, Type type) {
   auto column = std::make_shared<tuples::Column>();
   column->type = type;
   auto symName = SymbolRefAttr::get(ctx, "graphalg", {FlatSymbolRefAttr::get(ctx, name)});
   return tuples::ColumnDefAttr::get(ctx, symName, column, Attribute());
}

static tuples::ColumnRefAttr createColumnRef(tuples::ColumnDefAttr def) {
   return tuples::ColumnRefAttr::get(def.getContext(), def.getName(), def.getColumnPtr());
}

struct AttributeGenerator {
   static std::string nextName(const std::string& prefix = "attr") {
      static std::atomic<size_t> counter{0};
      return prefix + "_" + std::to_string(counter.fetch_add(1));
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

static std::pair<MatrixMeta, Value> renameMeta(OpBuilder& rewriter, Location loc, MatrixMeta meta, StringRef prefix, Value rel, Type valType) {
   SmallVector<Attribute> renameDefs;
   MatrixMeta renamedMeta;
   auto* ctx = rewriter.getContext();

   if (meta.hasRow()) {
      auto newDef = createColumnDef(ctx, AttributeGenerator::nextName(prefix.str() + "_row"), rewriter.getI64Type());
      renameDefs.push_back(tuples::ColumnDefAttr::get(ctx, newDef.getName(), newDef.getColumnPtr(), ArrayAttr::get(ctx, {meta.row})));
      renamedMeta.rowDef = newDef;
      renamedMeta.row = createColumnRef(newDef);
   }
   if (meta.hasCol()) {
      auto newDef = createColumnDef(ctx, AttributeGenerator::nextName(prefix.str() + "_col"), rewriter.getI64Type());
      renameDefs.push_back(tuples::ColumnDefAttr::get(ctx, newDef.getName(), newDef.getColumnPtr(), ArrayAttr::get(ctx, {meta.col})));
      renamedMeta.colDef = newDef;
      renamedMeta.col = createColumnRef(newDef);
   }
   if (meta.val) {
      auto newDef = createColumnDef(ctx, AttributeGenerator::nextName(prefix.str() + "_val"), valType);
      renameDefs.push_back(tuples::ColumnDefAttr::get(ctx, newDef.getName(), newDef.getColumnPtr(), ArrayAttr::get(ctx, {meta.val})));
      renamedMeta.valDef = newDef;
      renamedMeta.val = createColumnRef(newDef);
   }
   Value newRel = rel;
   if (!renameDefs.empty()) {
      newRel = rewriter.create<relalg::RenamingOp>(loc, tuples::TupleStreamType::get(ctx), rel, ArrayAttr::get(ctx, renameDefs)).getResult();
   }
   return {renamedMeta, newRel};
}

class GraphAlgTypeConverter : public TypeConverter {
   public:
   GraphAlgTypeConverter(MLIRContext* ctx) {
      addConversion([](Type type) { return type; });

      addConversion([](MatrixType type) -> Type {
         return tuples::TupleStreamType::get(type.getContext());
      });

      addConversion([](Type type) -> std::optional<Type> {
         if (llvm::isa<SemiringTypeInterface>(type)) {
            return convertSemiringType(type);
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

   static Type convertSemiringType(Type t) {
      if (t.isInteger(1) || t.isInteger(64) || t.isF64()) return t;
      if (auto* ctx = t.getContext()) {
         if (t == SemiringTypes::forBool(ctx)) return IntegerType::get(ctx, 1);
         if (t == SemiringTypes::forInt(ctx) ||
             t == SemiringTypes::forTropInt(ctx) ||
             t == SemiringTypes::forTropMaxInt(ctx)) return IntegerType::get(ctx, 64);
         if (t == SemiringTypes::forReal(ctx) ||
             t == SemiringTypes::forTropReal(ctx)) return Float64Type::get(ctx);
      }
      return t;
   }
};

class ConversionState {
   llvm::DenseMap<Value, MatrixMeta> metaMap;

   public:
   void set(Value origVal, const MatrixMeta& meta) { metaMap[origVal] = meta; }

   MatrixMeta get(Value origVal, MLIRContext* ctx) {
      auto it = metaMap.find(origVal);
      if (it != metaMap.end()) return it->second;

      if (auto matType = llvm::dyn_cast<MatrixType>(origVal.getType())) {
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

         Type valType = GraphAlgTypeConverter::convertSemiringType(matType.getSemiring());
         auto valDef = createColumnDef(ctx, prefix + "_val", valType);
         meta.valDef = valDef;
         meta.val = createColumnRef(valDef);

         metaMap[origVal] = meta;
         return meta;
      }
      return {};
   }
};

template <typename T>
class StatefulConversion : public OpConversionPattern<T> {
   protected:
   ConversionState& state;

   public:
   StatefulConversion(TypeConverter& tc, MLIRContext* ctx, ConversionState& state)
      : OpConversionPattern<T>(tc, ctx), state(state) {}
};

class LoadCastConversion : public ConversionPattern {
   ConversionState& state;

   static tuples::ColumnRefAttr resolveColumnRef(Operation* contextOp, Attribute attr) {
      if (auto colRef = llvm::dyn_cast<tuples::ColumnRefAttr>(attr)) return colRef;
      if (auto symRef = llvm::dyn_cast<SymbolRefAttr>(attr)) {
         tuples::ColumnRefAttr resolved;
         contextOp->getParentOfType<ModuleOp>()->walk([&](Operation* child) {
            if (resolved) return WalkResult::interrupt();
            for (auto namedAttr : child->getAttrs()) {
               if (auto arrayAttr = llvm::dyn_cast<ArrayAttr>(namedAttr.getValue())) {
                  for (auto item : arrayAttr) {
                     if (auto def = llvm::dyn_cast<tuples::ColumnDefAttr>(item)) {
                        if (def.getName() == symRef) {
                           resolved = createColumnRef(def);
                           return WalkResult::interrupt();
                        }
                     }
                  }
               }
            }
            return WalkResult::advance();
         });
         return resolved;
      }
      return nullptr;
   }

   public:
   LoadCastConversion(const TypeConverter& tc, MLIRContext* ctx, ConversionState& state)
      : ConversionPattern(tc, "builtin.unrealized_conversion_cast", 1, ctx), state(state) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      if (operands.empty()) return failure();

      // Case 1: TupleStream -> Matrix (Loading data IN)
      if (llvm::isa<MatrixType>(op->getResultTypes()[0])) {
         auto colsAttr = op->getAttrOfType<ArrayAttr>("cols");
         if (!colsAttr) return failure();

         MatrixMeta meta;
         auto matType = llvm::cast<MatrixType>(op->getResultTypes()[0]);
         if (colsAttr.size() == 3) {
            meta.row = resolveColumnRef(op, colsAttr[0]);
            meta.col = resolveColumnRef(op, colsAttr[1]);
            meta.val = resolveColumnRef(op, colsAttr[2]);
         } else if (colsAttr.size() == 2) {
            if (matType.getRows().isOne()) {
               meta.col = resolveColumnRef(op, colsAttr[0]);
               meta.val = resolveColumnRef(op, colsAttr[1]);
            } else {
               meta.row = resolveColumnRef(op, colsAttr[0]);
               meta.val = resolveColumnRef(op, colsAttr[1]);
            }
         } else if (colsAttr.size() == 1) {
            meta.val = resolveColumnRef(op, colsAttr[0]);
         }

         state.set(op->getResult(0), meta);
         rewriter.replaceOp(op, operands[0]);
         return success();
      }

      // Case 2: Matrix -> TupleStream (Returning data OUT for execution)
      if (llvm::isa<tuples::TupleStreamType>(op->getResultTypes()[0])) {
         auto colsAttr = op->getAttrOfType<ArrayAttr>("cols");
         if (!colsAttr) {
            rewriter.replaceOp(op, operands[0]);
            return success();
         }

         auto meta = state.get(op->getOperand(0), rewriter.getContext());
         SmallVector<Attribute> renameDefs;
         auto ctx = rewriter.getContext();
         auto addRename = [&](tuples::ColumnRefAttr exp, tuples::ColumnRefAttr from) {
            if (exp != from) {
               renameDefs.push_back(tuples::ColumnDefAttr::get(ctx, exp.getName(), exp.getColumnPtr(), ArrayAttr::get(ctx, {from})));
            }
         };

         if (colsAttr.size() == 3) {
            auto expRow = resolveColumnRef(op, colsAttr[0]);
            auto expCol = resolveColumnRef(op, colsAttr[1]);
            auto expVal = resolveColumnRef(op, colsAttr[2]);
            if (meta.hasRow()) addRename(expRow, meta.row);
            if (meta.hasCol()) addRename(expCol, meta.col);
            if (meta.val) addRename(expVal, meta.val);
         } else if (colsAttr.size() == 2) {
            if (!meta.hasRow() && meta.hasCol()) {
               auto expCol = resolveColumnRef(op, colsAttr[0]);
               auto expVal = resolveColumnRef(op, colsAttr[1]);
               addRename(expCol, meta.col);
               if (meta.val) addRename(expVal, meta.val);
            } else {
               auto expRow = resolveColumnRef(op, colsAttr[0]);
               auto expVal = resolveColumnRef(op, colsAttr[1]);
               if (meta.hasRow()) addRename(expRow, meta.row);
               if (meta.val) addRename(expVal, meta.val);
            }
         } else if (colsAttr.size() == 1) {
            auto expVal = resolveColumnRef(op, colsAttr[0]);
            if (meta.val) addRename(expVal, meta.val);
         }

         if (!renameDefs.empty()) {
            auto renameOp = rewriter.create<relalg::RenamingOp>(op->getLoc(), tuples::TupleStreamType::get(ctx), operands[0], ArrayAttr::get(ctx, renameDefs));
            rewriter.replaceOp(op, renameOp.getResult());
         } else {
            rewriter.replaceOp(op, operands[0]);
         }
         return success();
      }
      return failure();
   }
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

class ConstantMatrixConversion : public StatefulConversion<ConstantMatrixOp> {
   public:
   using StatefulConversion::StatefulConversion;
   LogicalResult matchAndRewrite(ConstantMatrixOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto* ctx = rewriter.getContext();
      Type valType = GraphAlgTypeConverter::convertSemiringType(op.getType().getSemiring());

      auto valDef = createColumnDef(ctx, AttributeGenerator::nextName("const_val"), valType);

      Attribute rawVal = op.getValue();
      if (auto tropInt = llvm::dyn_cast<TropIntAttr>(rawVal)) rawVal = tropInt.getValue();
      if (auto tropF = llvm::dyn_cast<TropFloatAttr>(rawVal)) rawVal = tropF.getValue();

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

class ApplyOpConversion : public StatefulConversion<ApplyOp> {
   public:
   using StatefulConversion::StatefulConversion;
   LogicalResult matchAndRewrite(ApplyOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (op.getInputs().empty()) return failure();

      auto loc = op.getLoc();
      auto ctx = rewriter.getContext();

      SmallVector<MatrixMeta> inputMetas;
      Value currentRel = adaptor.getInputs()[0];
      MatrixMeta currentMeta = state.get(op.getInputs()[0], ctx);
      inputMetas.push_back(currentMeta);

      for (size_t i = 1; i < op.getInputs().size(); ++i) {
         MatrixMeta meta = state.get(op.getInputs()[i], ctx);
         auto matType = llvm::cast<MatrixType>(op.getInputs()[i].getType());
         Type valType = GraphAlgTypeConverter::convertSemiringType(matType.getSemiring());

         auto [renamedMeta, renamedRel] = renameMeta(rewriter, loc, meta, "apply_rhs", adaptor.getInputs()[i], valType);
         inputMetas.push_back(renamedMeta);

         auto joinOp = rewriter.create<relalg::InnerJoinOp>(loc, currentRel, renamedRel);
         auto joinBlock = new Block;
         joinOp.getPredicate().push_back(joinBlock);
         {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(joinBlock);

            auto tupleArg = joinBlock->addArgument(tuples::TupleType::get(ctx), loc);
            Value cmpAcc;
            if (currentMeta.hasRow() && renamedMeta.hasRow()) {
               auto valA = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), currentMeta.row, tupleArg);
               auto valB = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), renamedMeta.row, tupleArg);
               Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, valA, valB);
               cmpAcc = cmp;
            }
            if (currentMeta.hasCol() && renamedMeta.hasCol()) {
               auto valA = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), currentMeta.col, tupleArg);
               auto valB = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), renamedMeta.col, tupleArg);
               Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, valA, valB);
               if (cmpAcc)
                  cmpAcc = rewriter.create<arith::AndIOp>(loc, cmpAcc, cmp);
               else
                  cmpAcc = cmp;
            }
            if (!cmpAcc) cmpAcc = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
            rewriter.create<tuples::ReturnOp>(loc, ValueRange{cmpAcc});
         }

         currentRel = joinOp.getResult();
         if (!currentMeta.hasRow() && renamedMeta.hasRow()) currentMeta.row = renamedMeta.row;
         if (!currentMeta.hasCol() && renamedMeta.hasCol()) currentMeta.col = renamedMeta.col;
      }

      auto mapOp = rewriter.create<relalg::MapOp>(loc, currentRel);
      Type resType = GraphAlgTypeConverter::convertSemiringType(op.getType().getSemiring());

      auto resDef = createColumnDef(ctx, AttributeGenerator::nextName("apply_res"), resType);
      mapOp.setComputedColsAttr(ArrayAttr::get(ctx, {resDef}));

      {
         auto* mapBlock = new Block;
         mapOp.getPredicate().push_back(mapBlock);
         auto tupleArg = mapBlock->addArgument(tuples::TupleType::get(ctx), loc);

         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(mapBlock);

         SmallVector<Value> newArgs;
         for (size_t i = 0; i < inputMetas.size(); ++i) {
            auto matType = llvm::cast<MatrixType>(op.getInputs()[i].getType());
            Type vType = GraphAlgTypeConverter::convertSemiringType(matType.getSemiring());
            newArgs.push_back(rewriter.create<tuples::GetColumnOp>(loc, vType, inputMetas[i].val, tupleArg));
         }

         rewriter.mergeBlocks(&op.getBody().front(), mapBlock, newArgs);
      }

      MatrixMeta resMeta = currentMeta;
      resMeta.valDef = resDef;
      resMeta.val = createColumnRef(resDef);

      SmallVector<Attribute> keepCols;
      if (resMeta.hasRow()) keepCols.push_back(resMeta.row);
      if (resMeta.hasCol()) keepCols.push_back(resMeta.col);
      if (resMeta.val) keepCols.push_back(resMeta.val);

      auto aggOp = rewriter.create<relalg::AggregationOp>(loc, tuples::TupleStreamType::get(ctx), mapOp.getResult(), ArrayAttr::get(ctx, keepCols), ArrayAttr::get(ctx, {}));

      {
         Block* aggBlock = rewriter.createBlock(&aggOp.getAggrFunc());
         aggBlock->addArgument(tuples::TupleStreamType::get(ctx), loc);
         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(aggBlock);
         rewriter.create<tuples::ReturnOp>(loc, ValueRange{});
      }

      state.set(op->getResult(0), resMeta);
      rewriter.replaceOp(op, aggOp.getResult());
      return success();
   }
};

class ApplyReturnOpConversion : public OpConversionPattern<ApplyReturnOp> {
   public:
   using OpConversionPattern::OpConversionPattern;
   LogicalResult matchAndRewrite(ApplyReturnOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<tuples::ReturnOp>(op, ValueRange{adaptor.getValue()});
      return success();
   }
};

class MatMulOpConversion : public ConversionPattern {
   ConversionState& state;

   public:
   MatMulOpConversion(const TypeConverter& tc, MLIRContext* ctx, ConversionState& state)
      : ConversionPattern(tc, "graphalg.mxm", 1, ctx), state(state) {}

   static relalg::AggrFunc getAggrFuncForSemiring(Type semiringType) {
      if (isTropicalNonMax(semiringType)) return relalg::AggrFunc::min;
      if (isTropicalMax(semiringType)) return relalg::AggrFunc::max;
      if (isBool(semiringType)) return relalg::AggrFunc::max;
      return relalg::AggrFunc::sum;
   }

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto lhsMeta = state.get(op->getOperand(0), rewriter.getContext());
      auto rhsMeta = state.get(op->getOperand(1), rewriter.getContext());

      if (!lhsMeta.val || !rhsMeta.val) return op->emitError("Missing metadata for MxmOp");

      auto ctx = rewriter.getContext();
      auto loc = op->getLoc();

      auto rhsMatType = llvm::cast<MatrixType>(op->getOperand(1).getType());
      Type rhsValType = GraphAlgTypeConverter::convertSemiringType(rhsMatType.getSemiring());

      auto isConflict = [&](tuples::ColumnRefAttr c) {
         if (!c) return false;
         return c == lhsMeta.row || c == lhsMeta.col || c == lhsMeta.val;
      };

      SmallVector<Attribute> renameDefs;
      MatrixMeta renamedRhsMeta = rhsMeta;
      StringRef prefix = "rhs";

      if (rhsMeta.hasRow() && isConflict(rhsMeta.row)) {
         auto newDef = createColumnDef(ctx, AttributeGenerator::nextName(prefix.str() + "_row"), rewriter.getI64Type());
         renameDefs.push_back(tuples::ColumnDefAttr::get(ctx, newDef.getName(), newDef.getColumnPtr(), ArrayAttr::get(ctx, {rhsMeta.row})));
         renamedRhsMeta.rowDef = newDef;
         renamedRhsMeta.row = createColumnRef(newDef);
      }
      if (rhsMeta.hasCol() && isConflict(rhsMeta.col)) {
         auto newDef = createColumnDef(ctx, AttributeGenerator::nextName(prefix.str() + "_col"), rewriter.getI64Type());
         renameDefs.push_back(tuples::ColumnDefAttr::get(ctx, newDef.getName(), newDef.getColumnPtr(), ArrayAttr::get(ctx, {rhsMeta.col})));
         renamedRhsMeta.colDef = newDef;
         renamedRhsMeta.col = createColumnRef(newDef);
      }
      if (rhsMeta.val && isConflict(rhsMeta.val)) {
         auto newDef = createColumnDef(ctx, AttributeGenerator::nextName(prefix.str() + "_val"), rhsValType);
         renameDefs.push_back(tuples::ColumnDefAttr::get(ctx, newDef.getName(), newDef.getColumnPtr(), ArrayAttr::get(ctx, {rhsMeta.val})));
         renamedRhsMeta.valDef = newDef;
         renamedRhsMeta.val = createColumnRef(newDef);
      }

      Value rhsRel = operands[1];
      if (!renameDefs.empty()) {
         rhsRel = rewriter.create<relalg::RenamingOp>(loc, tuples::TupleStreamType::get(ctx), rhsRel, ArrayAttr::get(ctx, renameDefs)).getResult();
      }

      auto joinOp = rewriter.create<relalg::InnerJoinOp>(loc, operands[0], rhsRel);
      {
         auto* predBlock = new Block;
         joinOp.getPredicate().push_back(predBlock);
         auto tupleArg = predBlock->addArgument(tuples::TupleType::get(ctx), loc);

         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(predBlock);

         Value cmp;
         if (lhsMeta.hasCol() && renamedRhsMeta.hasRow()) {
            auto lhsColVal = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), lhsMeta.col, tupleArg);
            auto rhsRowVal = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), renamedRhsMeta.row, tupleArg);
            cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lhsColVal, rhsRowVal);
         } else {
            cmp = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
         }
         rewriter.create<tuples::ReturnOp>(loc, ValueRange{cmp});
      }

      auto mapOp = rewriter.create<relalg::MapOp>(loc, joinOp.getResult());
      auto matrixType = llvm::cast<MatrixType>(op->getResultTypes()[0]);
      Type resType = GraphAlgTypeConverter::convertSemiringType(matrixType.getSemiring());

      auto mulValDef = createColumnDef(ctx, AttributeGenerator::nextName("mul_res"), resType);
      mapOp.setComputedColsAttr(ArrayAttr::get(ctx, {mulValDef}));
      auto mulValRef = createColumnRef(mulValDef);

      {
         auto* mapBlock = new Block;
         mapOp.getPredicate().push_back(mapBlock);
         auto tupleArg = mapBlock->addArgument(tuples::TupleType::get(ctx), loc);

         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(mapBlock);

         auto lhsMatType = llvm::cast<MatrixType>(op->getOperand(0).getType());
         Type lhsValType = GraphAlgTypeConverter::convertSemiringType(lhsMatType.getSemiring());

         auto lhsVal = rewriter.create<tuples::GetColumnOp>(loc, lhsValType, lhsMeta.val, tupleArg);
         auto rhsVal = rewriter.create<tuples::GetColumnOp>(loc, rhsValType, renamedRhsMeta.val, tupleArg);

         Value res;
         if (isTropical(matrixType.getSemiring())) {
            if (resType.isF64())
               res = rewriter.create<arith::AddFOp>(loc, lhsVal, rhsVal);
            else
               res = rewriter.create<arith::AddIOp>(loc, lhsVal, rhsVal);
         } else {
            if (resType.isInteger(1))
               res = rewriter.create<arith::AndIOp>(loc, lhsVal, rhsVal);
            else if (resType.isF64())
               res = rewriter.create<arith::MulFOp>(loc, lhsVal, rhsVal);
            else
               res = rewriter.create<arith::MulIOp>(loc, lhsVal, rhsVal);
         }

         rewriter.create<tuples::ReturnOp>(loc, ValueRange{res});
      }

      bool needsAggregation = lhsMeta.hasCol() && renamedRhsMeta.hasRow();

      MatrixMeta resMeta;
      if (lhsMeta.hasRow()) resMeta.row = lhsMeta.row;
      if (renamedRhsMeta.hasCol()) resMeta.col = renamedRhsMeta.col;

      if (!needsAggregation) {
         resMeta.valDef = mulValDef;
         resMeta.val = mulValRef;
         state.set(op->getResult(0), resMeta);
         rewriter.replaceOp(op, mapOp.getResult());
         return success();
      }

      SmallVector<Attribute> groupByAttrs;
      if (resMeta.hasRow()) groupByAttrs.push_back(resMeta.row);
      if (resMeta.hasCol()) groupByAttrs.push_back(resMeta.col);

      auto aggDefAttr = createColumnDef(ctx, AttributeGenerator::nextName("agg_val"), resType);
      resMeta.valDef = aggDefAttr;
      resMeta.val = createColumnRef(aggDefAttr);

      auto aggOp = rewriter.create<relalg::AggregationOp>(
         loc,
         tuples::TupleStreamType::get(ctx),
         mapOp.getResult(),
         ArrayAttr::get(ctx, groupByAttrs),
         ArrayAttr::get(ctx, {aggDefAttr}));

      {
         Block* aggBlock = rewriter.createBlock(&aggOp.getAggrFunc());
         Value groupStream = aggBlock->addArgument(tuples::TupleStreamType::get(ctx), loc);
         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(aggBlock);

         relalg::AggrFunc func = getAggrFuncForSemiring(matrixType.getSemiring());
         Value res = rewriter.create<relalg::AggrFuncOp>(
            loc, resType, relalg::AggrFuncAttr::get(ctx, func), groupStream, mulValRef);

         rewriter.create<tuples::ReturnOp>(loc, ValueRange{res});
      }

      state.set(op->getResult(0), resMeta);
      rewriter.replaceOp(op, aggOp.getResult());
      return success();
   }
};

// Not explicit lowered or in Core, so maybe unused
class MatMulJoinOpConversion : public ConversionPattern {
   ConversionState& state;

   public:
   MatMulJoinOpConversion(const TypeConverter& tc, MLIRContext* ctx, ConversionState& state)
      : ConversionPattern(tc, "graphalg.mxm_join", 1, ctx), state(state) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto lhsMeta = state.get(op->getOperand(0), rewriter.getContext());
      auto rhsMeta = state.get(op->getOperand(1), rewriter.getContext());

      if (!lhsMeta.val || !rhsMeta.val) return op->emitError("Missing metadata for MatMulJoinOp");

      auto ctx = rewriter.getContext();
      auto loc = op->getLoc();

      auto rhsMatType = llvm::cast<MatrixType>(op->getOperand(1).getType());
      Type rhsValType = GraphAlgTypeConverter::convertSemiringType(rhsMatType.getSemiring());

      auto isConflict = [&](tuples::ColumnRefAttr c) {
         if (!c) return false;
         return c == lhsMeta.row || c == lhsMeta.col || c == lhsMeta.val;
      };

      SmallVector<Attribute> renameDefs;
      MatrixMeta renamedRhsMeta = rhsMeta;
      StringRef prefix = "rhs";

      if (rhsMeta.hasRow() && isConflict(rhsMeta.row)) {
         auto newDef = createColumnDef(ctx, AttributeGenerator::nextName(prefix.str() + "_row"), rewriter.getI64Type());
         renameDefs.push_back(tuples::ColumnDefAttr::get(ctx, newDef.getName(), newDef.getColumnPtr(), ArrayAttr::get(ctx, {rhsMeta.row})));
         renamedRhsMeta.rowDef = newDef;
         renamedRhsMeta.row = createColumnRef(newDef);
      }
      if (rhsMeta.hasCol() && isConflict(rhsMeta.col)) {
         auto newDef = createColumnDef(ctx, AttributeGenerator::nextName(prefix.str() + "_col"), rewriter.getI64Type());
         renameDefs.push_back(tuples::ColumnDefAttr::get(ctx, newDef.getName(), newDef.getColumnPtr(), ArrayAttr::get(ctx, {rhsMeta.col})));
         renamedRhsMeta.colDef = newDef;
         renamedRhsMeta.col = createColumnRef(newDef);
      }
      if (rhsMeta.val && isConflict(rhsMeta.val)) {
         auto newDef = createColumnDef(ctx, AttributeGenerator::nextName(prefix.str() + "_val"), rhsValType);
         renameDefs.push_back(tuples::ColumnDefAttr::get(ctx, newDef.getName(), newDef.getColumnPtr(), ArrayAttr::get(ctx, {rhsMeta.val})));
         renamedRhsMeta.valDef = newDef;
         renamedRhsMeta.val = createColumnRef(newDef);
      }

      Value rhsRel = operands[1];
      if (!renameDefs.empty()) {
         rhsRel = rewriter.create<relalg::RenamingOp>(loc, tuples::TupleStreamType::get(ctx), rhsRel, ArrayAttr::get(ctx, renameDefs)).getResult();
      }

      auto joinOp = rewriter.create<relalg::InnerJoinOp>(loc, operands[0], rhsRel);
      {
         auto* predBlock = new Block;
         joinOp.getPredicate().push_back(predBlock);
         auto tupleArg = predBlock->addArgument(tuples::TupleType::get(ctx), loc);

         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(predBlock);

         Value cmp;
         if (lhsMeta.hasCol() && renamedRhsMeta.hasRow()) {
            auto lhsColVal = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), lhsMeta.col, tupleArg);
            auto rhsRowVal = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), renamedRhsMeta.row, tupleArg);
            cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lhsColVal, rhsRowVal);
         } else {
            cmp = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
         }
         rewriter.create<tuples::ReturnOp>(loc, ValueRange{cmp});
      }

      auto mapOp = rewriter.create<relalg::MapOp>(loc, joinOp.getResult());
      auto matrixType = llvm::cast<MatrixType>(op->getResultTypes()[0]);
      Type resType = GraphAlgTypeConverter::convertSemiringType(matrixType.getSemiring());

      auto mulValDef = createColumnDef(ctx, AttributeGenerator::nextName("mul_res"), resType);
      mapOp.setComputedColsAttr(ArrayAttr::get(ctx, {mulValDef}));
      auto mulValRef = createColumnRef(mulValDef);

      {
         auto* mapBlock = new Block;
         mapOp.getPredicate().push_back(mapBlock);
         auto tupleArg = mapBlock->addArgument(tuples::TupleType::get(ctx), loc);

         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(mapBlock);

         auto lhsMatType = llvm::cast<MatrixType>(op->getOperand(0).getType());
         Type lhsValType = GraphAlgTypeConverter::convertSemiringType(lhsMatType.getSemiring());

         auto lhsVal = rewriter.create<tuples::GetColumnOp>(loc, lhsValType, lhsMeta.val, tupleArg);
         auto rhsVal = rewriter.create<tuples::GetColumnOp>(loc, rhsValType, renamedRhsMeta.val, tupleArg);

         Value res;
         if (isTropical(matrixType.getSemiring())) {
            if (resType.isF64())
               res = rewriter.create<arith::AddFOp>(loc, lhsVal, rhsVal);
            else
               res = rewriter.create<arith::AddIOp>(loc, lhsVal, rhsVal);
         } else {
            if (resType.isInteger(1))
               res = rewriter.create<arith::AndIOp>(loc, lhsVal, rhsVal);
            else if (resType.isF64())
               res = rewriter.create<arith::MulFOp>(loc, lhsVal, rhsVal);
            else
               res = rewriter.create<arith::MulIOp>(loc, lhsVal, rhsVal);
         }

         rewriter.create<tuples::ReturnOp>(loc, ValueRange{res});
      }

      MatrixMeta resMeta;
      if (lhsMeta.hasRow()) resMeta.row = lhsMeta.row;
      if (renamedRhsMeta.hasCol()) resMeta.col = renamedRhsMeta.col;

      resMeta.valDef = mulValDef;
      resMeta.val = mulValRef;
      state.set(op->getResult(0), resMeta);
      rewriter.replaceOp(op, mapOp.getResult());
      return success();
   }
};

class ForDimOpConversion : public ConversionPattern {
   ConversionState& state;

   public:
   ForDimOpConversion(const TypeConverter& tc, MLIRContext* ctx, ConversionState& state)
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

      SmallVector<Value> replacedArgs;
      Type oldArgType = oldBody->getArgument(0).getType();
      if (llvm::isa<MatrixType>(oldArgType)) {
         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(forBody);

         auto dummyDef = createColumnDef(ctx, AttributeGenerator::nextName("dummy_idx"), rewriter.getI64Type());
         auto constRel = rewriter.create<relalg::ConstRelationOp>(
            loc,
            ArrayAttr::get(ctx, {dummyDef}),
            ArrayAttr::get(ctx, {ArrayAttr::get(ctx, {rewriter.getI64IntegerAttr(0)})}));

         auto mapOp = rewriter.create<relalg::MapOp>(loc, constRel.getResult());
         auto actualIdxDef = createColumnDef(ctx, AttributeGenerator::nextName("loop_idx"), rewriter.getI64Type());
         mapOp.setComputedColsAttr(ArrayAttr::get(ctx, {actualIdxDef}));

         auto* mapBlock = new Block;
         mapOp.getPredicate().push_back(mapBlock);
         mapBlock->addArgument(tuples::TupleType::get(ctx), loc);
         {
            OpBuilder::InsertionGuard guardMap(rewriter);
            rewriter.setInsertionPointToStart(mapBlock);
            auto castedIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), forBody->getArgument(0));
            rewriter.create<tuples::ReturnOp>(loc, ValueRange{castedIdx});
         }

         MatrixMeta dummyMeta;
         dummyMeta.valDef = actualIdxDef;
         dummyMeta.val = createColumnRef(actualIdxDef);
         state.set(oldBody->getArgument(0), dummyMeta);

         replacedArgs.push_back(mapOp.getResult());
      } else {
         replacedArgs.push_back(forBody->getArgument(0));
      }

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

class ForConstOpConversion : public ConversionPattern {
   ConversionState& state;

   public:
   ForConstOpConversion(const TypeConverter& tc, MLIRContext* ctx, ConversionState& state)
      : ConversionPattern(tc, "graphalg.for_const", 1, ctx), state(state) {}

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

      SmallVector<Value> replacedArgs;
      Type oldArgType = oldBody->getArgument(0).getType();
      if (llvm::isa<MatrixType>(oldArgType)) {
         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(forBody);

         auto dummyDef = createColumnDef(ctx, AttributeGenerator::nextName("dummy_idx"), rewriter.getI64Type());
         auto constRel = rewriter.create<relalg::ConstRelationOp>(
            loc,
            ArrayAttr::get(ctx, {dummyDef}),
            ArrayAttr::get(ctx, {ArrayAttr::get(ctx, {rewriter.getI64IntegerAttr(0)})}));

         auto mapOp = rewriter.create<relalg::MapOp>(loc, constRel.getResult());
         auto actualIdxDef = createColumnDef(ctx, AttributeGenerator::nextName("loop_idx"), rewriter.getI64Type());
         mapOp.setComputedColsAttr(ArrayAttr::get(ctx, {actualIdxDef}));

         auto* mapBlock = new Block;
         mapOp.getPredicate().push_back(mapBlock);
         mapBlock->addArgument(tuples::TupleType::get(ctx), loc);
         {
            OpBuilder::InsertionGuard guardMap(rewriter);
            rewriter.setInsertionPointToStart(mapBlock);
            auto castedIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), forBody->getArgument(0));
            rewriter.create<tuples::ReturnOp>(loc, ValueRange{castedIdx});
         }

         MatrixMeta dummyMeta;
         dummyMeta.valDef = actualIdxDef;
         dummyMeta.val = createColumnRef(actualIdxDef);
         state.set(oldBody->getArgument(0), dummyMeta);

         replacedArgs.push_back(mapOp.getResult());
      } else {
         replacedArgs.push_back(forBody->getArgument(0));
      }

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
   YieldOpConversion(const TypeConverter& tc, MLIRContext* ctx, ConversionState& state)
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
         if (blockArgMeta.hasRow() && yieldMeta.hasRow() && blockArgMeta.row != yieldMeta.row) {
            renameDefs.push_back(tuples::ColumnDefAttr::get(
               ctx, blockArgMeta.rowDef.getName(), blockArgMeta.rowDef.getColumnPtr(),
               ArrayAttr::get(ctx, {yieldMeta.row})));
         }
         if (blockArgMeta.hasCol() && yieldMeta.hasCol() && blockArgMeta.col != yieldMeta.col) {
            renameDefs.push_back(tuples::ColumnDefAttr::get(
               ctx, blockArgMeta.colDef.getName(), blockArgMeta.colDef.getColumnPtr(),
               ArrayAttr::get(ctx, {yieldMeta.col})));
         }
         if (blockArgMeta.val && yieldMeta.val && blockArgMeta.val != yieldMeta.val) {
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

class DeferredReduceConversion : public StatefulConversion<DeferredReduceOp> {
   public:
   using StatefulConversion::StatefulConversion;

   static relalg::AggrFunc getAggrFuncForSemiring(Type semiringType) {
      if (isTropicalNonMax(semiringType)) return relalg::AggrFunc::min;
      if (isTropicalMax(semiringType)) return relalg::AggrFunc::max;
      if (isBool(semiringType)) return relalg::AggrFunc::max;
      return relalg::AggrFunc::sum;
   }

   LogicalResult matchAndRewrite(DeferredReduceOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value inputRel = adaptor.getInputs()[0];
      auto inputMeta = state.get(op.getInputs()[0], rewriter.getContext());

      auto resMatrixType = op.getType();
      Type valType = GraphAlgTypeConverter::convertSemiringType(resMatrixType.getSemiring());

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

      auto aggDefAttr = createColumnDef(rewriter.getContext(), AttributeGenerator::nextName("agg_val"), valType);
      resMeta.valDef = aggDefAttr;
      resMeta.val = createColumnRef(aggDefAttr);

      auto aggOp = rewriter.create<relalg::AggregationOp>(
         op.getLoc(),
         tuples::TupleStreamType::get(rewriter.getContext()),
         inputRel,
         ArrayAttr::get(rewriter.getContext(), groupByAttrs),
         ArrayAttr::get(rewriter.getContext(), {aggDefAttr}));

      {
         Block* aggBlock = rewriter.createBlock(&aggOp.getAggrFunc());
         Value groupStream = aggBlock->addArgument(tuples::TupleStreamType::get(rewriter.getContext()), op.getLoc());
         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(aggBlock);

         relalg::AggrFunc func = getAggrFuncForSemiring(resMatrixType.getSemiring());
         Value res = rewriter.create<relalg::AggrFuncOp>(
            op.getLoc(), valType, relalg::AggrFuncAttr::get(rewriter.getContext(), func), groupStream, inputMeta.val);

         rewriter.create<tuples::ReturnOp>(op.getLoc(), ValueRange{res});
      }

      state.set(op.getResult(), resMeta);
      rewriter.replaceOp(op, aggOp.getResult());
      return success();
   }
};

class TransposeOpConversion : public StatefulConversion<TransposeOp> {
   public:
   using StatefulConversion::StatefulConversion;
   LogicalResult matchAndRewrite(TransposeOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto info = state.get(op.getInput(), rewriter.getContext());

      MatrixMeta resMeta = info;
      resMeta.row = info.col;
      resMeta.col = info.row;
      resMeta.rowDef = info.colDef;
      resMeta.colDef = info.rowDef;

      state.set(op.getResult(), resMeta);
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
   }
};

class DiagOpConversion : public StatefulConversion<DiagOp> {
   public:
   using StatefulConversion::StatefulConversion;
   LogicalResult matchAndRewrite(DiagOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto info = state.get(op.getInput(), rewriter.getContext());
      auto loc = op.getLoc();
      auto ctx = rewriter.getContext();

      MatrixMeta resMeta = info;
      auto mapOp = rewriter.create<relalg::MapOp>(loc, adaptor.getInput());

      if (info.hasRow() && !info.hasCol()) {
         auto colDef = createColumnDef(ctx, AttributeGenerator::nextName("diag_col"), rewriter.getI64Type());
         mapOp.setComputedColsAttr(ArrayAttr::get(ctx, {colDef}));
         resMeta.colDef = colDef;
         resMeta.col = createColumnRef(colDef);

         auto* mapBlock = new Block;
         mapOp.getPredicate().push_back(mapBlock);
         {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(mapBlock);

            auto tupleArg = mapBlock->addArgument(tuples::TupleType::get(ctx), loc);
            auto rowVal = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), info.row, tupleArg);
            rewriter.create<tuples::ReturnOp>(loc, ValueRange{rowVal});
         }
      } else if (!info.hasRow() && info.hasCol()) {
         auto rowDef = createColumnDef(ctx, AttributeGenerator::nextName("diag_row"), rewriter.getI64Type());
         mapOp.setComputedColsAttr(ArrayAttr::get(ctx, {rowDef}));
         resMeta.rowDef = rowDef;
         resMeta.row = createColumnRef(rowDef);

         auto* mapBlock = new Block;
         mapOp.getPredicate().push_back(mapBlock);

         {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(mapBlock);
            auto tupleArg = mapBlock->addArgument(tuples::TupleType::get(ctx), loc);
            auto colVal = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), info.col, tupleArg);
            rewriter.create<tuples::ReturnOp>(loc, ValueRange{colVal});
         }
      } else {
         rewriter.replaceOp(op, adaptor.getInput());
         return success();
      }

      state.set(op.getResult(), resMeta);
      rewriter.replaceOp(op, mapOp.getResult());
      return success();
   }
};

class BroadcastOpConversion : public StatefulConversion<BroadcastOp> {
   public:
   using StatefulConversion::StatefulConversion;
   LogicalResult matchAndRewrite(BroadcastOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto info = state.get(op.getInput(), rewriter.getContext());
      auto loc = op.getLoc();
      auto ctx = rewriter.getContext();

      auto outType = llvm::cast<MatrixType>(op.getResult().getType());
      Value currentRel = adaptor.getInput();
      MatrixMeta resMeta = info;

      auto addBroadcast = [&](StringRef prefix, auto dimObj, tuples::ColumnRefAttr& outRef, tuples::ColumnDefAttr& outDef) {
         int64_t size = dimObj.getConcreteDim();
         outDef = createColumnDef(ctx, AttributeGenerator::nextName(prefix.str()), rewriter.getI64Type());
         SmallVector<Attribute> rows;
         for (int64_t i = 0; i < size; ++i) {
            rows.push_back(ArrayAttr::get(ctx, {rewriter.getI64IntegerAttr(i)}));
         }
         auto constRel = rewriter.create<relalg::ConstRelationOp>(
            loc, ArrayAttr::get(ctx, {outDef}), ArrayAttr::get(ctx, rows));

         auto joinOp = rewriter.create<relalg::InnerJoinOp>(loc, currentRel, constRel.getResult());
         auto* joinBlock = new Block;
         joinOp.getPredicate().push_back(joinBlock);
         joinBlock->addArgument(tuples::TupleType::get(ctx), loc);
         {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(joinBlock);
            Value trueVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
            rewriter.create<tuples::ReturnOp>(loc, ValueRange{trueVal});
         }
         currentRel = joinOp.getResult();
         outRef = createColumnRef(outDef);
      };

      if (!info.hasRow() && !outType.getRows().isOne()) {
         addBroadcast("broadcast_row", outType.getRows(), resMeta.row, resMeta.rowDef);
      }
      if (!info.hasCol() && !outType.getCols().isOne()) {
         addBroadcast("broadcast_col", outType.getCols(), resMeta.col, resMeta.colDef);
      }

      state.set(op.getResult(), resMeta);
      rewriter.replaceOp(op, currentRel);
      return success();
   }
};

class TrilOpConversion : public StatefulConversion<TrilOp> {
   public:
   using StatefulConversion::StatefulConversion;
   LogicalResult matchAndRewrite(TrilOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto info = state.get(op.getInput(), rewriter.getContext());
      auto loc = op.getLoc();
      auto ctx = rewriter.getContext();

      if (!info.hasRow() || !info.hasCol()) {
         rewriter.replaceOp(op, adaptor.getInput());
         return success();
      }

      auto selOp = rewriter.create<relalg::SelectionOp>(loc, adaptor.getInput());
      auto* selBlock = new Block;
      selOp.getPredicate().push_back(selBlock);
      {
         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(selBlock);

         auto tupleArg = selBlock->addArgument(tuples::TupleType::get(ctx), loc);
         auto colVal = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), info.col, tupleArg);
         auto rowVal = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), info.row, tupleArg);
         auto cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, colVal, rowVal);

         rewriter.create<tuples::ReturnOp>(loc, ValueRange{cmp});
      }

      state.set(op.getResult(), info);
      rewriter.replaceOp(op, selOp.getResult());
      return success();
   }
};

class PickAnyOpConversion : public StatefulConversion<PickAnyOp> {
   public:
   using StatefulConversion::StatefulConversion;
   LogicalResult matchAndRewrite(PickAnyOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto info = state.get(op.getInput(), rewriter.getContext());
      auto loc = op.getLoc();
      auto ctx = rewriter.getContext();
      Type valType = GraphAlgTypeConverter::convertSemiringType(op.getType().getSemiring());
      Type sring = op.getType().getSemiring();

      // 1. Filter out structural zeros (σ_{val != 0})
      auto selOp = rewriter.create<relalg::SelectionOp>(loc, adaptor.getInput());
      auto* selBlock = new Block;
      selOp.getPredicate().push_back(selBlock);

      {
         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(selBlock);

         auto tupleArgSel = selBlock->addArgument(tuples::TupleType::get(ctx), loc);
         auto val = rewriter.create<tuples::GetColumnOp>(loc, valType, info.val, tupleArgSel);

         Value zero;
         if (sring == SemiringTypes::forBool(ctx))
            zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
         else if (sring == SemiringTypes::forInt(ctx))
            zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
         else if (sring == SemiringTypes::forReal(ctx))
            zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(rewriter.getF64Type(), 0.0));
         else if (sring == SemiringTypes::forTropInt(ctx))
            zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(std::numeric_limits<int64_t>::max()));
         else if (sring == SemiringTypes::forTropMaxInt(ctx))
            zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(std::numeric_limits<int64_t>::min()));
         else if (sring == SemiringTypes::forTropReal(ctx))
            zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(rewriter.getF64Type(), std::numeric_limits<double>::infinity()));

         Value cmpZero;
         if (valType.isInteger(1) || valType.isInteger(64)) {
            cmpZero = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, val, zero);
         } else {
            cmpZero = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, val, zero);
         }
         rewriter.create<tuples::ReturnOp>(loc, ValueRange{cmpZero});
      }
      Value selRel = selOp.getResult();

      // 2. Perform Group-By Aggregation on Row computing min(col)
      SmallVector<Attribute> groupByAttrs;
      if (info.hasRow()) groupByAttrs.push_back(info.row);

      auto minColDef = createColumnDef(ctx, AttributeGenerator::nextName("pick_min_col"), rewriter.getI64Type());
      auto minColRef = createColumnRef(minColDef);

      auto aggOp = rewriter.create<relalg::AggregationOp>(
         loc, tuples::TupleStreamType::get(ctx), selRel,
         ArrayAttr::get(ctx, groupByAttrs), ArrayAttr::get(ctx, {minColDef}));

      {
         OpBuilder::InsertionGuard guard(rewriter);
         auto* aggBlock = rewriter.createBlock(&aggOp.getAggrFunc());
         rewriter.setInsertionPointToStart(aggBlock);

         Value groupStream = aggBlock->addArgument(tuples::TupleStreamType::get(ctx), loc);
         Value minCol = rewriter.create<relalg::AggrFuncOp>(
            loc, rewriter.getI64Type(), relalg::AggrFuncAttr::get(ctx, relalg::AggrFunc::min), groupStream, info.col);
         rewriter.create<tuples::ReturnOp>(loc, ValueRange{minCol});
      }
      Value aggRel = aggOp.getResult();

      // 3. For bool, simply Map the constant 1 to a new val column and we're done.
      if (sring == SemiringTypes::forBool(ctx)) {
         auto trueValDef = createColumnDef(ctx, AttributeGenerator::nextName("pick_val_true"), valType);
         auto mapOp = rewriter.create<relalg::MapOp>(loc, aggRel);
         mapOp.setComputedColsAttr(ArrayAttr::get(ctx, {trueValDef}));
         auto* mapBlock = new Block;
         mapOp.getPredicate().push_back(mapBlock);
         mapBlock->addArgument(tuples::TupleType::get(ctx), loc);
         {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(mapBlock);
            Value trueVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
            rewriter.create<tuples::ReturnOp>(loc, ValueRange{trueVal});
         }

         MatrixMeta resMeta;
         resMeta.row = info.row;
         resMeta.rowDef = info.rowDef;
         resMeta.col = minColRef;
         resMeta.colDef = minColDef;
         resMeta.val = createColumnRef(trueValDef);
         resMeta.valDef = trueValDef;

         state.set(op.getResult(), resMeta);
         rewriter.replaceOp(op, mapOp.getResult());
         return success();
      }

      // 4. Renaming grouped row to avoid collision in join for non-bool cases
      Value renamedAggRel = aggRel;
      tuples::ColumnRefAttr aggRowRef = nullptr;
      if (info.hasRow()) {
         auto aggRowDef = createColumnDef(ctx, AttributeGenerator::nextName("pick_agg_row"), rewriter.getI64Type());
         aggRowRef = createColumnRef(aggRowDef);
         auto renameDef = tuples::ColumnDefAttr::get(ctx, aggRowDef.getName(), aggRowDef.getColumnPtr(), ArrayAttr::get(ctx, {info.row}));
         renamedAggRel = rewriter.create<relalg::RenamingOp>(loc, tuples::TupleStreamType::get(ctx), aggRel, ArrayAttr::get(ctx, {renameDef})).getResult();
      }

      // 5. InnerJoin with the original filtered stream to extract the associated original `val`
      auto joinOp = rewriter.create<relalg::InnerJoinOp>(loc, selRel, renamedAggRel);
      auto* joinBlock = new Block;
      joinOp.getPredicate().push_back(joinBlock);
      {
         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(joinBlock);

         auto tupleArgJoin = joinBlock->addArgument(tuples::TupleType::get(ctx), loc);
         Value cmpAcc = nullptr;
         if (info.hasRow()) {
            auto r1 = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), info.row, tupleArgJoin);
            auto r2 = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), aggRowRef, tupleArgJoin);
            cmpAcc = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, r1, r2);
         }

         auto c1 = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), info.col, tupleArgJoin);
         auto c2 = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), minColRef, tupleArgJoin);
         Value cmpCol = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, c1, c2);

         if (cmpAcc)
            cmpAcc = rewriter.create<arith::AndIOp>(loc, cmpAcc, cmpCol);
         else
            cmpAcc = cmpCol;

         rewriter.create<tuples::ReturnOp>(loc, ValueRange{cmpAcc});
      }

      state.set(op.getResult(), info);
      rewriter.replaceOp(op, joinOp.getResult());
      return success();
   }
};

class UnionOpConversion : public StatefulConversion<UnionOp> {
   public:
   using StatefulConversion::StatefulConversion;
   LogicalResult matchAndRewrite(UnionOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op.getLoc();
      auto ctx = rewriter.getContext();

      auto lhsMeta = state.get(op.getInputs()[0], ctx);
      auto rhsMeta = state.get(op.getInputs()[1], ctx);

      SmallVector<Attribute> renameDefs;
      if (lhsMeta.hasRow() && rhsMeta.hasRow() && lhsMeta.row != rhsMeta.row) {
         renameDefs.push_back(tuples::ColumnDefAttr::get(ctx, lhsMeta.rowDef.getName(), lhsMeta.rowDef.getColumnPtr(), ArrayAttr::get(ctx, {rhsMeta.row})));
      }
      if (lhsMeta.hasCol() && rhsMeta.hasCol() && lhsMeta.col != rhsMeta.col) {
         renameDefs.push_back(tuples::ColumnDefAttr::get(ctx, lhsMeta.colDef.getName(), lhsMeta.colDef.getColumnPtr(), ArrayAttr::get(ctx, {rhsMeta.col})));
      }
      if (lhsMeta.val && rhsMeta.val && lhsMeta.val != rhsMeta.val) {
         renameDefs.push_back(tuples::ColumnDefAttr::get(ctx, lhsMeta.valDef.getName(), lhsMeta.valDef.getColumnPtr(), ArrayAttr::get(ctx, {rhsMeta.val})));
      }

      Value rhsRel = adaptor.getInputs()[1];
      if (!renameDefs.empty()) {
         rhsRel = rewriter.create<relalg::RenamingOp>(loc, tuples::TupleStreamType::get(ctx), rhsRel, ArrayAttr::get(ctx, renameDefs)).getResult();
      }

      auto unionOp = rewriter.create<relalg::UnionOp>(loc, tuples::TupleStreamType::get(ctx), ValueRange{adaptor.getInputs()[0], rhsRel});
      state.set(op.getResult(), lhsMeta);
      rewriter.replaceOp(op, unionOp.getResult());
      return success();
   }
};

class AddConversion : public OpConversionPattern<AddOp> {
   public:
   using OpConversionPattern::OpConversionPattern;
   LogicalResult matchAndRewrite(AddOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto sring = op.getType();
      auto* ctx = rewriter.getContext();
      if (sring == SemiringTypes::forBool(ctx)) {
         rewriter.replaceOpWithNewOp<arith::OrIOp>(op, adaptor.getLhs(), adaptor.getRhs());
      } else if (sring == SemiringTypes::forInt(ctx)) {
         rewriter.replaceOpWithNewOp<arith::AddIOp>(op, adaptor.getLhs(), adaptor.getRhs());
      } else if (sring == SemiringTypes::forReal(ctx)) {
         rewriter.replaceOpWithNewOp<arith::AddFOp>(op, adaptor.getLhs(), adaptor.getRhs());
      } else if (sring == SemiringTypes::forTropInt(ctx)) {
         rewriter.replaceOpWithNewOp<arith::MinSIOp>(op, adaptor.getLhs(), adaptor.getRhs());
      } else if (sring == SemiringTypes::forTropReal(ctx)) {
         rewriter.replaceOpWithNewOp<arith::MinimumFOp>(op, adaptor.getLhs(), adaptor.getRhs());
      } else if (sring == SemiringTypes::forTropMaxInt(ctx)) {
         rewriter.replaceOpWithNewOp<arith::MaxSIOp>(op, adaptor.getLhs(), adaptor.getRhs());
      }
      return success();
   }
};

class MulConversion : public OpConversionPattern<MulOp> {
   public:
   using OpConversionPattern::OpConversionPattern;
   LogicalResult matchAndRewrite(MulOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto sring = op.getType();
      auto* ctx = rewriter.getContext();
      if (sring == SemiringTypes::forBool(ctx)) {
         rewriter.replaceOpWithNewOp<arith::AndIOp>(op, adaptor.getLhs(), adaptor.getRhs());
      } else if (sring == SemiringTypes::forInt(ctx)) {
         rewriter.replaceOpWithNewOp<arith::MulIOp>(op, adaptor.getLhs(), adaptor.getRhs());
      } else if (sring == SemiringTypes::forReal(ctx)) {
         rewriter.replaceOpWithNewOp<arith::MulFOp>(op, adaptor.getLhs(), adaptor.getRhs());
      } else if (sring == SemiringTypes::forTropInt(ctx) || sring == SemiringTypes::forTropMaxInt(ctx)) {
         rewriter.replaceOpWithNewOp<arith::AddIOp>(op, adaptor.getLhs(), adaptor.getRhs());
      } else if (sring == SemiringTypes::forTropReal(ctx)) {
         rewriter.replaceOpWithNewOp<arith::AddFOp>(op, adaptor.getLhs(), adaptor.getRhs());
      }
      return success();
   }
};

class EqConversion : public OpConversionPattern<EqOp> {
   public:
   using OpConversionPattern::OpConversionPattern;
   LogicalResult matchAndRewrite(EqOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto type = adaptor.getLhs().getType();
      if (type.isInteger(1) || type.isInteger(64)) {
         rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, arith::CmpIPredicate::eq, adaptor.getLhs(), adaptor.getRhs());
      } else if (type.isF64()) {
         rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, arith::CmpFPredicate::OEQ, adaptor.getLhs(), adaptor.getRhs());
      } else {
         return failure();
      }
      return success();
   }
};

class ConstantOpConversion : public OpConversionPattern<ConstantOp> {
   public:
   using OpConversionPattern::OpConversionPattern;
   LogicalResult matchAndRewrite(ConstantOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto type = GraphAlgTypeConverter::convertSemiringType(op.getType());
      Attribute rawVal = op.getValue();
      if (auto tropInt = llvm::dyn_cast<TropIntAttr>(rawVal)) rawVal = tropInt.getValue();
      if (auto tropF = llvm::dyn_cast<TropFloatAttr>(rawVal)) rawVal = tropF.getValue();

      TypedAttr newAttr;
      if (type.isInteger(1) || type.isInteger(64)) {
         newAttr = rewriter.getIntegerAttr(type, llvm::cast<IntegerAttr>(rawVal).getInt());
      } else if (type.isF64()) {
         newAttr = rewriter.getFloatAttr(type, llvm::cast<FloatAttr>(rawVal).getValueAsDouble());
      } else {
         newAttr = llvm::cast<TypedAttr>(rawVal);
      }

      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, newAttr);
      return success();
   }
};

class CastScalarOpConversion : public OpConversionPattern<CastScalarOp> {
   public:
   using OpConversionPattern::OpConversionPattern;
   LogicalResult matchAndRewrite(CastScalarOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto outRing = op.getType();
      auto inRing = op.getInput().getType();
      auto outType = GraphAlgTypeConverter::convertSemiringType(outRing);
      auto inType = adaptor.getInput().getType();

      auto loc = op.getLoc();
      auto input = adaptor.getInput();
      auto* ctx = rewriter.getContext();

      if (inRing == outRing) {
         rewriter.replaceOp(op, input);
         return success();
      }

      auto getZero = [&](Type ring) -> Value {
         if (ring == SemiringTypes::forBool(ctx)) return rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
         if (ring == SemiringTypes::forInt(ctx)) return rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
         if (ring == SemiringTypes::forReal(ctx)) return rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(rewriter.getF64Type(), 0.0));
         if (ring == SemiringTypes::forTropInt(ctx)) return rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(std::numeric_limits<int64_t>::max()));
         if (ring == SemiringTypes::forTropMaxInt(ctx)) return rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(std::numeric_limits<int64_t>::min()));
         if (ring == SemiringTypes::forTropReal(ctx)) return rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(rewriter.getF64Type(), std::numeric_limits<double>::infinity()));
         return nullptr;
      };

      auto getOne = [&](Type ring) -> Value {
         if (ring == SemiringTypes::forBool(ctx)) return rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
         if (ring == SemiringTypes::forInt(ctx)) return rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
         if (ring == SemiringTypes::forReal(ctx)) return rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(rewriter.getF64Type(), 1.0));
         if (ring == SemiringTypes::forTropInt(ctx) || ring == SemiringTypes::forTropMaxInt(ctx)) return rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
         if (ring == SemiringTypes::forTropReal(ctx)) return rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(rewriter.getF64Type(), 0.0));
         return nullptr;
      };

      if (outRing == SemiringTypes::forBool(ctx)) {
         Value zeroIn = getZero(inRing);
         if (!zeroIn) return failure();
         if (inType.isInteger(64)) {
            rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, arith::CmpIPredicate::ne, input, zeroIn);
         } else if (inType.isF64()) {
            rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, arith::CmpFPredicate::UNE, input, zeroIn);
         }
         return success();
      }

      if (inRing == SemiringTypes::forBool(ctx)) {
         Value oneOut = getOne(outRing);
         Value zeroOut = getZero(outRing);
         if (!oneOut || !zeroOut) return failure();
         rewriter.replaceOpWithNewOp<arith::SelectOp>(op, input, oneOut, zeroOut);
         return success();
      }

      if (inRing == SemiringTypes::forInt(ctx) && outRing == SemiringTypes::forReal(ctx)) {
         rewriter.replaceOpWithNewOp<arith::SIToFPOp>(op, outType, input);
         return success();
      }

      if (inRing == SemiringTypes::forInt(ctx) && outRing == SemiringTypes::forTropInt(ctx)) {
         Value zeroIn = getZero(inRing);
         Value infOut = getZero(outRing);
         Value isZero = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, input, zeroIn);
         rewriter.replaceOpWithNewOp<arith::SelectOp>(op, isZero, infOut, input);
         return success();
      }

      if (inRing == SemiringTypes::forInt(ctx) && outRing == SemiringTypes::forTropMaxInt(ctx)) {
         Value zeroIn = getZero(inRing);
         Value infOut = getZero(outRing);
         Value isZero = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, input, zeroIn);
         rewriter.replaceOpWithNewOp<arith::SelectOp>(op, isZero, infOut, input);
         return success();
      }

      if (inRing == SemiringTypes::forInt(ctx) && outRing == SemiringTypes::forTropReal(ctx)) {
         Value zeroIn = getZero(inRing);
         Value infOut = getZero(outRing);
         Value isZero = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, input, zeroIn);
         Value promoted = rewriter.create<arith::SIToFPOp>(loc, outType, input);
         rewriter.replaceOpWithNewOp<arith::SelectOp>(op, isZero, infOut, promoted);
         return success();
      }

      if (inRing == SemiringTypes::forReal(ctx) && outRing == SemiringTypes::forInt(ctx)) {
         rewriter.replaceOpWithNewOp<arith::FPToSIOp>(op, outType, input);
         return success();
      }

      if (inRing == SemiringTypes::forReal(ctx) && outRing == SemiringTypes::forTropInt(ctx)) {
         Value zeroIn = getZero(inRing);
         Value infOut = getZero(outRing);
         Value isZero = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, input, zeroIn);
         Value promoted = rewriter.create<arith::FPToSIOp>(loc, outType, input);
         rewriter.replaceOpWithNewOp<arith::SelectOp>(op, isZero, infOut, promoted);
         return success();
      }

      if (inRing == SemiringTypes::forReal(ctx) && outRing == SemiringTypes::forTropMaxInt(ctx)) {
         Value zeroIn = getZero(inRing);
         Value infOut = getZero(outRing);
         Value isZero = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, input, zeroIn);
         Value promoted = rewriter.create<arith::FPToSIOp>(loc, outType, input);
         rewriter.replaceOpWithNewOp<arith::SelectOp>(op, isZero, infOut, promoted);
         return success();
      }

      if (inRing == SemiringTypes::forReal(ctx) && outRing == SemiringTypes::forTropReal(ctx)) {
         Value zeroIn = getZero(inRing);
         Value infOut = getZero(outRing);
         Value isZero = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, input, zeroIn);
         rewriter.replaceOpWithNewOp<arith::SelectOp>(op, isZero, infOut, input);
         return success();
      }

      return failure();
   }
};

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

   target.addIllegalOp<UnrealizedConversionCastOp>();

   RewritePatternSet patterns(context);

   patterns.add<ReturnOpConversion>(typeConverter, context);
   patterns.add<LoadCastConversion>(typeConverter, context, state);

   patterns.add<
      FuncOpConversion,
      ConstantMatrixConversion,
      ApplyOpConversion,
      DeferredReduceConversion,
      TransposeOpConversion>(typeConverter, context, state);

   patterns.add<
      MatMulOpConversion,
      MatMulJoinOpConversion,
      ForDimOpConversion,
      ForConstOpConversion,
      YieldOpConversion>(typeConverter, context, state);

   patterns.add<
      DiagOpConversion,
      BroadcastOpConversion,
      TrilOpConversion,
      UnionOpConversion,
      PickAnyOpConversion>(typeConverter, context, state);

   patterns.add<
      ApplyReturnOpConversion,
      AddConversion,
      MulConversion,
      EqConversion,
      ConstantOpConversion,
      CastScalarOpConversion>(typeConverter, context);

   if (failed(applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
   }
}

std::unique_ptr<OperationPass<ModuleOp>> createGraphAlgToRelAlgPass() {
   return std::make_unique<GraphAlgToRelAlgPass>();
}

} // namespace graphalg