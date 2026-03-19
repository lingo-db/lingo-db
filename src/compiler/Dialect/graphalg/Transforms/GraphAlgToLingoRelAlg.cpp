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
#include "mlir/IR/IRMapping.h"
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
         auto addRename = [&](Attribute expAttr, tuples::ColumnRefAttr from) {
            if (!expAttr || !from) return;
            auto exp = resolveColumnRef(op, expAttr);
            if (exp && exp.getColumnPtr()->type == from.getColumnPtr()->type) {
               if (exp != from) {
                  renameDefs.push_back(tuples::ColumnDefAttr::get(ctx, exp.getName(), exp.getColumnPtr(), ArrayAttr::get(ctx, {from})));
               }
            } else if (auto symRef = llvm::dyn_cast<SymbolRefAttr>(expAttr)) {
               auto column = std::make_shared<tuples::Column>();
               column->type = from.getColumnPtr()->type;
               renameDefs.push_back(tuples::ColumnDefAttr::get(ctx, symRef, column, ArrayAttr::get(ctx, {from})));
            }
         };

         if (colsAttr.size() == 3) {
            if (meta.hasRow()) addRename(colsAttr[0], meta.row);
            if (meta.hasCol()) addRename(colsAttr[1], meta.col);
            if (meta.val) addRename(colsAttr[2], meta.val);
         } else if (colsAttr.size() == 2) {
            if (!meta.hasRow() && meta.hasCol()) {
               addRename(colsAttr[0], meta.col);
               if (meta.val) addRename(colsAttr[1], meta.val);
            } else {
               if (meta.hasRow()) addRename(colsAttr[0], meta.row);
               if (meta.val) addRename(colsAttr[1], meta.val);
            }
         } else if (colsAttr.size() == 1) {
            if (meta.val) addRename(colsAttr[0], meta.val);
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

      MatrixMeta meta;
      auto matType = llvm::cast<MatrixType>(op.getType());
      SmallVector<Attribute> defs;

      if (!matType.getRows().isOne()) {
         meta.rowDef = createColumnDef(ctx, AttributeGenerator::nextName("const_row"), rewriter.getI64Type());
         meta.row = createColumnRef(meta.rowDef);
         defs.push_back(meta.rowDef);
      }
      if (!matType.getCols().isOne()) {
         meta.colDef = createColumnDef(ctx, AttributeGenerator::nextName("const_col"), rewriter.getI64Type());
         meta.col = createColumnRef(meta.colDef);
         defs.push_back(meta.colDef);
      }
      meta.valDef = valDef;
      meta.val = createColumnRef(valDef);
      defs.push_back(valDef);

      SmallVector<Attribute> tuples;
      int64_t rows = matType.getRows().isOne() ? 1 : matType.getRows().getConcreteDim();
      int64_t cols = matType.getCols().isOne() ? 1 : matType.getCols().getConcreteDim();

      bool isScalar = matType.getRows().isOne() && matType.getCols().isOne();

      bool isZero = false;
      if (valType.isInteger(1)) {
         isZero = (llvm::cast<IntegerAttr>(rawVal).getInt() == 0);
      } else if (valType.isInteger(64)) {
         if (matType.getSemiring() == SemiringTypes::forTropInt(ctx))
            isZero = (llvm::cast<IntegerAttr>(rawVal).getInt() == std::numeric_limits<int64_t>::max());
         else if (matType.getSemiring() == SemiringTypes::forTropMaxInt(ctx))
            isZero = (llvm::cast<IntegerAttr>(rawVal).getInt() == std::numeric_limits<int64_t>::min());
         else
            isZero = (llvm::cast<IntegerAttr>(rawVal).getInt() == 0);
      } else if (valType.isF64()) {
         if (matType.getSemiring() == SemiringTypes::forTropReal(ctx))
            isZero = std::isinf(llvm::cast<FloatAttr>(rawVal).getValueAsDouble()) && llvm::cast<FloatAttr>(rawVal).getValueAsDouble() > 0;
         else
            isZero = (llvm::cast<FloatAttr>(rawVal).getValueAsDouble() == 0.0);
      }
      if (isScalar) {
         isZero = false;
      }

      // LingoDB requires at least 1 tuple. If structurally zero or dimension is 0, emit 1 dummy and filter it.
      bool isEmpty = isZero || (rows <= 0) || (cols <= 0);
      int64_t rowsToGen = isEmpty ? 1 : rows;
      int64_t colsToGen = isEmpty ? 1 : cols;

      for (int64_t r = 0; r < rowsToGen; ++r) {
         for (int64_t c = 0; c < colsToGen; ++c) {
            SmallVector<Attribute> rowVals;
            if (!matType.getRows().isOne()) rowVals.push_back(rewriter.getI64IntegerAttr(r));
            if (!matType.getCols().isOne()) rowVals.push_back(rewriter.getI64IntegerAttr(c));
            rowVals.push_back(rawVal);
            tuples.push_back(ArrayAttr::get(ctx, rowVals));
         }
      }

      Value currentRel = rewriter.create<relalg::ConstRelationOp>(
                                    op.getLoc(), ArrayAttr::get(ctx, defs), ArrayAttr::get(ctx, tuples))
                            .getResult();

      if (isEmpty) {
         auto selOp = rewriter.create<relalg::SelectionOp>(op.getLoc(), currentRel);
         auto* selBlock = new Block;
         selOp.getPredicate().push_back(selBlock);
         selBlock->addArgument(tuples::TupleType::get(ctx), op.getLoc());

         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(selBlock);
         Value falseVal = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
         rewriter.create<tuples::ReturnOp>(op.getLoc(), ValueRange{falseVal});
         currentRel = selOp.getResult();
      }

      state.set(op.getResult(), meta);
      rewriter.replaceOp(op, currentRel);
      return success();
   }
};

class ApplyOpConversion : public StatefulConversion<ApplyOp> {
   public:
   using StatefulConversion::StatefulConversion;
   LogicalResult matchAndRewrite(ApplyOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (op.getInputs().empty()) return failure();

      auto loc = op.getLoc();
      auto* ctx = rewriter.getContext();

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
         if (!currentMeta.hasRow() && renamedMeta.hasRow()) {
            currentMeta.row = renamedMeta.row;
            currentMeta.rowDef = renamedMeta.rowDef;
         }
         if (!currentMeta.hasCol() && renamedMeta.hasCol()) {
            currentMeta.col = renamedMeta.col;
            currentMeta.colDef = renamedMeta.colDef;
         }
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

      state.set(op->getResult(0), resMeta);
      rewriter.replaceOp(op, mapOp.getResult());
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

class DeferredReduceConversion : public StatefulConversion<DeferredReduceOp> {
   public:
   using StatefulConversion::StatefulConversion;

   static relalg::AggrFunc getAggrFuncForSemiring(Type semiringType) {
      if (isTropicalNonMax(semiringType)) return relalg::AggrFunc::min;
      if (isTropicalMax(semiringType)) return relalg::AggrFunc::max;
      if (isBool(semiringType)) return relalg::AggrFunc::any;
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

      auto addBroadcast = [&](StringRef prefix, Attribute dimObj, tuples::ColumnRefAttr& outRef, tuples::ColumnDefAttr& outDef) {
         auto dimAttr = llvm::cast<graphalg::DimAttr>(dimObj);
         int64_t size = dimAttr.getConcreteDim();

         outDef = createColumnDef(ctx, AttributeGenerator::nextName(prefix.str()), rewriter.getI64Type());
         SmallVector<Attribute> rows;

         bool isEmpty = (size <= 0);
         int64_t safeSize = isEmpty ? 1 : size;
         for (int64_t i = 0; i < safeSize; ++i) {
            rows.push_back(ArrayAttr::get(ctx, {rewriter.getI64IntegerAttr(i)}));
         }
         Value broadcastRel = rewriter.create<relalg::ConstRelationOp>(
                                         loc, ArrayAttr::get(ctx, {outDef}), ArrayAttr::get(ctx, rows))
                                 .getResult();

         if (isEmpty) {
            auto selOp = rewriter.create<relalg::SelectionOp>(loc, broadcastRel);
            auto* selBlock = new Block;
            selOp.getPredicate().push_back(selBlock);
            selBlock->addArgument(tuples::TupleType::get(ctx), loc);
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(selBlock);
            Value falseVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
            rewriter.create<tuples::ReturnOp>(loc, ValueRange{falseVal});
            broadcastRel = selOp.getResult();
         }

         auto joinOp = rewriter.create<relalg::InnerJoinOp>(loc, currentRel, broadcastRel);
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

      auto selOp = rewriter.create<relalg::SelectionOp>(loc, adaptor.getInput());
      auto* selBlock = new Block;
      selOp.getPredicate().push_back(selBlock);

      {
         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(selBlock);

         auto tupleArgSel = selBlock->addArgument(tuples::TupleType::get(ctx), loc);
         auto val = rewriter.create<tuples::GetColumnOp>(loc, valType, info.val, tupleArgSel);

         Value zero;
         if (isBool(sring))
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

      SmallVector<Attribute> groupByAttrs;
      if (info.hasRow()) groupByAttrs.push_back(info.row);

      auto minColDef = createColumnDef(ctx, AttributeGenerator::nextName("pick_min_col"), rewriter.getI64Type());
      auto anyValDef = createColumnDef(ctx, AttributeGenerator::nextName("pick_any_val"), valType);
      auto minColRef = createColumnRef(minColDef);

      Value finalRel;

      if (isBool(sring)) {
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

         auto mapOp = rewriter.create<relalg::MapOp>(loc, aggOp.getResult());
         mapOp.setComputedColsAttr(ArrayAttr::get(ctx, {anyValDef}));

         {
            OpBuilder::InsertionGuard guard(rewriter);
            auto* mapBlock = new Block;
            mapOp.getPredicate().push_back(mapBlock);
            rewriter.setInsertionPointToStart(mapBlock);

            mapBlock->addArgument(tuples::TupleType::get(ctx), loc);
            Value anyVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));

            rewriter.create<tuples::ReturnOp>(loc, ValueRange{anyVal});
         }
         finalRel = mapOp.getResult();
      } else {
         auto aggOp = rewriter.create<relalg::AggregationOp>(
            loc, tuples::TupleStreamType::get(ctx), selRel,
            ArrayAttr::get(ctx, groupByAttrs), ArrayAttr::get(ctx, {minColDef, anyValDef}));

         {
            OpBuilder::InsertionGuard guard(rewriter);
            auto* aggBlock = rewriter.createBlock(&aggOp.getAggrFunc());
            rewriter.setInsertionPointToStart(aggBlock);

            Value groupStream = aggBlock->addArgument(tuples::TupleStreamType::get(ctx), loc);
            Value minCol = rewriter.create<relalg::AggrFuncOp>(
               loc, rewriter.getI64Type(), relalg::AggrFuncAttr::get(ctx, relalg::AggrFunc::min), groupStream, info.col);

            Value anyVal = rewriter.create<relalg::AggrFuncOp>(
               loc, valType, relalg::AggrFuncAttr::get(ctx, relalg::AggrFunc::any), groupStream, info.val);

            rewriter.create<tuples::ReturnOp>(loc, ValueRange{minCol, anyVal});
         }
         finalRel = aggOp.getResult();
      }

      MatrixMeta resMeta;
      resMeta.row = info.row;
      resMeta.rowDef = info.rowDef;
      resMeta.col = minColRef;
      resMeta.colDef = minColDef;
      resMeta.val = createColumnRef(anyValDef);
      resMeta.valDef = anyValDef;

      state.set(op.getResult(), resMeta);
      rewriter.replaceOp(op, finalRel);
      return success();
   }
};

class UnionOpConversion : public StatefulConversion<UnionOp> {
   public:
   using StatefulConversion::StatefulConversion;
   LogicalResult matchAndRewrite(UnionOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op.getLoc();
      auto* ctx = rewriter.getContext();

      if (op.getInputs().empty()) return failure();

      auto outMatrixType = llvm::cast<MatrixType>(op.getResult().getType());
      bool hasOutRow = !outMatrixType.getRows().isOne();
      bool hasOutCol = !outMatrixType.getCols().isOne();
      Type valType = GraphAlgTypeConverter::convertSemiringType(outMatrixType.getSemiring());

      Value currentRel = adaptor.getInputs()[0];
      MatrixMeta currentMeta = state.get(op.getInputs()[0], ctx);

      if (op.getInputs().size() == 1) {
         if (!hasOutRow) {
            currentMeta.row = nullptr;
            currentMeta.rowDef = nullptr;
         }
         if (!hasOutCol) {
            currentMeta.col = nullptr;
            currentMeta.colDef = nullptr;
         }
         state.set(op.getResult(), currentMeta);
         rewriter.replaceOp(op, currentRel);
         return success();
      }

      for (size_t i = 1; i < op.getInputs().size(); ++i) {
         Value rightRel = adaptor.getInputs()[i];
         MatrixMeta rightMeta = state.get(op.getInputs()[i], ctx);

         SmallVector<Attribute> mappingDefs;
         MatrixMeta nextMeta;

         auto mapColumn = [&](StringRef prefix, Type type,
                              tuples::ColumnRefAttr leftRef,
                              tuples::ColumnRefAttr rightRef,
                              tuples::ColumnDefAttr& outDef,
                              tuples::ColumnRefAttr& outRef) {
            if (!leftRef || !rightRef) return;

            outDef = createColumnDef(ctx, AttributeGenerator::nextName(prefix.str()), type);
            outRef = createColumnRef(outDef);

            auto mappingDef = tuples::ColumnDefAttr::get(
               ctx, outDef.getName(), outDef.getColumnPtr(),
               ArrayAttr::get(ctx, {leftRef, rightRef}));

            mappingDefs.push_back(mappingDef);
         };

         if (hasOutRow) {
            mapColumn("union_row", rewriter.getI64Type(),
                      currentMeta.row, rightMeta.row, nextMeta.rowDef, nextMeta.row);
         }
         if (hasOutCol) {
            mapColumn("union_col", rewriter.getI64Type(),
                      currentMeta.col, rightMeta.col, nextMeta.colDef, nextMeta.col);
         }
         mapColumn("union_val", valType,
                   currentMeta.val, rightMeta.val, nextMeta.valDef, nextMeta.val);

         auto setSemanticAttr = relalg::SetSemanticAttr::get(ctx, relalg::SetSemantic::all);
         auto mappingAttr = ArrayAttr::get(ctx, mappingDefs);

         auto unionOp = rewriter.create<relalg::UnionOp>(
            loc, tuples::TupleStreamType::get(ctx), setSemanticAttr,
            currentRel, rightRel, mappingAttr);

         currentRel = unionOp.getResult();
         currentMeta = nextMeta;
      }

      state.set(op.getResult(), currentMeta);
      rewriter.replaceOp(op, currentRel);
      return success();
   }
};

class AddConversion : public OpConversionPattern<AddOp> {
   public:
   using OpConversionPattern::OpConversionPattern;
   LogicalResult matchAndRewrite(AddOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto sring = op.getType();
      auto* ctx = rewriter.getContext();
      if (isBool(sring)) {
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
      if (isBool(sring)) {
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
         if (isBool(ring)) return rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
         if (ring == SemiringTypes::forInt(ctx)) return rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
         if (ring == SemiringTypes::forReal(ctx)) return rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(rewriter.getF64Type(), 0.0));
         if (ring == SemiringTypes::forTropInt(ctx)) return rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(std::numeric_limits<int64_t>::max()));
         if (ring == SemiringTypes::forTropMaxInt(ctx)) return rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(std::numeric_limits<int64_t>::min()));
         if (ring == SemiringTypes::forTropReal(ctx)) return rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(rewriter.getF64Type(), std::numeric_limits<double>::infinity()));
         return nullptr;
      };

      auto getOne = [&](Type ring) -> Value {
         if (isBool(ring)) return rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
         if (ring == SemiringTypes::forInt(ctx)) return rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
         if (ring == SemiringTypes::forReal(ctx)) return rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(rewriter.getF64Type(), 1.0));
         if (ring == SemiringTypes::forTropInt(ctx) || ring == SemiringTypes::forTropMaxInt(ctx)) return rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
         if (ring == SemiringTypes::forTropReal(ctx)) return rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(rewriter.getF64Type(), 0.0));
         return nullptr;
      };

      if (isBool(outRing)) {
         Value zeroIn = getZero(inRing);
         if (!zeroIn) return failure();
         if (inType.isInteger(64)) {
            rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, arith::CmpIPredicate::ne, input, zeroIn);
         } else if (inType.isF64()) {
            rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, arith::CmpFPredicate::UNE, input, zeroIn);
         }
         return success();
      }

      if (isBool(inRing)) {
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

static mlir::IntegerAttr tryGetConstantInt(mlir::Value v) {
   mlir::Attribute attr;
   if (!mlir::matchPattern(v, mlir::m_Constant(&attr))) {
      return nullptr;
   }
   return llvm::cast<mlir::IntegerAttr>(attr);
}

class MaskOpConversion : public StatefulConversion<MaskOp> {
   public:
   using StatefulConversion::StatefulConversion;
   LogicalResult matchAndRewrite(MaskOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op.getLoc();
      auto ctx = rewriter.getContext();

      Value inputRel = adaptor.getOperands()[2];
      Value maskRel = adaptor.getMask();

      auto inputMeta = state.get(op.getOperand(2), ctx);
      auto maskMeta = state.get(op.getMask(), ctx);

      auto* joinBlock = new Block;
      auto tupleArg = joinBlock->addArgument(tuples::TupleType::get(ctx), loc);
      Value cmpAcc;
      {
         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(joinBlock);

         if (inputMeta.hasRow() && maskMeta.hasRow()) {
            auto valA = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), inputMeta.row, tupleArg);
            auto valB = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), maskMeta.row, tupleArg);
            cmpAcc = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, valA, valB);
         }
         if (inputMeta.hasCol() && maskMeta.hasCol()) {
            auto valA = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), inputMeta.col, tupleArg);
            auto valB = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), maskMeta.col, tupleArg);
            auto cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, valA, valB);
            if (cmpAcc)
               cmpAcc = rewriter.create<arith::AndIOp>(loc, cmpAcc, cmp);
            else
               cmpAcc = cmp;
         }
         if (!cmpAcc) cmpAcc = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
         rewriter.create<tuples::ReturnOp>(loc, ValueRange{cmpAcc});
      }

      if (op.getComplement()) {
         auto joinOp = rewriter.create<relalg::AntiSemiJoinOp>(loc, inputRel, maskRel);
         joinOp.getPredicate().push_back(joinBlock);
         state.set(op.getResult(), inputMeta);
         rewriter.replaceOp(op, joinOp.getResult());
      } else {
         auto joinOp = rewriter.create<relalg::SemiJoinOp>(loc, inputRel, maskRel);
         joinOp.getPredicate().push_back(joinBlock);
         state.set(op.getResult(), inputMeta);
         rewriter.replaceOp(op, joinOp.getResult());
      }
      return success();
   }
};

void GraphAlgToRelAlgPass::runOnOperation() {
   MLIRContext* context = &getContext();

   // unroll ForDimOp, ForConstOp
   IRRewriter rewriter(context);
   SmallVector<Operation*> loopsToUnroll;
   getOperation().walk([&](Operation* op) {
      if (llvm::isa<ForDimOp, ForConstOp>(op)) {
         loopsToUnroll.push_back(op);
      }
   });

   for (Operation* op : loopsToUnroll) {
      auto loc = op->getLoc();

      int numIterations = 50;
      if (auto forConst = dyn_cast<ForConstOp>(op)) {
         auto rangeBegin = tryGetConstantInt(forConst.getRangeBegin());
         auto rangeEnd = tryGetConstantInt(forConst.getRangeEnd());
         if (rangeBegin && rangeEnd) {
            numIterations = rangeEnd.getInt() - rangeBegin.getInt();
         }
      } else if (auto forDim = dyn_cast<ForDimOp>(op)) {
         numIterations = forDim.getDim().getConcreteDim();
      }
      Block* oldBody = &op->getRegion(0).front();
      Operation* terminator = oldBody->getTerminator();

      SmallVector<Value> currentArgs(op->getOperands().begin(), op->getOperands().end());
      rewriter.setInsertionPoint(op);

      for (int step = 0; step < numIterations; ++step) {
         IRMapping mapping;

         Type arg0Type = oldBody->getArgument(0).getType();
         Value dummyIdx;
         if (llvm::isa<MatrixType>(arg0Type)) {
            dummyIdx = rewriter.create<ConstantMatrixOp>(loc, arg0Type, rewriter.getI64IntegerAttr(step));
         } else if (arg0Type.isIndex()) {
            dummyIdx = rewriter.create<arith::ConstantIndexOp>(loc, step);
         } else {
            dummyIdx = rewriter.create<arith::ConstantIntOp>(loc, step, arg0Type.getIntOrFloatBitWidth());
         }
         mapping.map(oldBody->getArgument(0), dummyIdx);

         for (size_t i = 0; i < currentArgs.size(); ++i) {
            mapping.map(oldBody->getArgument(i + 1), currentArgs[i]);
         }

         for (Operation& innerOp : oldBody->without_terminator()) {
            rewriter.clone(innerOp, mapping);
         }

         SmallVector<Value> nextArgs;
         for (Value yieldOperand : terminator->getOperands()) {
            nextArgs.push_back(mapping.lookupOrDefault(yieldOperand));
         }
         currentArgs = nextArgs;
      }
      rewriter.replaceOp(op, currentArgs);
   }

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

   patterns.add<MatMulJoinOpConversion>(typeConverter, context, state);

   patterns.add<
      DiagOpConversion,
      BroadcastOpConversion,
      TrilOpConversion,
      MaskOpConversion,
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