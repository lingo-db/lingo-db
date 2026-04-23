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

static tuples::ColumnRefAttr resolveColumnRef(MLIRContext* ctx, Attribute attr) {
   if (auto colRef = llvm::dyn_cast<tuples::ColumnRefAttr>(attr)) return colRef;
   if (auto symRef = llvm::dyn_cast<SymbolRefAttr>(attr)) {
      auto& colManager = ctx->getOrLoadDialect<tuples::TupleStreamDialect>()->getColumnManager();
      StringRef scope = symRef.getNestedReferences().empty() ? "" : symRef.getRootReference().getValue();
      StringRef name = symRef.getLeafReference().getValue();
      auto col = colManager.get(scope, name);
      return colManager.createRef(col.get());
   }
   return nullptr;
}

static tuples::ColumnDefAttr createColumnDef(MLIRContext* ctx, StringRef name, Type type) {
   auto& colManager = ctx->getOrLoadDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto col = colManager.get("graphalg", name);
   col->type = type;
   return colManager.createDef("graphalg", name);
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

   Type semiring;
   DimAttr rowDim;
   DimAttr colDim;

   bool hasRow() const { return row != nullptr; }
   bool hasCol() const { return col != nullptr; }
};

class GraphAlgTypeConverter : public TypeConverter {
   public:
   GraphAlgTypeConverter(MLIRContext* ctx) {
      addConversion([](Type type) { return type; });

      addConversion([](MatrixType type) -> Type {
         return tuples::TupleStreamType::get(type.getContext());
      });

      addConversion([](Type type) -> std::optional<Type> {
         if (llvm::isa<SemiringTypeInterface>(type)) {
            Type converted = convertSemiringType(type);
            if (converted == type) return std::nullopt;
            return converted;
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
         if (t == SemiringTypes::forInt(ctx) || t == SemiringTypes::forTropInt(ctx) || t == SemiringTypes::forTropMaxInt(ctx)) return IntegerType::get(ctx, 64);
         if (t == SemiringTypes::forReal(ctx) || t == SemiringTypes::forTropReal(ctx)) return Float64Type::get(ctx);
      }
      return t;
   }
};

class ConversionState {
   llvm::DenseMap<Value, MatrixMeta> metaMap;

   public:
   // Simpler yield state restored!
   llvm::DenseMap<Operation*, std::pair<Value, Value>> loopYieldData;

   // We now simply map abstract Dimensions to resolved integer limits statically
   llvm::DenseMap<Attribute, int64_t> globalDimSizes;

   void set(Value origVal, const MatrixMeta& meta) { metaMap[origVal] = meta; }
   MatrixMeta get(Value origVal, MLIRContext* ctx) {
      auto it = metaMap.find(origVal);
      if (it != metaMap.end()) return it->second;

      if (auto matType = llvm::dyn_cast<MatrixType>(origVal.getType())) {
         MatrixMeta meta;
         meta.semiring = matType.getSemiring();
         meta.rowDim = matType.getRows();
         meta.colDim = matType.getCols();
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

static std::pair<MatrixMeta, Value> renameMeta(OpBuilder& rewriter, Location loc, MatrixMeta meta, StringRef prefix, Value rel, Type valType) {
   SmallVector<Attribute> renameDefs;
   MatrixMeta renamedMeta = meta;
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

static void resolveDimensionsGlobally(ModuleOp module, ConversionState& state) {
   module.walk([&](UnrealizedConversionCastOp castOp) {
      if (castOp.getResultTypes().empty()) return;
      auto matType = llvm::dyn_cast<MatrixType>(castOp.getResultTypes()[0]);
      if (!matType) return;

      auto colsAttr = castOp->getAttrOfType<ArrayAttr>("cols");
      if (!colsAttr) return;

      auto constRelOp = castOp.getOperand(0).getDefiningOp<relalg::ConstRelationOp>();
      if (!constRelOp) return;

      auto extractMaxAndSet = [&](Attribute dimAttr, Attribute colRefAttr) {
         if (auto dim = llvm::dyn_cast<DimAttr>(dimAttr)) {
            if (dim.isAbstract()) {
               auto symRef = llvm::dyn_cast<SymbolRefAttr>(colRefAttr);
               if (!symRef) return;

               StringRef targetName = symRef.getLeafReference().getValue();
               int colIdx = -1;
               auto columns = constRelOp.getColumns();

               // Locate the column index associated with this dimension
               for (size_t i = 0; i < columns.size(); ++i) {
                  if (auto colDef = llvm::dyn_cast<tuples::ColumnDefAttr>(columns[i])) {
                     if (colDef.getName().getLeafReference().getValue() == targetName) {
                        colIdx = i;
                        break;
                     }
                  } else if (auto sym = llvm::dyn_cast<SymbolRefAttr>(columns[i])) {
                     if (sym.getLeafReference().getValue() == targetName) {
                        colIdx = i;
                        break;
                     }
                  }
               }
               if (colIdx == -1) return;

               // Sweep constant relation rows to find max integer
               int64_t maxVal = 0;
               for (Attribute rowAttr : constRelOp.getValues()) {
                  if (auto rowArray = llvm::dyn_cast<ArrayAttr>(rowAttr)) {
                     if (colIdx < (int) rowArray.size()) {
                        if (auto intAttr = llvm::dyn_cast<IntegerAttr>(rowArray[colIdx])) {
                           maxVal = std::max(maxVal, intAttr.getInt());
                        }
                     }
                  }
               }

               // Assuming 1-based node IDs (e.g., node 7 needs size 8)
               int64_t dimSize = maxVal + 1;
               if (state.globalDimSizes.count(dim)) {
                  state.globalDimSizes[dim] = std::max(state.globalDimSizes[dim], dimSize);
               } else {
                  state.globalDimSizes[dim] = dimSize;
               }
            }
         }
      };

      if (colsAttr.size() >= 2) {
         if (matType.getRows().isOne()) {
            extractMaxAndSet(matType.getCols(), colsAttr[0]);
         } else {
            extractMaxAndSet(matType.getRows(), colsAttr[0]);
            if (colsAttr.size() > 1) {
               extractMaxAndSet(matType.getCols(), colsAttr[1]);
            }
         }
      }
   });
}

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

   public:
   LoadCastConversion(const TypeConverter& tc, MLIRContext* ctx, ConversionState& state)
      : ConversionPattern(tc, "builtin.unrealized_conversion_cast", 1, ctx), state(state) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      if (operands.empty()) return failure();

      if (llvm::isa<MatrixType>(op->getResultTypes()[0])) {
         auto colsAttr = op->getAttrOfType<ArrayAttr>("cols");
         if (!colsAttr) return failure();

         MatrixMeta meta;
         auto matType = llvm::cast<MatrixType>(op->getResultTypes()[0]);
         meta.semiring = matType.getSemiring();
         meta.rowDim = matType.getRows();
         meta.colDim = matType.getCols();
         auto* ctx = rewriter.getContext();

         if (colsAttr.size() == 3) {
            meta.row = resolveColumnRef(ctx, colsAttr[0]);
            meta.col = resolveColumnRef(ctx, colsAttr[1]);
            meta.val = resolveColumnRef(ctx, colsAttr[2]);
         } else if (colsAttr.size() == 2) {
            if (matType.getRows().isOne()) {
               meta.col = resolveColumnRef(ctx, colsAttr[0]);
               meta.val = resolveColumnRef(ctx, colsAttr[1]);
            } else {
               meta.row = resolveColumnRef(ctx, colsAttr[0]);
               meta.val = resolveColumnRef(ctx, colsAttr[1]);
            }
         } else if (colsAttr.size() == 1) {
            meta.val = resolveColumnRef(ctx, colsAttr[0]);
         }

         state.set(op->getResult(0), meta);
         rewriter.replaceOp(op, operands[0]);
         return success();
      }

      if (llvm::isa<tuples::TupleStreamType>(op->getResultTypes()[0])) {
         auto colsAttr = op->getAttrOfType<ArrayAttr>("cols");
         if (!colsAttr) {
            rewriter.replaceOp(op, operands[0]);
            return success();
         }

         auto meta = state.get(op->getOperand(0), rewriter.getContext());
         SmallVector<Attribute> renameDefs;
         auto* ctx = rewriter.getContext();
         auto addRename = [&](Attribute expAttr, tuples::ColumnRefAttr from) {
            if (!expAttr || !from) return;
            auto exp = resolveColumnRef(ctx, expAttr);
            if (exp && exp.getColumnPtr()->type == from.getColumnPtr()->type) {
               if (exp != from) {
                  renameDefs.push_back(tuples::ColumnDefAttr::get(ctx, exp.getName(), exp.getColumnPtr(), ArrayAttr::get(ctx, {from})));
               }
            } else if (auto symRef = llvm::dyn_cast<SymbolRefAttr>(expAttr)) {
               auto& colManager = ctx->getOrLoadDialect<tuples::TupleStreamDialect>()->getColumnManager();
               StringRef scope = symRef.getRootReference().getValue();
               StringRef name = symRef.getLeafReference().getValue();
               auto col = colManager.get(scope, name);
               col->type = from.getColumnPtr()->type;
               renameDefs.push_back(colManager.createDef(symRef, ArrayAttr::get(ctx, {from})));
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
      meta.semiring = matType.getSemiring();

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
      int64_t rows = matType.getRows().isOne() ? 1 : (matType.getRows().isConcrete() ? matType.getRows().getConcreteDim() : state.globalDimSizes[matType.getRows()]);
      int64_t cols = matType.getCols().isOne() ? 1 : (matType.getCols().isConcrete() ? matType.getCols().getConcreteDim() : state.globalDimSizes[matType.getCols()]);

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
         Type valType = GraphAlgTypeConverter::convertSemiringType(meta.semiring);

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
            Type vType = GraphAlgTypeConverter::convertSemiringType(inputMetas[i].semiring);
            newArgs.push_back(rewriter.create<tuples::GetColumnOp>(loc, vType, inputMetas[i].val, tupleArg));
         }

         rewriter.mergeBlocks(&op.getBody().front(), mapBlock, newArgs);
      }

      MatrixMeta resMeta = currentMeta;
      resMeta.valDef = resDef;
      resMeta.val = createColumnRef(resDef);
      resMeta.semiring = op.getType().getSemiring();

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

      auto matrixType = llvm::cast<MatrixType>(op->getResultTypes()[0]);
      Type rhsValType = GraphAlgTypeConverter::convertSemiringType(rhsMeta.semiring);

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

         Type lhsValType = GraphAlgTypeConverter::convertSemiringType(lhsMeta.semiring);

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
      resMeta.semiring = matrixType.getSemiring();
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
      resMeta.semiring = resMatrixType.getSemiring();

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
         OpBuilder::InsertionGuard guard(rewriter);
         Block* aggBlock = rewriter.createBlock(&aggOp.getAggrFunc());
         Value groupStream = aggBlock->addArgument(tuples::TupleStreamType::get(rewriter.getContext()), op.getLoc());
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
      auto info = state.get(op->getOperand(0), rewriter.getContext());

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
      auto info = state.get(op->getOperand(0), rewriter.getContext());

      MatrixMeta resMeta = info;

      if (info.hasRow() && !info.hasCol()) {
         resMeta.colDef = info.rowDef;
         resMeta.col = info.row;
      } else if (!info.hasRow() && info.hasCol()) {
         resMeta.rowDef = info.colDef;
         resMeta.row = info.col;
      }

      state.set(op.getResult(), resMeta);
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
   }
};

class TrilOpConversion : public StatefulConversion<TrilOp> {
   public:
   using StatefulConversion::StatefulConversion;
   LogicalResult matchAndRewrite(TrilOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto info = state.get(op->getOperand(0), rewriter.getContext());
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

class BroadcastOpConversion : public StatefulConversion<BroadcastOp> {
   public:
   using StatefulConversion::StatefulConversion;

   LogicalResult matchAndRewrite(BroadcastOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto info = state.get(op->getOperand(0), rewriter.getContext());
      auto loc = op.getLoc();
      auto ctx = rewriter.getContext();

      auto outType = llvm::cast<MatrixType>(op.getResult().getType());
      Value currentRel = adaptor.getInput();
      MatrixMeta resMeta = info;

      auto addBroadcast = [&](StringRef prefix, Attribute dimObj, tuples::ColumnRefAttr& outRef, tuples::ColumnDefAttr& outDef) {
         auto dimAttr = llvm::cast<DimAttr>(dimObj);
         int64_t size = dimAttr.isConcrete() ? dimAttr.getConcreteDim() : state.globalDimSizes[dimAttr];

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

class PickAnyOpConversion : public StatefulConversion<PickAnyOp> {
   public:
   using StatefulConversion::StatefulConversion;
   LogicalResult matchAndRewrite(PickAnyOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto info = state.get(op->getOperand(0), rewriter.getContext());
      auto loc = op.getLoc();
      auto* ctx = rewriter.getContext();
      Type valType = GraphAlgTypeConverter::convertSemiringType(op.getType().getSemiring());
      Type sring = op.getType().getSemiring();

      SmallVector<Attribute> groupByAttrs;
      if (info.hasRow()) groupByAttrs.push_back(info.row);

      Value finalRel;
      MatrixMeta resMeta = info;
      resMeta.semiring = op.getType().getSemiring();

      if (isBool(sring)) {
         auto constValDef = createColumnDef(ctx, AttributeGenerator::nextName("filter_true"), valType);
         auto constRel = rewriter.create<relalg::ConstRelationOp>(
            loc, ArrayAttr::get(ctx, {constValDef}), ArrayAttr::get(ctx, {ArrayAttr::get(ctx, {rewriter.getIntegerAttr(rewriter.getI1Type(), 1)})}));
         auto constValRef = createColumnRef(constValDef);

         auto semiJoinOp = rewriter.create<relalg::SemiJoinOp>(loc, adaptor.getInput(), constRel);
         auto* joinBlock = new Block;
         semiJoinOp.getPredicate().push_back(joinBlock);
         {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(joinBlock);
            auto tupleArgSel = joinBlock->addArgument(tuples::TupleType::get(ctx), loc);

            auto valA = rewriter.create<tuples::GetColumnOp>(loc, valType, info.val, tupleArgSel);
            auto valB = rewriter.create<tuples::GetColumnOp>(loc, valType, constValRef, tupleArgSel);
            auto cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, valA, valB);

            rewriter.create<tuples::ReturnOp>(loc, ValueRange{cmp});
         }
         Value selRel = semiJoinOp.getResult();

         auto minColDef = createColumnDef(ctx, AttributeGenerator::nextName("pick_min_col"), rewriter.getI64Type());
         auto anyValDef = createColumnDef(ctx, AttributeGenerator::nextName("pick_any_val"), valType);

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
         resMeta.col = createColumnRef(minColDef);
         resMeta.colDef = minColDef;
         resMeta.val = createColumnRef(anyValDef);
         resMeta.valDef = anyValDef;

      } else {
         auto selOp = rewriter.create<relalg::SelectionOp>(loc, adaptor.getInput());
         auto* selBlock = new Block;
         selOp.getPredicate().push_back(selBlock);
         {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(selBlock);

            auto tupleArgSel = selBlock->addArgument(tuples::TupleType::get(ctx), loc);
            auto val = rewriter.create<tuples::GetColumnOp>(loc, valType, info.val, tupleArgSel);

            Value zero;
            if (sring.isInteger(64) || sring == SemiringTypes::forInt(ctx))
               zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
            else if (sring.isF64() || sring == SemiringTypes::forReal(ctx))
               zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(rewriter.getF64Type(), 0.0));
            else if (sring == SemiringTypes::forTropInt(ctx))
               zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(std::numeric_limits<int64_t>::max()));
            else if (sring == SemiringTypes::forTropMaxInt(ctx))
               zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(std::numeric_limits<int64_t>::min()));
            else if (sring == SemiringTypes::forTropReal(ctx))
               zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(rewriter.getF64Type(), std::numeric_limits<double>::infinity()));

            Value cmpZero;
            if (valType.isIntOrIndex()) {
               cmpZero = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, val, zero);
            } else {
               cmpZero = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, val, zero);
            }
            rewriter.create<tuples::ReturnOp>(loc, ValueRange{cmpZero});
         }
         Value selRel = selOp.getResult();

         SmallVector<Attribute> renameDefs;
         tuples::ColumnRefAttr aggRowRef;

         if (info.hasRow()) {
            tuples::ColumnDefAttr aggRowDef = createColumnDef(ctx, AttributeGenerator::nextName("pick_agg_row"), rewriter.getI64Type());
            aggRowRef = createColumnRef(aggRowDef);
            renameDefs.push_back(tuples::ColumnDefAttr::get(ctx, aggRowDef.getName(), aggRowDef.getColumnPtr(), ArrayAttr::get(ctx, {info.row})));
         }

         auto aggColDef = createColumnDef(ctx, AttributeGenerator::nextName("pick_agg_col"), rewriter.getI64Type());
         auto aggColRef = createColumnRef(aggColDef);
         renameDefs.push_back(tuples::ColumnDefAttr::get(ctx, aggColDef.getName(), aggColDef.getColumnPtr(), ArrayAttr::get(ctx, {info.col})));

         auto renamedRel = rewriter.create<relalg::RenamingOp>(
                                      loc, tuples::TupleStreamType::get(ctx), selRel, ArrayAttr::get(ctx, renameDefs))
                              .getResult();

         auto minColDef = createColumnDef(ctx, AttributeGenerator::nextName("pick_min_col"), rewriter.getI64Type());
         auto minColRef = createColumnRef(minColDef);

         SmallVector<Attribute> aggGroupByAttrs;
         if (info.hasRow()) aggGroupByAttrs.push_back(aggRowRef);

         auto aggOp = rewriter.create<relalg::AggregationOp>(
            loc, tuples::TupleStreamType::get(ctx), renamedRel,
            ArrayAttr::get(ctx, aggGroupByAttrs), ArrayAttr::get(ctx, {minColDef}));

         {
            OpBuilder::InsertionGuard guard(rewriter);
            auto* aggBlock = rewriter.createBlock(&aggOp.getAggrFunc());
            rewriter.setInsertionPointToStart(aggBlock);

            Value groupStream = aggBlock->addArgument(tuples::TupleStreamType::get(ctx), loc);
            Value minCol = rewriter.create<relalg::AggrFuncOp>(
               loc, rewriter.getI64Type(), relalg::AggrFuncAttr::get(ctx, relalg::AggrFunc::min), groupStream, aggColRef);

            rewriter.create<tuples::ReturnOp>(loc, ValueRange{minCol});
         }

         auto semiJoinOp = rewriter.create<relalg::SemiJoinOp>(loc, selRel, aggOp.getResult());
         auto* joinBlock = new Block;
         semiJoinOp.getPredicate().push_back(joinBlock);

         {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(joinBlock);
            auto tupleArg = joinBlock->addArgument(tuples::TupleType::get(ctx), loc);

            auto valColA = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), info.col, tupleArg);
            auto valColB = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), minColRef, tupleArg);
            Value joinCmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, valColA, valColB);

            if (info.hasRow()) {
               auto valRowA = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), info.row, tupleArg);
               auto valRowB = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), aggRowRef, tupleArg);
               auto cmpRow = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, valRowA, valRowB);
               joinCmp = rewriter.create<arith::AndIOp>(loc, cmpRow, joinCmp);
            }
            rewriter.create<tuples::ReturnOp>(loc, ValueRange{joinCmp});
         }

         auto finalColDef = createColumnDef(ctx, AttributeGenerator::nextName("pick_final_col"), rewriter.getI64Type());
         auto finalValDef = createColumnDef(ctx, AttributeGenerator::nextName("pick_final_val"), valType);

         auto finalAggOp = rewriter.create<relalg::AggregationOp>(
            loc, tuples::TupleStreamType::get(ctx), semiJoinOp.getResult(),
            ArrayAttr::get(ctx, groupByAttrs), ArrayAttr::get(ctx, {finalColDef, finalValDef}));

         {
            OpBuilder::InsertionGuard guard(rewriter);
            auto* finalAggBlock = rewriter.createBlock(&finalAggOp.getAggrFunc());
            rewriter.setInsertionPointToStart(finalAggBlock);

            Value groupStream = finalAggBlock->addArgument(tuples::TupleStreamType::get(ctx), loc);
            Value finalCol = rewriter.create<relalg::AggrFuncOp>(
               loc, rewriter.getI64Type(), relalg::AggrFuncAttr::get(ctx, relalg::AggrFunc::min), groupStream, info.col);
            Value finalVal = rewriter.create<relalg::AggrFuncOp>(
               loc, valType, relalg::AggrFuncAttr::get(ctx, relalg::AggrFunc::any), groupStream, info.val);

            rewriter.create<tuples::ReturnOp>(loc, ValueRange{finalCol, finalVal});
         }

         finalRel = finalAggOp.getResult();
         resMeta.col = createColumnRef(finalColDef);
         resMeta.colDef = finalColDef;
         resMeta.val = createColumnRef(finalValDef);
         resMeta.valDef = finalValDef;
      }

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
         MatrixMeta nextMeta = currentMeta; // Preserve semiring

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
      auto inRing = op->getOperand(0).getType();
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

class CastDimOpConversion : public StatefulConversion<CastDimOp> {
   public:
   using StatefulConversion::StatefulConversion;
   LogicalResult matchAndRewrite(CastDimOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op.getLoc();
      auto* ctx = rewriter.getContext();

      auto outMatrixType = llvm::cast<MatrixType>(op.getType());
      Type valType = GraphAlgTypeConverter::convertSemiringType(outMatrixType.getSemiring());

      auto dimAttr = op.getInput();
      int64_t dimSize = dimAttr.isConcrete() ? dimAttr.getConcreteDim() : state.globalDimSizes[dimAttr];

      tuples::ColumnDefAttr finalDef = createColumnDef(ctx, AttributeGenerator::nextName("castdim_concrete"), valType);

      Value sizeI64 = rewriter.create<arith::ConstantIndexOp>(loc, dimSize);
      Value castedSize = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), sizeI64);
      if (valType.isF64()) {
         castedSize = rewriter.create<arith::SIToFPOp>(loc, valType, castedSize);
      }

      auto dummyDef = createColumnDef(ctx, AttributeGenerator::nextName("dummy"), rewriter.getI1Type());
      Value constRel = rewriter.create<relalg::ConstRelationOp>(
                                  loc, ArrayAttr::get(ctx, {dummyDef}),
                                  ArrayAttr::get(ctx, {ArrayAttr::get(ctx, {rewriter.getIntegerAttr(rewriter.getI1Type(), 0)})}))
                          .getResult();

      auto mapOp = rewriter.create<relalg::MapOp>(loc, constRel);
      mapOp.setComputedColsAttr(ArrayAttr::get(ctx, {finalDef}));
      {
         OpBuilder::InsertionGuard guard(rewriter);
         Block* mapBlock = rewriter.createBlock(&mapOp.getPredicate());
         mapBlock->addArgument(tuples::TupleType::get(ctx), loc);
         rewriter.setInsertionPointToEnd(mapBlock);
         rewriter.create<tuples::ReturnOp>(loc, ValueRange{castedSize});
      }

      MatrixMeta meta;
      meta.semiring = outMatrixType.getSemiring();
      meta.valDef = finalDef;
      meta.val = createColumnRef(finalDef);

      state.set(op.getResult(), meta);
      rewriter.replaceOp(op, mapOp.getResult());
      return success();
   }
};

class MaskOpConversion : public StatefulConversion<MaskOp> {
   public:
   using StatefulConversion::StatefulConversion;

   LogicalResult matchAndRewrite(MaskOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op.getLoc();
      auto* ctx = rewriter.getContext();

      Value inputRel = adaptor.getOperands()[2];
      Value maskRel = adaptor.getMask();

      OpAdaptor origAdaptor(op->getOperands(), op->getAttrDictionary());

      auto inputMeta = state.get(op->getOperand(2), ctx);
      auto maskMeta = state.get(origAdaptor.getMask(), ctx);

      auto isConflict = [&](tuples::ColumnRefAttr c) {
         if (!c) return false;
         return c == inputMeta.row || c == inputMeta.col || c == inputMeta.val;
      };

      SmallVector<Attribute> renameDefs;
      MatrixMeta renamedMaskMeta = maskMeta;
      StringRef prefix = "mask";

      if (maskMeta.hasRow() && isConflict(maskMeta.row)) {
         auto newDef = createColumnDef(ctx, AttributeGenerator::nextName(prefix.str() + "_row"), rewriter.getI64Type());
         renameDefs.push_back(tuples::ColumnDefAttr::get(ctx, newDef.getName(), newDef.getColumnPtr(), ArrayAttr::get(ctx, {maskMeta.row})));
         renamedMaskMeta.rowDef = newDef;
         renamedMaskMeta.row = createColumnRef(newDef);
      }
      if (maskMeta.hasCol() && isConflict(maskMeta.col)) {
         auto newDef = createColumnDef(ctx, AttributeGenerator::nextName(prefix.str() + "_col"), rewriter.getI64Type());
         renameDefs.push_back(tuples::ColumnDefAttr::get(ctx, newDef.getName(), newDef.getColumnPtr(), ArrayAttr::get(ctx, {maskMeta.col})));
         renamedMaskMeta.colDef = newDef;
         renamedMaskMeta.col = createColumnRef(newDef);
      }
      if (maskMeta.val && isConflict(maskMeta.val)) {
         Type maskValType = GraphAlgTypeConverter::convertSemiringType(maskMeta.semiring);
         auto newDef = createColumnDef(ctx, AttributeGenerator::nextName(prefix.str() + "_val"), maskValType);
         renameDefs.push_back(tuples::ColumnDefAttr::get(ctx, newDef.getName(), newDef.getColumnPtr(), ArrayAttr::get(ctx, {maskMeta.val})));
         renamedMaskMeta.valDef = newDef;
         renamedMaskMeta.val = createColumnRef(newDef);
      }

      if (!renameDefs.empty()) {
         maskRel = rewriter.create<relalg::RenamingOp>(loc, tuples::TupleStreamType::get(ctx), maskRel, ArrayAttr::get(ctx, renameDefs)).getResult();
      }

      Type sring = maskMeta.semiring;
      if (Type maskValType = GraphAlgTypeConverter::convertSemiringType(sring); maskValType.isInteger(1)) {
         auto constValDef = createColumnDef(ctx, AttributeGenerator::nextName("filter_true"), maskValType);
         auto constRel = rewriter.create<relalg::ConstRelationOp>(
            loc, ArrayAttr::get(ctx, {constValDef}),
            ArrayAttr::get(ctx, {ArrayAttr::get(ctx, {rewriter.getIntegerAttr(rewriter.getI1Type(), 1)})}));
         auto constValRef = createColumnRef(constValDef);

         auto semiJoinOp = rewriter.create<relalg::SemiJoinOp>(loc, maskRel, constRel);
         {
            OpBuilder::InsertionGuard guard(rewriter);
            auto* joinBlock = rewriter.createBlock(&semiJoinOp.getPredicate());
            auto tupleArg = joinBlock->addArgument(tuples::TupleType::get(ctx), loc);
            rewriter.setInsertionPointToEnd(joinBlock);

            auto valA = rewriter.create<tuples::GetColumnOp>(loc, maskValType, renamedMaskMeta.val, tupleArg);
            auto valB = rewriter.create<tuples::GetColumnOp>(loc, maskValType, constValRef, tupleArg);
            auto cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, valA, valB);

            rewriter.create<tuples::ReturnOp>(loc, ValueRange{cmp});
         }
         maskRel = semiJoinOp.getResult();
      } else {
         auto selOp = rewriter.create<relalg::SelectionOp>(loc, maskRel);
         {
            OpBuilder::InsertionGuard guard(rewriter);
            auto* selBlock = rewriter.createBlock(&selOp.getPredicate());
            auto tupleArgSel = selBlock->addArgument(tuples::TupleType::get(ctx), loc);
            rewriter.setInsertionPointToEnd(selBlock);

            auto val = rewriter.create<tuples::GetColumnOp>(loc, maskValType, renamedMaskMeta.val, tupleArgSel);

            Value zero;
            if (sring.isInteger(64) || sring == SemiringTypes::forInt(ctx))
               zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
            else if (sring.isF64() || sring == SemiringTypes::forReal(ctx))
               zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(rewriter.getF64Type(), 0.0));
            else if (sring == SemiringTypes::forTropInt(ctx))
               zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(std::numeric_limits<int64_t>::max()));
            else if (sring == SemiringTypes::forTropMaxInt(ctx))
               zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(std::numeric_limits<int64_t>::min()));
            else if (sring == SemiringTypes::forTropReal(ctx))
               zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(rewriter.getF64Type(), std::numeric_limits<double>::infinity()));

            Value cmpZero;
            if (maskValType.isIntOrIndex()) {
               cmpZero = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, val, zero);
            } else {
               cmpZero = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, val, zero);
            }

            rewriter.create<tuples::ReturnOp>(loc, ValueRange{cmpZero});
         }
         maskRel = selOp.getResult();
      }

      auto* joinBlock = new Block;
      {
         auto tupleArg = joinBlock->addArgument(tuples::TupleType::get(ctx), loc);
         Value cmpAcc;
         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToEnd(joinBlock);

         if (inputMeta.hasRow() && renamedMaskMeta.hasRow()) {
            auto valA = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), inputMeta.row, tupleArg);
            auto valB = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), renamedMaskMeta.row, tupleArg);
            cmpAcc = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, valA, valB);
         }
         if (inputMeta.hasCol() && renamedMaskMeta.hasCol()) {
            auto valA = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), inputMeta.col, tupleArg);
            auto valB = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), renamedMaskMeta.col, tupleArg);
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

static LogicalResult convertLoop(Operation* op, ValueRange origInitArgs, ValueRange convInitArgs, ConversionPatternRewriter& rewriter, Value startBoundVal, Value endBoundVal, Value stepBoundVal, subop::MemberManager& memberManager, ConversionState& state) {
   auto loc = op->getLoc();
   auto* ctx = rewriter.getContext();
   auto& colManager = ctx->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

   llvm::SmallSetVector<Value, 4> capturedMatrices;
   Region& oldRegion = op->getRegion(0);
   oldRegion.walk([&](Operation* nestedOp) {
      for (Value operand : nestedOp->getOperands()) {
         if (llvm::isa<MatrixType>(operand.getType())) {
            Region* defRegion = operand.getParentRegion();
            if (!defRegion || !oldRegion.isAncestor(defRegion)) {
               capturedMatrices.insert(operand);
            }
         }
      }
   });

   llvm::DenseMap<Value, Value> capturedToScan;
   for (Value origArg : capturedMatrices) {
      Value convArg = rewriter.getRemappedValue(origArg);
      if (!convArg) {
         convArg = rewriter.create<UnrealizedConversionCastOp>(loc, tuples::TupleStreamType::get(ctx), origArg).getResult(0);
      } else if (!llvm::isa<tuples::TupleStreamType>(convArg.getType())) {
         convArg = rewriter.create<UnrealizedConversionCastOp>(loc, tuples::TupleStreamType::get(ctx), convArg).getResult(0);
      }

      MatrixMeta meta = state.get(origArg, ctx);
      Type valType = GraphAlgTypeConverter::convertSemiringType(meta.semiring);

      SmallVector<subop::Member> memberList;
      SmallVector<std::pair<subop::Member, tuples::ColumnRefAttr>> refMappingArgs;

      auto addMember = [&](StringRef name, Type type, tuples::ColumnRefAttr colRef) {
         subop::Member member = memberManager.createMember(name.str(), type);
         memberList.push_back(member);
         refMappingArgs.push_back({member, colRef});
      };

      if (meta.hasRow()) addMember("row", rewriter.getI64Type(), meta.row);
      if (meta.hasCol()) addMember("col", rewriter.getI64Type(), meta.col);
      addMember("val", valType, meta.val);

      auto stateMembersAttr = subop::StateMembersAttr::get(ctx, memberList);
      auto bufferType = subop::BufferType::get(ctx, stateMembersAttr);

      Value bufferState = rewriter.create<subop::GenericCreateOp>(loc, bufferType);
      auto mappingAttr = subop::ColumnRefMemberMappingAttr::get(ctx, refMappingArgs);
      rewriter.create<subop::MaterializeOp>(loc, convArg, bufferState, mappingAttr);

      capturedToScan[origArg] = bufferState;
   }

   SmallVector<Value> initStates;
   SmallVector<Type> stateTypes;

   for (size_t i = 0; i < origInitArgs.size(); ++i) {
      Value origArg = origInitArgs[i];
      Value convArg = convInitArgs[i];

      if (!llvm::isa<tuples::TupleStreamType>(convArg.getType())) {
         convArg = rewriter.create<UnrealizedConversionCastOp>(loc, tuples::TupleStreamType::get(ctx), convArg).getResult(0);
      }

      MatrixMeta meta = state.get(origArg, ctx);
      Type valType = GraphAlgTypeConverter::convertSemiringType(meta.semiring);

      SmallVector<subop::Member> memberList;
      SmallVector<std::pair<subop::Member, tuples::ColumnRefAttr>> refMappingArgs;

      auto addMember = [&](StringRef name, Type type, tuples::ColumnRefAttr colRef) {
         subop::Member member = memberManager.createMember(name.str(), type);
         memberList.push_back(member);
         refMappingArgs.push_back({member, colRef});
      };

      if (meta.hasRow()) addMember("row", rewriter.getI64Type(), meta.row);
      if (meta.hasCol()) addMember("col", rewriter.getI64Type(), meta.col);
      addMember("val", valType, meta.val);

      auto stateMembersAttr = subop::StateMembersAttr::get(ctx, memberList);
      auto bufferType = subop::BufferType::get(ctx, stateMembersAttr);
      stateTypes.push_back(bufferType);

      auto mappingAttr = subop::ColumnRefMemberMappingAttr::get(ctx, refMappingArgs);

      Value bufferState = rewriter.create<subop::GenericCreateOp>(loc, bufferType);
      rewriter.create<subop::MaterializeOp>(loc, convArg, bufferState, mappingAttr);
      initStates.push_back(bufferState);
   }

   SmallVector<Value> loopOperands;
   loopOperands.push_back(startBoundVal);
   loopOperands.append(initStates.begin(), initStates.end());

   SmallVector<Type> loopResultTypes;
   for (Value v : loopOperands) loopResultTypes.push_back(v.getType());

   auto loopOp = rewriter.create<subop::LoopOp>(loc, loopResultTypes, loopOperands);
   Block* newBody = rewriter.createBlock(&loopOp.getRegion());

   Value newIdx = newBody->addArgument(startBoundVal.getType(), loc);
   SmallVector<Value> newStates;
   for (Type t : stateTypes) newStates.push_back(newBody->addArgument(t, loc));

   rewriter.setInsertionPointToStart(newBody);

   llvm::DenseMap<Value, Value> capturedScans;
   for (auto& pair : capturedToScan) {
      Value origArg = pair.first;
      Value bufferState = pair.second;

      MatrixMeta loopMeta = state.get(origArg, ctx);
      auto bufferType = llvm::cast<subop::BufferType>(bufferState.getType());
      auto membersList = bufferType.getMembers().getMembers();
      size_t mIdx = 0;

      SmallVector<Attribute> scanComputedCols;
      SmallVector<Attribute> scanColumnMapping;

      auto addGatherMapping = [&](tuples::ColumnRefAttr ref, tuples::ColumnDefAttr def) {
         if (!def) {
            def = colManager.createDef(&ref.getColumn());
         }
         scanComputedCols.push_back(def);
         scanColumnMapping.push_back(rewriter.getStringAttr(memberManager.getName(membersList[mIdx++])));
      };

      if (loopMeta.hasRow()) addGatherMapping(loopMeta.row, loopMeta.rowDef);
      if (loopMeta.hasCol()) addGatherMapping(loopMeta.col, loopMeta.colDef);
      addGatherMapping(loopMeta.val, loopMeta.valDef);

      auto scanOp = rewriter.create<relalg::BufferScanOp>(
         loc, tuples::TupleStreamType::get(ctx),
         bufferState,
         rewriter.getArrayAttr(scanComputedCols),
         rewriter.getArrayAttr(scanColumnMapping));
      scanOp->setAttr("rows", rewriter.getF64FloatAttr(100.0));

      Value scanStream = scanOp.getResult();
      state.set(scanStream, loopMeta);
      capturedScans[origArg] = scanStream;
   }

   Value origIdxRepl;
   Value origArg0 = op->getRegion(0).front().getArgument(0);
   Type origArg0Type = origArg0.getType();
   if (origArg0Type.isIndex()) {
      origIdxRepl = newIdx;
   } else if (origArg0Type.isIntOrFloat()) {
      if (origArg0Type.isInteger(64))
         origIdxRepl = rewriter.create<arith::IndexCastOp>(loc, origArg0Type, newIdx);
      else if (origArg0Type.isF64()) {
         auto i64Idx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), newIdx);
         origIdxRepl = rewriter.create<arith::SIToFPOp>(loc, origArg0Type, i64Idx);
      } else {
         origIdxRepl = rewriter.create<arith::IndexCastOp>(loc, origArg0Type, newIdx);
      }
   } else if (auto matType = llvm::dyn_cast<MatrixType>(origArg0Type)) {
      MatrixMeta idxMeta = state.get(origArg0, ctx);
      Type loopValType = GraphAlgTypeConverter::convertSemiringType(matType.getSemiring());
      Value valIdx;
      if (loopValType.isInteger(64))
         valIdx = rewriter.create<arith::IndexCastOp>(loc, loopValType, newIdx);
      else if (loopValType.isF64()) {
         Value i64Idx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), newIdx);
         valIdx = rewriter.create<arith::SIToFPOp>(loc, loopValType, i64Idx);
      } else {
         valIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), newIdx);
      }

      auto constDef = createColumnDef(ctx, AttributeGenerator::nextName("dummy"), loopValType);

      Attribute zeroAttr;
      if (loopValType.isInteger(1)) {
         zeroAttr = rewriter.getIntegerAttr(rewriter.getI1Type(), 0);
      } else if (loopValType.isInteger(64)) {
         zeroAttr = rewriter.getI64IntegerAttr(0);
      } else {
         zeroAttr = rewriter.getFloatAttr(loopValType, 0.0);
      }

      Value constRel = rewriter.create<relalg::ConstRelationOp>(
                                  loc, rewriter.getArrayAttr({constDef}),
                                  rewriter.getArrayAttr({rewriter.getArrayAttr({zeroAttr})}))
                          .getResult();

      auto mapOp = rewriter.create<relalg::MapOp>(loc, constRel);
      mapOp.setComputedColsAttr(rewriter.getArrayAttr({idxMeta.valDef}));

      auto* mapBlock = new Block;
      mapOp.getPredicate().push_back(mapBlock);
      mapBlock->addArgument(tuples::TupleType::get(ctx), loc);
      {
         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(mapBlock);
         rewriter.create<tuples::ReturnOp>(loc, ValueRange{valIdx});
      }
      origIdxRepl = mapOp.getResult();
      state.set(origIdxRepl, idxMeta);
   }

   SmallVector<Value> origStateRepls;
   for (size_t i = 0; i < newStates.size(); ++i) {
      Value loopState = newStates[i];
      Value origArg = op->getRegion(0).front().getArgument(i + 1);
      MatrixMeta loopMeta = state.get(origArg, ctx);

      SmallVector<Attribute> scanComputedCols;
      SmallVector<Attribute> scanColumnMapping;

      auto bufferType = llvm::cast<subop::BufferType>(loopState.getType());
      auto membersList = bufferType.getMembers().getMembers();
      size_t mIdx = 0;

      auto addGatherMapping = [&](tuples::ColumnRefAttr ref, tuples::ColumnDefAttr def) {
         if (!def) {
            def = colManager.createDef(&ref.getColumn());
         }
         scanComputedCols.push_back(def);
         scanColumnMapping.push_back(rewriter.getStringAttr(memberManager.getName(membersList[mIdx++])));
      };

      if (loopMeta.hasRow()) addGatherMapping(loopMeta.row, loopMeta.rowDef);
      if (loopMeta.hasCol()) addGatherMapping(loopMeta.col, loopMeta.colDef);
      addGatherMapping(loopMeta.val, loopMeta.valDef);

      auto scanOp = rewriter.create<relalg::BufferScanOp>(
         loc, tuples::TupleStreamType::get(ctx),
         loopState,
         rewriter.getArrayAttr(scanComputedCols),
         rewriter.getArrayAttr(scanColumnMapping));
      scanOp->setAttr("rows", rewriter.getF64FloatAttr(100.0));

      Value scanStream = scanOp.getResult();
      state.set(scanStream, loopMeta);
      origStateRepls.push_back(scanStream);
   }

   SmallVector<Value> replValues;
   replValues.push_back(origIdxRepl);
   replValues.append(origStateRepls.begin(), origStateRepls.end());

   Block& oldBody = op->getRegion(0).front();
   rewriter.mergeBlocks(&oldBody, newBody, replValues);

   newBody->walk([&](Operation* nestedOp) {
      for (auto& pair : capturedScans) {
         nestedOp->replaceUsesOfWith(pair.first, pair.second);
      }
   });

   auto yieldOp = newBody->getTerminator();
   rewriter.setInsertionPoint(yieldOp);

   Value nextIdx = rewriter.create<arith::AddIOp>(loc, newIdx, stepBoundVal);
   Value cond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, nextIdx, endBoundVal);

   state.loopYieldData[yieldOp] = {nextIdx, cond};

   rewriter.setInsertionPointAfter(loopOp);
   SmallVector<Value> finalResults;
   for (size_t i = 0; i < op->getNumResults(); ++i) {
      Value finalState = loopOp.getResult(i + 1);
      MatrixMeta finalMeta = state.get(op->getResult(i), ctx);

      SmallVector<Attribute> scanComputedCols;
      SmallVector<Attribute> scanColumnMapping;

      auto bufferType = llvm::cast<subop::BufferType>(finalState.getType());
      auto membersList = bufferType.getMembers().getMembers();
      size_t mIdx = 0;

      auto addGatherMapping = [&](tuples::ColumnRefAttr ref, tuples::ColumnDefAttr def) {
         if (!def) {
            def = colManager.createDef(&ref.getColumn());
         }
         scanComputedCols.push_back(def);
         scanColumnMapping.push_back(rewriter.getStringAttr(memberManager.getName(membersList[mIdx++])));
      };

      if (finalMeta.hasRow()) addGatherMapping(finalMeta.row, finalMeta.rowDef);
      if (finalMeta.hasCol()) addGatherMapping(finalMeta.col, finalMeta.colDef);
      addGatherMapping(finalMeta.val, finalMeta.valDef);

      auto scanOp = rewriter.create<relalg::BufferScanOp>(
         loc, tuples::TupleStreamType::get(ctx),
         finalState,
         rewriter.getArrayAttr(scanComputedCols),
         rewriter.getArrayAttr(scanColumnMapping));
      scanOp->setAttr("rows", rewriter.getF64FloatAttr(100.0));

      Value finalStream = scanOp.getResult();
      state.set(finalStream, finalMeta);
      finalResults.push_back(finalStream);
   }

   rewriter.replaceOp(op, finalResults);
   return success();
}

class GraphAlgYieldOpConversion : public StatefulConversion<YieldOp> {
   public:
   using StatefulConversion::StatefulConversion;

   LogicalResult matchAndRewrite(YieldOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto it = state.loopYieldData.find(op.getOperation());
      if (it == state.loopYieldData.end()) return failure();

      auto loopOp = llvm::dyn_cast_or_null<subop::LoopOp>(op->getParentOp());
      if (!loopOp) return failure();

      auto loc = op.getLoc();
      auto* ctx = rewriter.getContext();
      auto& memberManager = ctx->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();

      Value nextIdx = it->second.first;
      Value cond = it->second.second;

      subop::Member condMember = memberManager.createMember("loop_cond", rewriter.getI1Type());
      auto condMembersAttr = subop::StateMembersAttr::get(ctx, {condMember});
      Type simpleStateType = subop::SimpleStateType::get(ctx, condMembersAttr);

      auto createStateOp = rewriter.create<subop::CreateSimpleStateOp>(loc, simpleStateType);
      {
         OpBuilder::InsertionGuard guard(rewriter);
         Block* initBlock = rewriter.createBlock(&createStateOp.getInitFn());
         rewriter.setInsertionPointToStart(initBlock);
         rewriter.create<tuples::ReturnOp>(loc, ValueRange{cond});
      }

      SmallVector<Value> yieldArgs;
      yieldArgs.push_back(nextIdx);

      for (size_t i = 0; i < op.getNumOperands(); ++i) {
         Value origYieldOperand = op.getOperand(i);
         Value streamOperand = adaptor.getOperands()[i];

         if (!llvm::isa<tuples::TupleStreamType>(streamOperand.getType())) {
            streamOperand = rewriter.create<UnrealizedConversionCastOp>(loc, tuples::TupleStreamType::get(ctx), streamOperand).getResult(0);
         }

         MatrixMeta actualYieldMeta = state.get(origYieldOperand, ctx);

         auto bufferType = llvm::cast<subop::BufferType>(loopOp.getOperand(i + 1).getType());
         auto members = bufferType.getMembers().getMembers();

         SmallVector<std::pair<subop::Member, tuples::ColumnRefAttr>> refMappingArgs;
         size_t memberIdx = 0;

         auto addMapping = [&](tuples::ColumnRefAttr colRef) {
            refMappingArgs.push_back({members[memberIdx++], colRef});
         };

         if (actualYieldMeta.hasRow()) addMapping(actualYieldMeta.row);
         if (actualYieldMeta.hasCol()) addMapping(actualYieldMeta.col);
         addMapping(actualYieldMeta.val);

         auto mappingAttr = subop::ColumnRefMemberMappingAttr::get(ctx, refMappingArgs);

         Value newState = rewriter.create<subop::GenericCreateOp>(loc, bufferType);
         rewriter.create<subop::MaterializeOp>(loc, streamOperand, newState, mappingAttr);

         yieldArgs.push_back(newState);
      }

      auto condMemberAttr = subop::MemberAttr::get(ctx, condMember);
      rewriter.replaceOpWithNewOp<subop::LoopContinueOp>(op, createStateOp.getResult(), condMemberAttr, yieldArgs);

      return success();
   }
};

static mlir::IntegerAttr tryGetConstantInt(mlir::Value v) {
   mlir::Attribute attr;
   if (!mlir::matchPattern(v, mlir::m_Constant(&attr))) {
      return nullptr;
   }
   return llvm::cast<mlir::IntegerAttr>(attr);
}

class ForConstOpConversion : public StatefulConversion<ForConstOp> {
   public:
   using StatefulConversion::StatefulConversion;

   LogicalResult matchAndRewrite(ForConstOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op.getLoc();
      auto* ctx = rewriter.getContext();
      auto& memberManager = ctx->getOrLoadDialect<subop::SubOperatorDialect>()->getMemberManager();

      auto rangeBeginAttr = tryGetConstantInt(op.getRangeBegin());
      auto rangeEndAttr = tryGetConstantInt(op.getRangeEnd());
      int64_t startIdx = rangeBeginAttr ? rangeBeginAttr.getInt() : 0;
      int64_t endIdx = rangeEndAttr ? rangeEndAttr.getInt() : 1;

      Value startBoundVal = rewriter.create<arith::ConstantIndexOp>(loc, startIdx);
      Value endBoundVal = rewriter.create<arith::ConstantIndexOp>(loc, endIdx);
      Value stepBoundVal = rewriter.create<arith::ConstantIndexOp>(loc, 1);

      return convertLoop(op, op.getInitArgs(), adaptor.getInitArgs(), rewriter, startBoundVal, endBoundVal, stepBoundVal, memberManager, state);
   }
};

class ForDimOpConversion : public StatefulConversion<ForDimOp> {
   public:
   using StatefulConversion::StatefulConversion;

   LogicalResult matchAndRewrite(ForDimOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op.getLoc();
      auto* ctx = rewriter.getContext();
      auto& memberManager = ctx->getOrLoadDialect<subop::SubOperatorDialect>()->getMemberManager();

      auto dim = op.getDim();
      int64_t dimSize = dim.isConcrete() ? dim.getConcreteDim() : state.globalDimSizes[dim];

      Value startBoundVal = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value endBoundVal = rewriter.create<arith::ConstantIndexOp>(loc, dimSize);
      Value stepBoundVal = rewriter.create<arith::ConstantIndexOp>(loc, 1);

      return convertLoop(op, op.getInitArgs(), adaptor.getInitArgs(), rewriter, startBoundVal, endBoundVal, stepBoundVal, memberManager, state);
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

   ConversionState state;
   // PRE-PASS: Statically detect Matrix Sizes!
   resolveDimensionsGlobally(getOperation(), state);

   GraphAlgTypeConverter typeConverter(context);

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
      PickAnyOpConversion,
      CastDimOpConversion>(typeConverter, context, state);

   patterns.add<
      ApplyReturnOpConversion,
      AddConversion,
      MulConversion,
      EqConversion,
      CastScalarOpConversion,
      ConstantOpConversion>(typeConverter, context);

   patterns.add<
      ForConstOpConversion,
      ForDimOpConversion,
      GraphAlgYieldOpConversion>(typeConverter, context, state);

   if (failed(applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
   }
}

std::unique_ptr<OperationPass<ModuleOp>> createGraphAlgToRelAlgPass() {
   return std::make_unique<GraphAlgToRelAlgPass>();
}

} // namespace graphalg