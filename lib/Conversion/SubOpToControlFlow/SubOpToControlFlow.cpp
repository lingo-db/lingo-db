#include "mlir-support/parsing.h"
#include "mlir/Conversion/SubOpToControlFlow/SubOpToControlFlowPass.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/SubOperator/SubOperatorDialect.h"
#include "mlir/Dialect/SubOperator/SubOperatorOps.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include <mlir/Dialect/util/FunctionHelper.h>

using namespace mlir;

namespace {
struct SubOpToControlFlowLoweringPass
   : public PassWrapper<SubOpToControlFlowLoweringPass, OperationPass<ModuleOp>> {
   virtual llvm::StringRef getArgument() const override { return "lower-subop-to-cf"; }

   SubOpToControlFlowLoweringPass() {}
   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<LLVM::LLVMDialect, mlir::db::DBDialect, scf::SCFDialect, mlir::cf::ControlFlowDialect, util::UtilDialect, memref::MemRefDialect, arith::ArithmeticDialect, mlir::dsa::DSADialect, mlir::subop::SubOperatorDialect>();
   }
   void runOnOperation() final;
};
class RemoveUnusedInFlight : public OpConversionPattern<mlir::subop::InFlightOp> {
   public:
   using OpConversionPattern<mlir::subop::InFlightOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::subop::InFlightOp inFlightOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (inFlightOp.use_empty()) {
         rewriter.eraseOp(inFlightOp);
         return success();
      }
      return failure();
   }
};
class ColumnMapping {
   std::unordered_map<const mlir::tuples::Column*, mlir::Value> mapping;

   public:
   ColumnMapping() : mapping() {}
   ColumnMapping(mlir::subop::InFlightOp inFlightOp) {
      assert(inFlightOp.columns().size() == inFlightOp.values().size());
      for (auto i = 0ul; i < inFlightOp.columns().size(); i++) {
         const auto* col = &inFlightOp.columns()[i].cast<mlir::tuples::ColumnDefAttr>().getColumn();
         auto val = inFlightOp.values()[i];
         mapping.insert(std::make_pair(col, val));
      }
   }
   mlir::Value resolve(mlir::tuples::ColumnRefAttr ref) {
      return mapping.at(&ref.getColumn());
   }
   std::vector<mlir::Value> resolve(mlir::ArrayAttr arr) {
      std::vector<mlir::Value> res;
      for (auto attr : arr) {
         res.push_back(resolve(attr.cast<mlir::tuples::ColumnRefAttr>()));
      }
      return res;
   }
   mlir::Value createInFlight(mlir::OpBuilder& builder) {
      std::vector<mlir::Value> values;
      std::vector<mlir::Attribute> columns;

      for (auto m : mapping) {
         columns.push_back(builder.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager().createDef(m.first));
         values.push_back(m.second);
      }
      return builder.create<mlir::subop::InFlightOp>(builder.getUnknownLoc(), values, builder.getArrayAttr(columns));
   }
   void define(mlir::tuples::ColumnDefAttr columnDefAttr, mlir::Value v) {
      mapping.insert(std::make_pair(&columnDefAttr.getColumn(), v));
   }
   void define(mlir::ArrayAttr columns, mlir::ValueRange values) {
      for (auto i = 0ul; i < columns.size(); i++) {
         define(columns[i].cast<mlir::tuples::ColumnDefAttr>(), values[i]);
      }
   }
};
class FilterLowering : public OpConversionPattern<mlir::subop::FilterOp> {
   public:
   using OpConversionPattern<mlir::subop::FilterOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::subop::FilterOp filterOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (auto inFlightOp = mlir::dyn_cast_or_null<mlir::subop::InFlightOp>(adaptor.stream().getDefiningOp())) {
         rewriter.setInsertionPointAfter(inFlightOp);
         ColumnMapping mapping(inFlightOp);
         mlir::Value cond = rewriter.create<mlir::db::AndOp>(filterOp.getLoc(), mapping.resolve(filterOp.conditions()));
         cond = rewriter.create<mlir::db::DeriveTruth>(filterOp.getLoc(), cond);
         mlir::Value newInFlight;
         rewriter.create<mlir::scf::IfOp>(
            filterOp->getLoc(), mlir::TypeRange{}, cond, [&](mlir::OpBuilder& builder1, mlir::Location) {
               newInFlight=mapping.createInFlight(builder1);
               builder1.create<mlir::scf::YieldOp>(filterOp->getLoc()); });
         rewriter.replaceOp(filterOp, newInFlight);

         return success();
      }
      return failure();
   }
};
class MapLowering : public OpConversionPattern<mlir::subop::MapOp> {
   public:
   using OpConversionPattern<mlir::subop::MapOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::subop::MapOp mapOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (auto inFlightOp = mlir::dyn_cast_or_null<mlir::subop::InFlightOp>(adaptor.rel().getDefiningOp())) {
         rewriter.setInsertionPointAfter(inFlightOp);
         ColumnMapping mapping(inFlightOp);

         {
            std::vector<mlir::Operation*> toErase;
            auto cloned = mapOp.clone();

            mlir::Block* source = &cloned.fn().front();
            auto* terminator = source->getTerminator();

            source->walk([&](mlir::tuples::GetColumnOp getColumnOp) {
               getColumnOp.replaceAllUsesWith(mapping.resolve(getColumnOp.attr()));
               toErase.push_back(getColumnOp.getOperation());
            });
            for (auto* op : toErase) {
               op->dropAllUses();
               op->erase();
            }
            source->dump();
            auto returnOp = mlir::cast<mlir::tuples::ReturnOp>(terminator);
            std::vector<Value> res(returnOp.results().begin(), returnOp.results().end());
            std::vector<mlir::Operation*> toInsert;
            for (auto& x : source->getOperations()) {
               llvm::dbgs() << "start...\n";
               x.dump();
               toInsert.push_back(&x);
               llvm::dbgs() << "next...\n";
            }
            for (auto* x : toInsert) {
               x->remove();
               rewriter.insert(x);
            }
            rewriter.eraseOp(terminator);
            cloned->erase();
            mapping.define(mapOp.computed_cols(), res);
         }
         mlir::Value newInFlight = mapping.createInFlight(rewriter);
         rewriter.replaceOp(mapOp, newInFlight);
         return success();
      }
      return failure();
   }
};
class GetTableRefLowering : public OpConversionPattern<mlir::subop::GetReferenceOp> {
   public:
   using OpConversionPattern<mlir::subop::GetReferenceOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::subop::GetReferenceOp getReferenceOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!getReferenceOp.getType().isa<mlir::subop::TableRefType>()) return failure();
      rewriter.replaceOpWithNewOp<mlir::dsa::ScanSource>(getReferenceOp, typeConverter->convertType(getReferenceOp.getType()), getReferenceOp.descrAttr());
      return mlir::success();
   }
};
class ConvertToExplicitTableLowering : public OpConversionPattern<mlir::subop::ConvertToExplicit> {
   public:
   using OpConversionPattern<mlir::subop::ConvertToExplicit>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::subop::ConvertToExplicit convertToExplicitOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto sourceType = convertToExplicitOp.state().getType().dyn_cast<mlir::subop::TableType>();
      auto targetType = convertToExplicitOp.getType().dyn_cast<mlir::dsa::TableType>();
      if (sourceType && targetType) {
         rewriter.replaceOpWithNewOp<mlir::dsa::Finalize>(convertToExplicitOp, targetType, adaptor.state());
         return mlir::success();
      }
      return mlir::failure();
   }
};
class MaterializeTableLowering : public OpConversionPattern<mlir::subop::MaterializeOp> {
   public:
   using OpConversionPattern<mlir::subop::MaterializeOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::subop::MaterializeOp materializeOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!materializeOp.state().getType().isa<mlir::subop::TableType>()) return failure();
      if (auto inFlightOp = mlir::dyn_cast_or_null<mlir::subop::InFlightOp>(adaptor.stream().getDefiningOp())) {
         rewriter.setInsertionPointAfter(inFlightOp);
         ColumnMapping mapping(inFlightOp);
         auto stateType = materializeOp.state().getType().cast<mlir::subop::TableType>();
         auto state = adaptor.state();
         for (size_t i = 0; i < stateType.getColumns().getTypes().size(); i++) {
            auto memberName = stateType.getColumns().getNames()[i].cast<mlir::StringAttr>().str();
            auto attribute = materializeOp.mapping().get(memberName).cast<mlir::tuples::ColumnRefAttr>();
            auto val = mapping.resolve(attribute);
            mlir::Value valid;
            if (val.getType().isa<mlir::db::NullableType>()) {
               valid = rewriter.create<mlir::db::IsNullOp>(materializeOp->getLoc(), val);
               valid = rewriter.create<mlir::db::NotOp>(materializeOp->getLoc(), valid);
               val = rewriter.create<mlir::db::NullableGetVal>(materializeOp->getLoc(), getBaseType(val.getType()), val);
            }
            rewriter.create<mlir::dsa::Append>(materializeOp->getLoc(), state, val, valid);
         }
         rewriter.create<mlir::dsa::NextRow>(materializeOp->getLoc(), state);
         rewriter.eraseOp(materializeOp);
         return mlir::success();
      }
      return mlir::failure();
   }
};
class CreateTableLowering : public OpConversionPattern<mlir::subop::CreateOp> {
   public:
   using OpConversionPattern<mlir::subop::CreateOp>::OpConversionPattern;
   std::string arrowDescrFromType(mlir::Type type) const {
      if (isIntegerType(type, 1)) {
         return "bool";
      } else if (auto intWidth = getIntegerWidth(type, false)) {
         return "int[" + std::to_string(intWidth) + "]";
      } else if (auto uIntWidth = getIntegerWidth(type, true)) {
         return "uint[" + std::to_string(uIntWidth) + "]";
      } else if (auto decimalType = type.dyn_cast_or_null<mlir::db::DecimalType>()) {
         // TODO: actually handle cases where 128 bits are insufficient.
         auto prec = std::min(decimalType.getP(), 38);
         return "decimal[" + std::to_string(prec) + "," + std::to_string(decimalType.getS()) + "]";
      } else if (auto floatType = type.dyn_cast_or_null<mlir::FloatType>()) {
         return "float[" + std::to_string(intWidth) + "]";
      } else if (auto stringType = type.dyn_cast_or_null<mlir::db::StringType>()) {
         return "string";
      } else if (auto dateType = type.dyn_cast_or_null<mlir::db::DateType>()) {
         if (dateType.getUnit() == mlir::db::DateUnitAttr::day) {
            return "date[32]";
         } else {
            return "date[64]";
         }
      } else if (auto charType = type.dyn_cast_or_null<mlir::db::CharType>()) {
         return "fixed_sized[" + std::to_string(charType.getBytes()) + "]";
      } else if (auto intervalType = type.dyn_cast_or_null<mlir::db::IntervalType>()) {
         if (intervalType.getUnit() == mlir::db::IntervalUnitAttr::months) {
            return "interval_months";
         } else {
            return "interval_daytime";
         }
      } else if (auto timestampType = type.dyn_cast_or_null<mlir::db::TimestampType>()) {
         return "timestamp[" + std::to_string(static_cast<uint32_t>(timestampType.getUnit())) + "]";
      }
      return "";
   }
   LogicalResult matchAndRewrite(mlir::subop::CreateOp createOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!createOp.getType().isa<mlir::subop::TableType>()) return failure();
      auto tableType = createOp.getType().cast<mlir::subop::TableType>();
      std::string descr;
      for (size_t i = 0; i < tableType.getColumns().getTypes().size(); i++) {
         if (!descr.empty()) {
            descr += ";";
         }
         descr += tableType.getColumns().getNames()[i].cast<mlir::StringAttr>().str() + ":" + arrowDescrFromType(getBaseType(tableType.getColumns().getTypes()[i].cast<mlir::TypeAttr>().getValue()));
      }
      rewriter.replaceOpWithNewOp<mlir::dsa::CreateDS>(createOp, typeConverter->convertType(tableType), rewriter.getStringAttr(descr));
      return mlir::success();
   }
};
class ScanTableRefLowering : public OpConversionPattern<mlir::subop::ScanOp> {
   public:
   using OpConversionPattern<mlir::subop::ScanOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::subop::ScanOp scanOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!scanOp.state().getType().isa<mlir::subop::TableRefType>()) return failure();
      auto columns = scanOp.state().getType().cast<mlir::subop::TableRefType>().getColumns();
      ColumnMapping mapping;
      auto state = adaptor.state();
      auto recordBatchType = state.getType().cast<mlir::dsa::GenericIterableType>().getElementType().cast<mlir::dsa::RecordBatchType>();
      auto forOp = rewriter.create<mlir::dsa::ForOp>(scanOp->getLoc(), mlir::TypeRange{}, state, mlir::Value(), mlir::ValueRange{});
      mlir::Block* block = new mlir::Block;
      block->addArgument(recordBatchType, scanOp->getLoc());
      forOp.getBodyRegion().push_back(block);
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(block);
         auto forOp2 = rewriter.create<mlir::dsa::ForOp>(scanOp->getLoc(), mlir::TypeRange{}, forOp.getInductionVar(), mlir::Value(), mlir::ValueRange{});
         mlir::Block* block2 = new mlir::Block;
         block2->addArgument(recordBatchType.getElementType(), scanOp->getLoc());
         forOp2.getBodyRegion().push_back(block2);
         {
            mlir::OpBuilder::InsertionGuard guard2(rewriter);
            rewriter.setInsertionPointToStart(block2);

            for (auto i = 0ul; i < columns.getTypes().size(); i++) {
               auto type = columns.getTypes()[i].cast<mlir::TypeAttr>().getValue();
               auto name = columns.getNames()[i].cast<mlir::StringAttr>().str();
               if (scanOp.mapping().contains(name)) {
                  auto columnDefAttr = scanOp.mapping().get(name).cast<mlir::tuples::ColumnDefAttr>();
                  std::vector<mlir::Type> types;
                  types.push_back(getBaseType(type));
                  if (type.isa<mlir::db::NullableType>()) {
                     types.push_back(rewriter.getI1Type());
                  }
                  auto atOp = rewriter.create<mlir::dsa::At>(scanOp->getLoc(), types, forOp2.getInductionVar(), i);
                  if (type.isa<mlir::db::NullableType>()) {
                     mlir::Value isNull = rewriter.create<mlir::db::NotOp>(scanOp->getLoc(), atOp.valid());
                     mlir::Value val = rewriter.create<mlir::db::AsNullableOp>(scanOp->getLoc(), type, atOp.val(), isNull);
                     mapping.define(columnDefAttr, val);
                  } else {
                     mapping.define(columnDefAttr, atOp.val());
                  }
               }
            }

            mlir::Value newInFlight = mapping.createInFlight(rewriter);
            rewriter.replaceOp(scanOp, newInFlight);
            rewriter.create<mlir::dsa::YieldOp>(scanOp->getLoc());
         }
         rewriter.create<mlir::dsa::YieldOp>(scanOp->getLoc());
      }

      return success();
   }
};
void SubOpToControlFlowLoweringPass::runOnOperation() {
   auto module = getOperation();
   getContext().getLoadedDialect<mlir::util::UtilDialect>()->getFunctionHelper().setParentModule(module);

   // Define Conversion Target
   ConversionTarget target(getContext());
   target.addLegalOp<ModuleOp>();
   target.addLegalOp<UnrealizedConversionCastOp>();
   target.addIllegalDialect<subop::SubOperatorDialect>();
   target.addLegalDialect<db::DBDialect>();
   target.addLegalDialect<dsa::DSADialect>();

   target.addLegalDialect<tuples::TupleStreamDialect>();
   target.addLegalDialect<func::FuncDialect>();
   target.addLegalDialect<memref::MemRefDialect>();
   target.addLegalDialect<arith::ArithmeticDialect>();
   target.addLegalDialect<cf::ControlFlowDialect>();
   target.addLegalDialect<scf::SCFDialect>();
   target.addLegalDialect<util::UtilDialect>();
   target.addLegalOp<subop::InFlightOp>();
   auto *ctxt = &getContext();

   TypeConverter typeConverter;
   typeConverter.addConversion([](mlir::Type t) { return t; });
   auto unpackTypes = [](mlir::ArrayAttr arr) {
      std::vector<Type> res;
      for (auto x : arr) { res.push_back(x.cast<mlir::TypeAttr>().getValue()); }
      return res;
   };
   typeConverter.addConversion([&](mlir::subop::TableRefType t) -> Type {
      auto tupleType = mlir::TupleType::get(ctxt, unpackTypes(t.getColumns().getTypes()));
      auto recordBatch = mlir::dsa::RecordBatchType::get(ctxt, tupleType);
      mlir::Type chunkIterable = mlir::dsa::GenericIterableType::get(ctxt, recordBatch, "table_chunk_iterator");
      return chunkIterable;
   });
   typeConverter.addConversion([&](mlir::subop::TableType t) -> Type {
      auto tupleType = mlir::TupleType::get(ctxt, unpackTypes(t.getColumns().getTypes()));
      return mlir::dsa::TableBuilderType::get(ctxt, tupleType);
   });

   RewritePatternSet patterns(&getContext());
   patterns.insert<FilterLowering>(typeConverter, ctxt);
   patterns.insert<MapLowering>(typeConverter, ctxt);
   patterns.insert<GetTableRefLowering>(typeConverter, ctxt);
   patterns.insert<ScanTableRefLowering>(typeConverter, ctxt);
   patterns.insert<CreateTableLowering>(typeConverter, ctxt);
   patterns.insert<MaterializeTableLowering>(typeConverter, ctxt);
   patterns.insert<ConvertToExplicitTableLowering>(typeConverter, ctxt);
   if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
}
} //namespace
std::unique_ptr<mlir::Pass>
mlir::subop::createLowerSubOpPass() {
   return std::make_unique<SubOpToControlFlowLoweringPass>();
}
void mlir::subop::createLowerSubOpPipeline(mlir::OpPassManager& pm) {
   pm.addPass(mlir::subop::createLowerSubOpPass());
   pm.addPass(mlir::createCanonicalizerPass());
}
void mlir::subop::registerSubOpToControlFlowConversionPasses() {
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::subop::createLowerSubOpPass();
   });
   mlir::PassPipelineRegistration<EmptyPipelineOptions>(
      "lower-subop",
      "",
      mlir::subop::createLowerSubOpPipeline);
}