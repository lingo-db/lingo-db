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
#include "mlir/Dialect/SubOperator/Transforms/Passes.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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

static std::vector<mlir::Value> inlineBlock(mlir::Block* b, ConversionPatternRewriter& rewriter, mlir::ValueRange arguments) {
   auto* terminator = b->getTerminator();
   auto returnOp = mlir::cast<mlir::tuples::ReturnOp>(terminator);
   mlir::BlockAndValueMapping mapper;
   assert(b->getNumArguments() == arguments.size());
   for (auto i = 0ull; i < b->getNumArguments(); i++) {
      mapper.map(b->getArgument(i), arguments[i]);
   }
   for (auto& x : b->getOperations()) {
      if (&x != terminator) {
         rewriter.clone(x, mapper);
      }
   }
   std::vector<mlir::Value> res;
   for (auto val : returnOp.results()) {
      res.push_back(mapper.lookup(val));
   }
   return res;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////// State management ///////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

class CreateSimpleStateLowering : public OpConversionPattern<mlir::subop::CreateOp> {
   public:
   using OpConversionPattern<mlir::subop::CreateOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::subop::CreateOp createOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!createOp.getType().isa<mlir::subop::SimpleStateType>()) return failure();
      mlir::Value ref = rewriter.create<mlir::util::AllocOp>(createOp->getLoc(), typeConverter->convertType(createOp.getType()), mlir::Value());
      auto x = rewriter.create<mlir::tuples::ReturnOp>(createOp->getLoc());
      rewriter.mergeBlockBefore(&createOp.initFn().front().front(), x);
      auto terminator = mlir::cast<mlir::tuples::ReturnOp>(x->getPrevNode());
      auto packed = rewriter.create<mlir::util::PackOp>(createOp->getLoc(), terminator.results());
      rewriter.eraseOp(x);
      rewriter.eraseOp(terminator);
      rewriter.create<mlir::util::StoreOp>(createOp->getLoc(), packed, ref, mlir::Value());
      rewriter.replaceOp(createOp, ref);
      return mlir::success();
   }
};
class CreateHashMapLowering : public OpConversionPattern<mlir::subop::CreateOp> {
   public:
   using OpConversionPattern<mlir::subop::CreateOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::subop::CreateOp createOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!createOp.getType().isa<mlir::subop::HashMapType>()) return failure();
      mlir::Value hashmap = rewriter.create<mlir::dsa::CreateDS>(createOp->getLoc(), typeConverter->convertType(createOp.getType()));
      rewriter.replaceOp(createOp, hashmap);
      return mlir::success();
   }
};
class CreateLazyMultiMapLowering : public OpConversionPattern<mlir::subop::CreateOp> {
   public:
   using OpConversionPattern<mlir::subop::CreateOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::subop::CreateOp createOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!createOp.getType().isa<mlir::subop::LazyMultiMapType>()) return failure();
      mlir::Value hashmap = rewriter.create<mlir::dsa::CreateDS>(createOp->getLoc(), typeConverter->convertType(createOp.getType()));
      rewriter.replaceOp(createOp, hashmap);
      return mlir::success();
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
         return "float[" + std::to_string(floatType.getWidth()) + "]";
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
      auto columnNames = createOp.descrAttr().cast<mlir::ArrayAttr>();
      for (size_t i = 0; i < tableType.getMembers().getTypes().size(); i++) {
         if (!descr.empty()) {
            descr += ";";
         }
         descr += columnNames[i].cast<mlir::StringAttr>().str() + ":" + arrowDescrFromType(getBaseType(tableType.getMembers().getTypes()[i].cast<mlir::TypeAttr>().getValue()));
      }
      rewriter.replaceOpWithNewOp<mlir::dsa::CreateDS>(createOp, typeConverter->convertType(tableType), rewriter.getStringAttr(descr));
      return mlir::success();
   }
};
class CreateVectorLowering : public OpConversionPattern<mlir::subop::CreateOp> {
   public:
   using OpConversionPattern<mlir::subop::CreateOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::subop::CreateOp createOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!createOp.getType().isa<mlir::subop::VectorType>()) return failure();
      auto vectorType = createOp.getType().cast<mlir::subop::VectorType>();
      mlir::Value vector = rewriter.replaceOpWithNewOp<mlir::dsa::CreateDS>(createOp, typeConverter->convertType(vectorType), rewriter.getStringAttr(""));
      for (auto& b : createOp.initFn()) {
         auto res = inlineBlock(&b.front(), rewriter, {});
         auto packed = rewriter.create<mlir::util::PackOp>(createOp->getLoc(), res);
         rewriter.create<mlir::dsa::Append>(createOp->getLoc(), vector, packed);
      }
      return mlir::success();
   }
};
class MaintainOpLowering : public OpConversionPattern<mlir::subop::MaintainOp> {
   public:
   using OpConversionPattern<mlir::subop::MaintainOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::subop::MaintainOp maintainOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<mlir::dsa::Finalize>(maintainOp, adaptor.state());
      return mlir::success();
   }
};
class SortLowering : public OpConversionPattern<mlir::subop::SortOp> {
   public:
   using OpConversionPattern<mlir::subop::SortOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::subop::SortOp sortOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto dbSortOp = rewriter.create<mlir::dsa::SortOp>(sortOp->getLoc(), adaptor.toSort());
      mlir::Block* block2 = new mlir::Block;
      mlir::TupleType tupleType = adaptor.toSort().getType().cast<mlir::dsa::VectorType>().getElementType().cast<mlir::TupleType>();
      block2->addArgument(tupleType, sortOp->getLoc());
      block2->addArguments(tupleType, sortOp->getLoc());
      dbSortOp.region().push_back(block2);
      auto vectorType = sortOp.toSort().getType().cast<mlir::subop::VectorType>();
      std::unordered_map<std::string, size_t> memberPositions;
      for (auto i = 0ull; i < vectorType.getMembers().getTypes().size(); i++) {
         memberPositions.insert({vectorType.getMembers().getNames()[i].cast<mlir::StringAttr>().str(), i});
      }

      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(block2);
         auto unpackedLeft = rewriter.create<mlir::util::UnPackOp>(sortOp->getLoc(), block2->getArgument(0));
         auto unpackedRight = rewriter.create<mlir::util::UnPackOp>(sortOp->getLoc(), block2->getArgument(1));
         std::vector<mlir::Value> args;
         for (auto sortByMember : sortOp.sortBy()) {
            args.push_back(unpackedLeft.getResult(memberPositions[sortByMember.cast<mlir::StringAttr>().str()]));
         }
         for (auto sortByMember : sortOp.sortBy()) {
            args.push_back(unpackedRight.getResult(memberPositions[sortByMember.cast<mlir::StringAttr>().str()]));
         }
         auto x = rewriter.create<mlir::dsa::YieldOp>(sortOp->getLoc());
         rewriter.mergeBlockBefore(&sortOp.region().front(), x, args);
         auto returnOp = mlir::cast<mlir::tuples::ReturnOp>(x->getPrevNode());
         rewriter.create<mlir::dsa::YieldOp>(sortOp->getLoc(), returnOp.results());
         rewriter.eraseOp(x);
         rewriter.eraseOp(returnOp);
         rewriter.eraseOp(sortOp);
      }
      return mlir::success();
   }
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////  Support ////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
   ColumnMapping(mlir::subop::InFlightTupleOp inFlightOp) {
      assert(inFlightOp.columns().size() == inFlightOp.values().size());
      for (auto i = 0ul; i < inFlightOp.columns().size(); i++) {
         const auto* col = &inFlightOp.columns()[i].cast<mlir::tuples::ColumnDefAttr>().getColumn();
         auto val = inFlightOp.values()[i];
         mapping.insert(std::make_pair(col, val));
      }
   }
   void merge(mlir::subop::InFlightOp inFlightOp) {
      assert(inFlightOp.columns().size() == inFlightOp.values().size());
      for (auto i = 0ul; i < inFlightOp.columns().size(); i++) {
         const auto* col = &inFlightOp.columns()[i].cast<mlir::tuples::ColumnDefAttr>().getColumn();
         auto val = inFlightOp.values()[i];
         mapping.insert(std::make_pair(col, val));
      }
   }
   void merge(mlir::subop::InFlightTupleOp inFlightOp) {
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
   mlir::Value createInFlightTuple(mlir::OpBuilder& builder) {
      std::vector<mlir::Value> values;
      std::vector<mlir::Attribute> columns;

      for (auto m : mapping) {
         columns.push_back(builder.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager().createDef(m.first));
         values.push_back(m.second);
      }
      return builder.create<mlir::subop::InFlightTupleOp>(builder.getUnknownLoc(), values, builder.getArrayAttr(columns));
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////// Starting a TupleStream//////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class ScanTableRefLowering : public OpConversionPattern<mlir::subop::ScanOp> {
   public:
   using OpConversionPattern<mlir::subop::ScanOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::subop::ScanOp scanOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!scanOp.state().getType().isa<mlir::subop::TableRefType>()) return failure();
      auto columns = scanOp.state().getType().cast<mlir::subop::TableRefType>().getMembers();
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

class ScanSimpleStateLowering : public OpConversionPattern<mlir::subop::ScanOp> {
   public:
   using OpConversionPattern<mlir::subop::ScanOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::subop::ScanOp scanOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!scanOp.state().getType().isa<mlir::subop::SimpleStateType>()) return failure();
      auto columns = scanOp.state().getType().cast<mlir::subop::SimpleStateType>().getMembers();
      ColumnMapping mapping;
      auto loaded = rewriter.create<mlir::util::LoadOp>(scanOp->getLoc(), adaptor.state());
      auto unpackedValues = rewriter.create<mlir::util::UnPackOp>(scanOp->getLoc(), loaded).getResults();
      for (auto i = 0ul; i < columns.getTypes().size(); i++) {
         auto name = columns.getNames()[i].cast<mlir::StringAttr>().str();
         if (scanOp.mapping().contains(name)) {
            auto columnDefAttr = scanOp.mapping().get(name).cast<mlir::tuples::ColumnDefAttr>();
            mapping.define(columnDefAttr, unpackedValues[i]);
         }
      }
      mlir::Value newInFlight = mapping.createInFlight(rewriter);
      rewriter.replaceOp(scanOp, newInFlight);

      return success();
   }
};

class ScanVectorLowering : public OpConversionPattern<mlir::subop::ScanOp> {
   public:
   using OpConversionPattern<mlir::subop::ScanOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::subop::ScanOp scanOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!scanOp.state().getType().isa<mlir::subop::VectorType>()) return failure();
      auto columns = scanOp.state().getType().cast<mlir::subop::VectorType>().getMembers();
      ColumnMapping mapping;
      auto state = adaptor.state();
      auto vectorType = mlir::util::RefType::get(getContext(), state.getType().cast<mlir::dsa::VectorType>().getElementType());
      auto forOp = rewriter.create<mlir::dsa::ForOp>(scanOp->getLoc(), mlir::TypeRange{}, state, mlir::Value(), mlir::ValueRange{});
      mlir::Block* block = new mlir::Block;
      block->addArgument(vectorType, scanOp->getLoc());
      forOp.getBodyRegion().push_back(block);
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(block);
         auto loaded = rewriter.create<mlir::util::LoadOp>(scanOp->getLoc(), forOp.getInductionVar());
         auto unpackedValues = rewriter.create<mlir::util::UnPackOp>(scanOp->getLoc(), loaded).getResults();

         for (auto i = 0ul; i < columns.getTypes().size(); i++) {
            auto name = columns.getNames()[i].cast<mlir::StringAttr>().str();
            if (scanOp.mapping().contains(name)) {
               auto columnDefAttr = scanOp.mapping().get(name).cast<mlir::tuples::ColumnDefAttr>();
               mapping.define(columnDefAttr, unpackedValues[i]);
            }
         }

         mlir::Value newInFlight = mapping.createInFlight(rewriter);
         rewriter.replaceOp(scanOp, newInFlight);
         rewriter.create<mlir::dsa::YieldOp>(scanOp->getLoc());
      }

      return success();
   }
};
class ScanLazyMultiMapLowering : public OpConversionPattern<mlir::subop::ScanOp> {
   public:
   using OpConversionPattern<mlir::subop::ScanOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::subop::ScanOp scanOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!scanOp.state().getType().isa<mlir::subop::LazyMultiMapType>()) return failure();
      auto columns = scanOp.state().getType().cast<mlir::subop::LazyMultiMapType>().getValueMembers();
      ColumnMapping mapping;
      auto state = adaptor.state();
      auto vectorType = state.getType().cast<mlir::dsa::JoinHashtableType>().getElementType();
      auto forOp = rewriter.create<mlir::dsa::ForOp>(scanOp->getLoc(), mlir::TypeRange{}, state, mlir::Value(), mlir::ValueRange{});
      mlir::Block* block = new mlir::Block;
      block->addArgument(vectorType, scanOp->getLoc());
      forOp.getBodyRegion().push_back(block);
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(block);
         auto loaded = rewriter.create<mlir::util::UnPackOp>(scanOp->getLoc(), forOp.getInductionVar());
         auto unpackedValues = rewriter.create<mlir::util::UnPackOp>(scanOp->getLoc(), loaded.getResult(1)).getResults();

         for (auto i = 0ul; i < columns.getTypes().size(); i++) {
            auto name = columns.getNames()[i].cast<mlir::StringAttr>().str();
            if (scanOp.mapping().contains(name)) {
               auto columnDefAttr = scanOp.mapping().get(name).cast<mlir::tuples::ColumnDefAttr>();
               mapping.define(columnDefAttr, unpackedValues[i]);
            }
         }

         mlir::Value newInFlight = mapping.createInFlight(rewriter);
         rewriter.replaceOp(scanOp, newInFlight);
         rewriter.create<mlir::dsa::YieldOp>(scanOp->getLoc());
      }

      return success();
   }
};
class ScanRefsVectorLowering : public OpConversionPattern<mlir::subop::ScanRefsOp> {
   public:
   using OpConversionPattern<mlir::subop::ScanRefsOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::subop::ScanRefsOp scanOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!scanOp.state().getType().isa<mlir::subop::VectorType>()) return failure();
      ColumnMapping mapping;
      auto state = adaptor.state();
      auto vectorType = mlir::util::RefType::get(getContext(), state.getType().cast<mlir::dsa::VectorType>().getElementType());
      auto forOp = rewriter.create<mlir::dsa::ForOp>(scanOp->getLoc(), mlir::TypeRange{}, state, mlir::Value(), mlir::ValueRange{});
      mlir::Block* block = new mlir::Block;
      block->addArgument(vectorType, scanOp->getLoc());
      forOp.getBodyRegion().push_back(block);
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(block);
         mapping.define(scanOp.ref(), forOp.getInductionVar());

         mlir::Value newInFlight = mapping.createInFlight(rewriter);
         rewriter.replaceOp(scanOp, newInFlight);
         rewriter.create<mlir::dsa::YieldOp>(scanOp->getLoc());
      }

      return success();
   }
};
class ScanHashMapLowering : public OpConversionPattern<mlir::subop::ScanOp> {
   public:
   using OpConversionPattern<mlir::subop::ScanOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::subop::ScanOp scanOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!scanOp.state().getType().isa<mlir::subop::HashMapType>()) return failure();
      auto keyMembers = scanOp.state().getType().cast<mlir::subop::HashMapType>().getKeyMembers();
      auto valMembers = scanOp.state().getType().cast<mlir::subop::HashMapType>().getValueMembers();
      ColumnMapping mapping;
      auto state = adaptor.state();
      auto hashmapType = state.getType().cast<mlir::dsa::AggregationHashtableType>().getElementType();
      auto forOp = rewriter.create<mlir::dsa::ForOp>(scanOp->getLoc(), mlir::TypeRange{}, state, mlir::Value(), mlir::ValueRange{});
      mlir::Block* block = new mlir::Block;
      block->addArgument(hashmapType, scanOp->getLoc());
      forOp.getBodyRegion().push_back(block);
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(block);
         auto kv = rewriter.create<mlir::util::UnPackOp>(scanOp->getLoc(), forOp.getInductionVar()).getResults();
         auto unpackedKeys = rewriter.create<mlir::util::UnPackOp>(scanOp->getLoc(), kv[0]).getResults();
         auto unpackedVals = rewriter.create<mlir::util::UnPackOp>(scanOp->getLoc(), kv[1]).getResults();

         for (auto i = 0ul; i < keyMembers.getTypes().size(); i++) {
            auto name = keyMembers.getNames()[i].cast<mlir::StringAttr>().str();
            if (scanOp.mapping().contains(name)) {
               auto columnDefAttr = scanOp.mapping().get(name).cast<mlir::tuples::ColumnDefAttr>();
               mapping.define(columnDefAttr, unpackedKeys[i]);
            }
         }
         for (auto i = 0ul; i < valMembers.getTypes().size(); i++) {
            auto name = valMembers.getNames()[i].cast<mlir::StringAttr>().str();
            if (scanOp.mapping().contains(name)) {
               auto columnDefAttr = scanOp.mapping().get(name).cast<mlir::tuples::ColumnDefAttr>();
               mapping.define(columnDefAttr, unpackedVals[i]);
            }
         }

         mlir::Value newInFlight = mapping.createInFlight(rewriter);
         rewriter.replaceOp(scanOp, newInFlight);
         rewriter.create<mlir::dsa::YieldOp>(scanOp->getLoc());
      }

      return success();
   }
};
class ScanListLowering : public OpConversionPattern<mlir::subop::ScanListOp> {
   public:
   using OpConversionPattern<mlir::subop::ScanListOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::subop::ScanListOp scanOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      //if (!scanOp.list().getType().isa<mlir::subop::ListType>()) return failure();
      ColumnMapping mapping;

      auto loc = scanOp->getLoc();
      {
         auto forOp2 = rewriter.create<mlir::dsa::ForOp>(loc, mlir::TypeRange{}, adaptor.list(), mlir::Value(), mlir::ValueRange{});
         mlir::Block* block2 = new mlir::Block;
         block2->addArgument(adaptor.list().getType().cast<mlir::dsa::GenericIterableType>().getElementType(), loc);
         forOp2.getBodyRegion().push_back(block2);
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(block2);
         Value entry = forOp2.getInductionVar();
         Value valuePtr;
         auto separated = rewriter.create<mlir::util::UnPackOp>(loc, forOp2.getInductionVar()).getResults();
         entry = separated[0];
         valuePtr = separated[1];
         mapping.define(scanOp.elem(), valuePtr);
         mlir::Value newInFlight = mapping.createInFlight(rewriter);
         rewriter.replaceOp(scanOp, newInFlight);
         rewriter.create<mlir::dsa::YieldOp>(loc);
      }

      return success();
   }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////// Consuming a TupleStream//////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class UnionLowering : public OpConversionPattern<mlir::subop::UnionOp> {
   public:
   using OpConversionPattern<mlir::subop::UnionOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::subop::UnionOp unionOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      std::vector<mlir::Value> newStreams;
      for (auto x : adaptor.streams()) {
         if (auto nestedUnion = mlir::dyn_cast_or_null<mlir::subop::UnionOp>(x.getDefiningOp())) {
            newStreams.insert(newStreams.end(), nestedUnion.streams().begin(), nestedUnion.streams().end());
         } else {
            newStreams.push_back(x);
         }
      }
      rewriter.replaceOpWithNewOp<mlir::subop::UnionOp>(unionOp, newStreams);
      return mlir::success();
   }
};
template <class T>
class TupleStreamConsumerLowering : public OpConversionPattern<T> {
   using OpConversionPattern<T>::OpConversionPattern;
   virtual LogicalResult matchAndRewriteConsumer(T subOp, typename OpConversionPattern<T>::OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const = 0;
   LogicalResult matchAndRewrite(T subOp, typename OpConversionPattern<T>::OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (auto inFlightOp = mlir::dyn_cast_or_null<mlir::subop::InFlightOp>(adaptor.stream().getDefiningOp())) {
         rewriter.setInsertionPointAfter(inFlightOp);
         ColumnMapping mapping(inFlightOp);
         mlir::Value newStream;
         LogicalResult rewritten = matchAndRewriteConsumer(subOp, adaptor, rewriter, newStream, mapping);
         if (rewritten.succeeded()) {
            if (newStream) {
               rewriter.replaceOp(subOp, newStream);
            } else {
               rewriter.eraseOp(subOp);
            }
            return success();
         } else {
            return failure();
         }
      }

      if (auto unionOp = mlir::dyn_cast_or_null<mlir::subop::UnionOp>(adaptor.stream().getDefiningOp())) {
         std::vector<mlir::subop::InFlightOp> inFlightOps;
         for (auto x : unionOp.streams()) {
            if (auto inFlightOpLeft = mlir::dyn_cast_or_null<mlir::subop::InFlightOp>(x.getDefiningOp())) {
               inFlightOps.push_back(inFlightOpLeft);
            } else {
               return failure();
            }
         }
         std::vector<mlir::Value> newStreams;

         for (auto inFlightOp : inFlightOps) {
            rewriter.setInsertionPointAfter(inFlightOp);
            ColumnMapping mapping(inFlightOp);
            mlir::Value newStream;
            LogicalResult rewritten = matchAndRewriteConsumer(subOp, adaptor, rewriter, newStream, mapping);
            if (!rewritten.succeeded()) return failure();
            if (newStream) newStreams.push_back(newStream);
         }
         if (newStreams.size() == 0) {
            rewriter.eraseOp(subOp);
         } else if (newStreams.size() == 1) {
            rewriter.replaceOp(subOp, newStreams[0]);
         } else {
            rewriter.setInsertionPointAfter(subOp);
            rewriter.template replaceOpWithNewOp<mlir::subop::UnionOp>(subOp, newStreams);
         }
         return success();
      }

      return failure();
   }
};
class FilterLowering : public TupleStreamConsumerLowering<mlir::subop::FilterOp> {
   public:
   using TupleStreamConsumerLowering<mlir::subop::FilterOp>::TupleStreamConsumerLowering;
   LogicalResult matchAndRewriteConsumer(mlir::subop::FilterOp filterOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const override {
      mlir::Value cond = rewriter.create<mlir::db::AndOp>(filterOp.getLoc(), mapping.resolve(filterOp.conditions()));
      cond = rewriter.create<mlir::db::DeriveTruth>(filterOp.getLoc(), cond);
      if (filterOp.filterSemantic() == mlir::subop::FilterSemantic::none_true) {
         cond = rewriter.create<mlir::db::NotOp>(filterOp->getLoc(), cond);
      }
      rewriter.create<mlir::scf::IfOp>(
         filterOp->getLoc(), mlir::TypeRange{}, cond, [&](mlir::OpBuilder& builder1, mlir::Location) {
               newStream=mapping.createInFlight(builder1);
               builder1.create<mlir::scf::YieldOp>(filterOp->getLoc()); });

      return success();
   }
};
class RepeatLowering : public TupleStreamConsumerLowering<mlir::subop::RepeatOp> {
   public:
   using TupleStreamConsumerLowering<mlir::subop::RepeatOp>::TupleStreamConsumerLowering;
   LogicalResult matchAndRewriteConsumer(mlir::subop::RepeatOp repeatOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const override {
      auto loc = repeatOp->getLoc();
      mlir::Value zeroIdx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      mlir::Value oneIdx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
      rewriter.create<mlir::scf::ForOp>(loc, zeroIdx, mapping.resolve(repeatOp.times()), oneIdx, mlir::ValueRange{}, [&](mlir::OpBuilder& b, mlir::Location loc, mlir::Value idx, mlir::ValueRange vr) {
         newStream = mapping.createInFlight(b);
         b.create<mlir::scf::YieldOp>(loc);
      });

      return success();
   }
};
class NestedMapLowering : public TupleStreamConsumerLowering<mlir::subop::NestedMapOp> {
   public:
   using TupleStreamConsumerLowering<mlir::subop::NestedMapOp>::TupleStreamConsumerLowering;

   LogicalResult matchAndRewriteConsumer(mlir::subop::NestedMapOp nestedMapOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const override {
      mlir::Value inFlightTuple = mapping.createInFlightTuple(rewriter);
      std::vector<mlir::Value> args;
      args.push_back(inFlightTuple);
      auto parameters = mapping.resolve(nestedMapOp.parameters());
      args.insert(args.end(), parameters.begin(), parameters.end());
      auto results = inlineBlock(&nestedMapOp.region().front(), rewriter, args);
      if (!results.empty()) {
         newStream = rewriter.create<mlir::subop::CombineTupleOp>(nestedMapOp->getLoc(), results[0], inFlightTuple);
      }
      return success();
   }
};
class CombineInFlightLowering : public TupleStreamConsumerLowering<mlir::subop::CombineTupleOp> {
   public:
   using TupleStreamConsumerLowering<mlir::subop::CombineTupleOp>::TupleStreamConsumerLowering;

   LogicalResult matchAndRewriteConsumer(mlir::subop::CombineTupleOp combineInFlightOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const override {
      if (auto inFlightOpRight = mlir::dyn_cast_or_null<mlir::subop::InFlightTupleOp>(adaptor.right().getDefiningOp())) {
         mapping.merge(inFlightOpRight);
         newStream = mapping.createInFlight(rewriter);
         return success();
      }
      return failure();
   }
};
class RenameLowering : public TupleStreamConsumerLowering<mlir::subop::RenamingOp> {
   public:
   using TupleStreamConsumerLowering<mlir::subop::RenamingOp>::TupleStreamConsumerLowering;

   LogicalResult matchAndRewriteConsumer(mlir::subop::RenamingOp renamingOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const override {
      for (mlir::Attribute attr : renamingOp.columns()) {
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::tuples::ColumnDefAttr>();
         mlir::Attribute from = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>()[0];
         auto relationRefAttr = from.dyn_cast_or_null<mlir::tuples::ColumnRefAttr>();
         mapping.define(relationDefAttr, mapping.resolve(relationRefAttr));
      }
      newStream = mapping.createInFlight(rewriter);
      return success();
   }
};
class MapLowering : public TupleStreamConsumerLowering<mlir::subop::MapOp> {
   public:
   using TupleStreamConsumerLowering<mlir::subop::MapOp>::TupleStreamConsumerLowering;

   LogicalResult matchAndRewriteConsumer(mlir::subop::MapOp mapOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const override {
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
         auto returnOp = mlir::cast<mlir::tuples::ReturnOp>(terminator);
         std::vector<Value> res(returnOp.results().begin(), returnOp.results().end());
         std::vector<mlir::Operation*> toInsert;
         for (auto& x : source->getOperations()) {
            toInsert.push_back(&x);
         }
         for (auto* x : toInsert) {
            x->remove();
            rewriter.insert(x);
         }
         rewriter.eraseOp(terminator);
         cloned->erase();
         mapping.define(mapOp.computed_cols(), res);
      }
      newStream = mapping.createInFlight(rewriter);
      return success();
   }
};

class MaterializeTableLowering : public TupleStreamConsumerLowering<mlir::subop::MaterializeOp> {
   public:
   using TupleStreamConsumerLowering<mlir::subop::MaterializeOp>::TupleStreamConsumerLowering;

   LogicalResult matchAndRewriteConsumer(mlir::subop::MaterializeOp materializeOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const override {
      if (!materializeOp.state().getType().isa<mlir::subop::TableType>()) return failure();
      auto stateType = materializeOp.state().getType().cast<mlir::subop::TableType>();
      auto state = adaptor.state();
      for (size_t i = 0; i < stateType.getMembers().getTypes().size(); i++) {
         auto memberName = stateType.getMembers().getNames()[i].cast<mlir::StringAttr>().str();
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
      return mlir::success();
   }
};
class MaterializeVectorLowering : public TupleStreamConsumerLowering<mlir::subop::MaterializeOp> {
   public:
   using TupleStreamConsumerLowering<mlir::subop::MaterializeOp>::TupleStreamConsumerLowering;

   LogicalResult matchAndRewriteConsumer(mlir::subop::MaterializeOp materializeOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const override {
      if (!materializeOp.state().getType().isa<mlir::subop::VectorType>()) return failure();
      auto stateType = materializeOp.state().getType().cast<mlir::subop::VectorType>();
      std::vector<mlir::Value> values;
      for (size_t i = 0; i < stateType.getMembers().getTypes().size(); i++) {
         auto memberName = stateType.getMembers().getNames()[i].cast<mlir::StringAttr>().str();
         auto attribute = materializeOp.mapping().get(memberName).cast<mlir::tuples::ColumnRefAttr>();
         auto val = mapping.resolve(attribute);
         values.push_back(val);
      }
      mlir::Value packed = rewriter.create<mlir::util::PackOp>(materializeOp->getLoc(), values);

      rewriter.create<mlir::dsa::Append>(materializeOp->getLoc(), adaptor.state(), packed);
      return mlir::success();
   }
};
class MaterializeLazyMultiMapLowering : public TupleStreamConsumerLowering<mlir::subop::MaterializeOp> {
   public:
   using TupleStreamConsumerLowering<mlir::subop::MaterializeOp>::TupleStreamConsumerLowering;

   LogicalResult matchAndRewriteConsumer(mlir::subop::MaterializeOp materializeOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const override {
      if (!materializeOp.state().getType().isa<mlir::subop::LazyMultiMapType>()) return failure();
      auto stateType = materializeOp.state().getType().cast<mlir::subop::LazyMultiMapType>();
      std::vector<mlir::Value> keys;
      std::vector<mlir::Value> values;
      for (size_t i = 0; i < stateType.getKeyMembers().getTypes().size(); i++) {
         auto memberName = stateType.getKeyMembers().getNames()[i].cast<mlir::StringAttr>().str();
         auto attribute = materializeOp.mapping().get(memberName).cast<mlir::tuples::ColumnRefAttr>();
         auto val = mapping.resolve(attribute);
         keys.push_back(val);
      }
      for (size_t i = 0; i < stateType.getValueMembers().getTypes().size(); i++) {
         auto memberName = stateType.getValueMembers().getNames()[i].cast<mlir::StringAttr>().str();
         auto attribute = materializeOp.mapping().get(memberName).cast<mlir::tuples::ColumnRefAttr>();
         auto val = mapping.resolve(attribute);
         values.push_back(val);
      }
      mlir::Value packedValues = rewriter.create<mlir::util::PackOp>(materializeOp->getLoc(), values);
      rewriter.create<mlir::dsa::JoinHashtableInsert>(materializeOp->getLoc(), adaptor.state(), keys[0], packedValues);
      return mlir::success();
   }
};

class LookupSimpleStateLowering : public TupleStreamConsumerLowering<mlir::subop::LookupOp> {
   public:
   using TupleStreamConsumerLowering<mlir::subop::LookupOp>::TupleStreamConsumerLowering;
   LogicalResult matchAndRewriteConsumer(mlir::subop::LookupOp lookupOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const override {
      if (!lookupOp.state().getType().isa<mlir::subop::SimpleStateType>()) return failure();
      mapping.define(lookupOp.ref(), adaptor.state());
      newStream = mapping.createInFlight(rewriter);
      return mlir::success();
   }
};
class LookupLazyMultiMapLowering : public TupleStreamConsumerLowering<mlir::subop::LookupOp> {
   public:
   using TupleStreamConsumerLowering<mlir::subop::LookupOp>::TupleStreamConsumerLowering;
   LogicalResult matchAndRewriteConsumer(mlir::subop::LookupOp lookupOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const override {
      if (!lookupOp.state().getType().isa<mlir::subop::LazyMultiMapType>()) return failure();
      auto lazyMultiMapType = lookupOp.state().getType().cast<mlir::subop::LazyMultiMapType>();
      auto loc = lookupOp->getLoc();
      auto unpackTypes = [](mlir::ArrayAttr arr) {
         std::vector<Type> res;
         for (auto x : arr) { res.push_back(x.cast<mlir::TypeAttr>().getValue()); }
         return res;
      };
      mlir::TupleType keyTupleType = mlir::TupleType::get(rewriter.getContext(), {});
      mlir::TupleType valTupleType = mlir::TupleType::get(rewriter.getContext(), unpackTypes(lazyMultiMapType.getValueMembers().getTypes()));
      mlir::TupleType entryType = mlir::TupleType::get(rewriter.getContext(), {keyTupleType, valTupleType});
      mlir::TupleType entryAndValuePtrType = mlir::TupleType::get(rewriter.getContext(), TypeRange{entryType, util::RefType::get(rewriter.getContext(), valTupleType)});
      mlir::Type htIterable = mlir::dsa::GenericIterableType::get(rewriter.getContext(), entryAndValuePtrType, "join_ht_mod_iterator");

      auto matches = rewriter.create<mlir::dsa::Lookup>(loc, htIterable, adaptor.state(), mapping.resolve(lookupOp.keys())[0]);

      mapping.define(lookupOp.ref(), matches);
      newStream = mapping.createInFlight(rewriter);
      return mlir::success();
   }
};

class LookupHashMapLowering : public TupleStreamConsumerLowering<mlir::subop::LookupOrInsertOp> {
   public:
   using TupleStreamConsumerLowering<mlir::subop::LookupOrInsertOp>::TupleStreamConsumerLowering;
   LogicalResult matchAndRewriteConsumer(mlir::subop::LookupOrInsertOp lookupOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const override {
      if (!lookupOp.state().getType().isa<mlir::subop::HashMapType>()) return failure();
      mlir::dsa::AggregationHashtableType htType = adaptor.state().getType().cast<mlir::dsa::AggregationHashtableType>();
      auto packed = rewriter.create<mlir::util::PackOp>(lookupOp->getLoc(), mapping.resolve(lookupOp.keys()));

      mlir::Value hash = rewriter.create<mlir::db::Hash>(lookupOp->getLoc(), packed); //todo: external hash
      auto getRefOp = rewriter.create<mlir::dsa::HashtableGetRefOrInsert>(lookupOp->getLoc(), mlir::util::RefType::get(htType.getElementType()), adaptor.state(), hash, packed);
      mlir::Block* block = new mlir::Block;
      block->addArguments({htType.getKeyType(), htType.getKeyType()}, {lookupOp->getLoc(), lookupOp->getLoc()});
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(block);
         auto unpackedLeft = rewriter.create<mlir::util::UnPackOp>(lookupOp->getLoc(), block->getArgument(0)).getResults();
         auto unpackedRight = rewriter.create<mlir::util::UnPackOp>(lookupOp->getLoc(), block->getArgument(1)).getResults();
         std::vector<mlir::Value> arguments;
         arguments.insert(arguments.end(), unpackedLeft.begin(), unpackedLeft.end());
         arguments.insert(arguments.end(), unpackedRight.begin(), unpackedRight.end());
         auto res = inlineBlock(&lookupOp.eqFn().front(), rewriter, arguments);
         rewriter.create<mlir::dsa::YieldOp>(lookupOp->getLoc(), res);
      }
      getRefOp.equal().push_back(block);
      mlir::Block* initFnBlock = new Block;
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(initFnBlock);
         auto res = inlineBlock(&lookupOp.initFn().front(), rewriter, {});
         rewriter.create<mlir::dsa::YieldOp>(lookupOp->getLoc(), ValueRange{rewriter.create<mlir::util::PackOp>(lookupOp->getLoc(), res)});
      }
      getRefOp.initial().push_back(initFnBlock);
      mapping.define(lookupOp.ref(), rewriter.create<mlir::util::TupleElementPtrOp>(lookupOp->getLoc(), mlir::util::RefType::get(htType.getValType()), getRefOp.ref(), 1));
      newStream = mapping.createInFlight(rewriter);

      return mlir::success();
   }
};

class GatherOpLowering : public TupleStreamConsumerLowering<mlir::subop::GatherOp> {
   public:
   using TupleStreamConsumerLowering<mlir::subop::GatherOp>::TupleStreamConsumerLowering;

   LogicalResult matchAndRewriteConsumer(mlir::subop::GatherOp gatherOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const override {
      auto columns = gatherOp.ref().getColumn().type.cast<mlir::subop::EntryRefType>().getT().cast<mlir::subop::StateSupportingLookup>().getValueMembers();
      auto loaded = rewriter.create<mlir::util::LoadOp>(gatherOp->getLoc(), mapping.resolve(gatherOp.ref()));
      auto unpackedValues = rewriter.create<mlir::util::UnPackOp>(gatherOp->getLoc(), loaded).getResults();
      std::unordered_map<std::string, mlir::Value> values;
      for (auto i = 0ul; i < columns.getTypes().size(); i++) {
         auto name = columns.getNames()[i].cast<mlir::StringAttr>().str();
         values[name] = unpackedValues[i];
      }
      for (auto x : gatherOp.mapping()) {
         mapping.define(x.getValue().cast<mlir::tuples::ColumnDefAttr>(), values[x.getName().str()]);
      }
      newStream = mapping.createInFlight(rewriter);
      return success();
   }
};
class ScatterOpLowering : public TupleStreamConsumerLowering<mlir::subop::ScatterOp> {
   public:
   using TupleStreamConsumerLowering<mlir::subop::ScatterOp>::TupleStreamConsumerLowering;

   LogicalResult matchAndRewriteConsumer(mlir::subop::ScatterOp scatterOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const override {
      auto columns = scatterOp.ref().getColumn().type.cast<mlir::subop::EntryRefType>().getT().cast<mlir::subop::StateSupportingLookup>().getValueMembers();

      auto loaded = rewriter.create<mlir::util::LoadOp>(scatterOp->getLoc(), mapping.resolve(scatterOp.ref()));
      auto unpackedValues = rewriter.create<mlir::util::UnPackOp>(scatterOp->getLoc(), loaded).getResults();
      std::unordered_map<std::string, mlir::Value> values;
      for (auto i = 0ul; i < columns.getTypes().size(); i++) {
         auto name = columns.getNames()[i].cast<mlir::StringAttr>().str();
         values[name] = unpackedValues[i];
      }
      for (auto x : scatterOp.mapping()) {
         values[x.getName().str()] = mapping.resolve(x.getValue().cast<mlir::tuples::ColumnRefAttr>());
      }
      std::vector<mlir::Value> toStore;
      for (auto i = 0ul; i < columns.getTypes().size(); i++) {
         auto name = columns.getNames()[i].cast<mlir::StringAttr>().str();
         toStore.push_back(values[name]);
      }
      mlir::Value packed = rewriter.create<mlir::util::PackOp>(scatterOp->getLoc(), toStore);
      rewriter.create<mlir::util::StoreOp>(scatterOp->getLoc(), packed, mapping.resolve(scatterOp.ref()), mlir::Value());
      return success();
   }
};
class ReduceOpLowering : public TupleStreamConsumerLowering<mlir::subop::ReduceOp> {
   public:
   using TupleStreamConsumerLowering<mlir::subop::ReduceOp>::TupleStreamConsumerLowering;

   LogicalResult matchAndRewriteConsumer(mlir::subop::ReduceOp reduceOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const override {
      auto columns = reduceOp.ref().getColumn().type.cast<mlir::subop::EntryRefType>().getT().cast<mlir::subop::StateSupportingLookup>().getValueMembers();

      auto loaded = rewriter.create<mlir::util::LoadOp>(reduceOp->getLoc(), mapping.resolve(reduceOp.ref()));
      auto unpackedValues = rewriter.create<mlir::util::UnPackOp>(reduceOp->getLoc(), loaded).getResults();
      std::unordered_map<std::string, mlir::Value> values;
      for (auto i = 0ul; i < columns.getTypes().size(); i++) {
         auto name = columns.getNames()[i].cast<mlir::StringAttr>().str();
         values[name] = unpackedValues[i];
      }
      std::vector<mlir::Value> arguments;
      for (auto attr : reduceOp.columns()) {
         arguments.push_back(mapping.resolve(attr.cast<mlir::tuples::ColumnRefAttr>()));
      }
      for (auto member : reduceOp.members()) {
         arguments.push_back(values.at(member.cast<mlir::StringAttr>().str()));
      }
      auto updated = inlineBlock(&reduceOp.region().front(), rewriter, arguments);
      for (size_t i = 0; i < reduceOp.members().size(); i++) {
         values[reduceOp.members()[i].cast<mlir::StringAttr>().str()] = updated[i];
      }
      std::vector<mlir::Value> toStore;
      for (auto i = 0ul; i < columns.getTypes().size(); i++) {
         auto name = columns.getNames()[i].cast<mlir::StringAttr>().str();
         toStore.push_back(values[name]);
      }
      mlir::Value packed = rewriter.create<mlir::util::PackOp>(reduceOp->getLoc(), toStore);
      rewriter.create<mlir::util::StoreOp>(reduceOp->getLoc(), packed, mapping.resolve(reduceOp.ref()), mlir::Value());
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
   target.addLegalOp<subop::InFlightTupleOp>();
   target.addDynamicallyLegalOp<subop::UnionOp>([](mlir::subop::UnionOp unionOp) -> bool {
      bool res = llvm::all_of(unionOp.streams(), [](mlir::Value v) { return mlir::isa_and_nonnull<mlir::subop::InFlightOp>(v.getDefiningOp()); });
      return res;
   });
   auto* ctxt = &getContext();

   TypeConverter typeConverter;
   typeConverter.addConversion([](mlir::Type t) { return t; });
   auto unpackTypes = [](mlir::ArrayAttr arr) {
      std::vector<Type> res;
      for (auto x : arr) { res.push_back(x.cast<mlir::TypeAttr>().getValue()); }
      return res;
   };
   auto baseTypes = [](mlir::TypeRange arr) {
      std::vector<Type> res;
      for (auto x : arr) { res.push_back(getBaseType(x)); }
      return res;
   };
   typeConverter.addConversion([&](mlir::subop::TableRefType t) -> Type {
      auto tupleType = mlir::TupleType::get(ctxt, baseTypes(unpackTypes(t.getMembers().getTypes())));
      auto recordBatch = mlir::dsa::RecordBatchType::get(ctxt, tupleType);
      mlir::Type chunkIterable = mlir::dsa::GenericIterableType::get(ctxt, recordBatch, "table_chunk_iterator");
      return chunkIterable;
   });
   typeConverter.addConversion([&](mlir::subop::TableType t) -> Type {
      auto tupleType = mlir::TupleType::get(ctxt, unpackTypes(t.getMembers().getTypes()));
      return mlir::dsa::TableBuilderType::get(ctxt, tupleType);
   });
   typeConverter.addConversion([&](mlir::subop::VectorType t) -> Type {
      auto tupleType = mlir::TupleType::get(ctxt, unpackTypes(t.getMembers().getTypes()));
      return mlir::dsa::VectorType::get(ctxt, tupleType);
   });
   typeConverter.addConversion([&](mlir::subop::SimpleStateType t) -> Type {
      auto tupleType = mlir::TupleType::get(ctxt, unpackTypes(t.getMembers().getTypes()));
      return mlir::util::RefType::get(t.getContext(), tupleType);
   });
   typeConverter.addConversion([&](mlir::subop::HashMapType t) -> Type {
      auto keyTupleType = mlir::TupleType::get(ctxt, unpackTypes(t.getKeyMembers().getTypes()));
      auto valTupleType = mlir::TupleType::get(ctxt, unpackTypes(t.getValueMembers().getTypes()));
      return mlir::dsa::AggregationHashtableType::get(t.getContext(), keyTupleType, valTupleType);
   });
   typeConverter.addConversion([&](mlir::subop::LazyMultiMapType t) -> Type {
      auto keyTupleType = mlir::TupleType::get(ctxt, {});
      auto valTupleType = mlir::TupleType::get(ctxt, unpackTypes(t.getValueMembers().getTypes()));
      return mlir::dsa::JoinHashtableType::get(t.getContext(), keyTupleType, valTupleType);
   });
   RewritePatternSet patterns(&getContext());
   patterns.insert<FilterLowering>(typeConverter, ctxt);
   patterns.insert<RenameLowering>(typeConverter, ctxt);
   patterns.insert<MapLowering>(typeConverter, ctxt);
   patterns.insert<NestedMapLowering>(typeConverter, ctxt);
   patterns.insert<GetTableRefLowering>(typeConverter, ctxt);
   patterns.insert<ScanTableRefLowering>(typeConverter, ctxt);
   patterns.insert<ScanVectorLowering>(typeConverter, ctxt);
   patterns.insert<ScanLazyMultiMapLowering>(typeConverter, ctxt);

   patterns.insert<ScanRefsVectorLowering>(typeConverter, ctxt);
   patterns.insert<CreateTableLowering>(typeConverter, ctxt);
   patterns.insert<CreateVectorLowering>(typeConverter, ctxt);
   patterns.insert<CreateHashMapLowering>(typeConverter, ctxt);
   patterns.insert<CreateLazyMultiMapLowering>(typeConverter, ctxt);
   patterns.insert<CreateSimpleStateLowering>(typeConverter, ctxt);
   patterns.insert<ScanSimpleStateLowering>(typeConverter, ctxt);
   patterns.insert<ScanListLowering>(typeConverter, ctxt);
   patterns.insert<MaterializeTableLowering>(typeConverter, ctxt);
   patterns.insert<MaterializeVectorLowering>(typeConverter, ctxt);
   patterns.insert<MaterializeLazyMultiMapLowering>(typeConverter, ctxt);
   patterns.insert<MaintainOpLowering>(typeConverter, ctxt);

   patterns.insert<ConvertToExplicitTableLowering>(typeConverter, ctxt);
   patterns.insert<SortLowering>(typeConverter, ctxt);
   patterns.insert<CombineInFlightLowering>(typeConverter, ctxt);
   patterns.insert<LookupSimpleStateLowering>(typeConverter, ctxt);
   patterns.insert<LookupHashMapLowering>(typeConverter, ctxt);
   patterns.insert<LookupLazyMultiMapLowering>(typeConverter, ctxt);
   patterns.insert<ScanHashMapLowering>(typeConverter, ctxt);
   patterns.insert<ReduceOpLowering>(typeConverter, ctxt);
   patterns.insert<ScatterOpLowering>(typeConverter, ctxt);
   patterns.insert<GatherOpLowering>(typeConverter, ctxt);
   patterns.insert<UnionLowering>(typeConverter, ctxt);
   patterns.insert<RepeatLowering>(typeConverter, ctxt);
   if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
   }
   if (mlir::applyPatternsAndFoldGreedily(module, mlir::FrozenRewritePatternSet()).failed()) {
      signalPassFailure();
   }
}
} //namespace
std::unique_ptr<mlir::Pass>
mlir::subop::createLowerSubOpPass() {
   return std::make_unique<SubOpToControlFlowLoweringPass>();
}
void mlir::subop::createLowerSubOpPipeline(mlir::OpPassManager& pm) {
   pm.addPass(mlir::subop::createEnforceOrderPass());
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