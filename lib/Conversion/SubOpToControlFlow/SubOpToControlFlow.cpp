#include "mlir-support/parsing.h"
#include "mlir/Conversion/SubOpToControlFlow/SubOpToControlFlowPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SubOperator/SubOperatorDialect.h"
#include "mlir/Dialect/SubOperator/SubOperatorOps.h"
#include "mlir/Dialect/SubOperator/Transforms/Passes.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/Dialect/util/FunctionHelper.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "runtime-defs/Buffer.h"
#include "runtime-defs/DataSourceIteration.h"
#include "runtime-defs/GrowingBuffer.h"
#include "runtime-defs/Hashtable.h"
#include "runtime-defs/LazyJoinHashtable.h"

using namespace mlir;

namespace {
struct SubOpToControlFlowLoweringPass
   : public PassWrapper<SubOpToControlFlowLoweringPass, OperationPass<ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SubOpToControlFlowLoweringPass)
   virtual llvm::StringRef getArgument() const override { return "lower-subop-to-cf"; }

   SubOpToControlFlowLoweringPass() {}
   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<LLVM::LLVMDialect, mlir::db::DBDialect, scf::SCFDialect, mlir::cf::ControlFlowDialect, util::UtilDialect, memref::MemRefDialect, arith::ArithDialect, mlir::dsa::DSADialect, mlir::subop::SubOperatorDialect>();
   }
   void runOnOperation() final;
};

static std::vector<mlir::Value> inlineBlock(mlir::Block* b, mlir::OpBuilder& rewriter, mlir::ValueRange arguments) {
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
   for (auto val : returnOp.getResults()) {
      res.push_back(mapper.lookup(val));
   }
   return res;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////// State management ///////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GetExternalTableLowering : public OpConversionPattern<mlir::subop::GetExternalOp> {
   public:
   using OpConversionPattern<mlir::subop::GetExternalOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::subop::GetExternalOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!op.getType().isa<mlir::subop::TableType>()) return failure();
      auto parentModule = op->getParentOfType<ModuleOp>();
      mlir::func::FuncOp funcOp = parentModule.lookupSymbol<mlir::func::FuncOp>("rt_get_execution_context");
      if (!funcOp) {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(parentModule.getBody());
         funcOp = rewriter.create<mlir::func::FuncOp>(op->getLoc(), "rt_get_execution_context", rewriter.getFunctionType({}, {mlir::util::RefType::get(getContext(), rewriter.getI8Type())}), rewriter.getStringAttr("private"));
      }

      mlir::Value executionContext = rewriter.create<mlir::func::CallOp>(op->getLoc(), funcOp, mlir::ValueRange{}).getResult(0);
      mlir::Value description = rewriter.create<mlir::util::CreateConstVarLen>(op->getLoc(), mlir::util::VarLen32Type::get(rewriter.getContext()), op.getDescrAttr());

      rewriter.replaceOp(op, rt::DataSource::get(rewriter, op->getLoc())({executionContext, description})[0]);
      return mlir::success();
   }
};
static std::vector<Type> unpackTypes(mlir::ArrayAttr arr) {
   std::vector<Type> res;
   for (auto x : arr) { res.push_back(x.cast<mlir::TypeAttr>().getValue()); }
   return res;
};
static mlir::TupleType getTupleType(mlir::subop::State state) {
   return mlir::TupleType::get(state.getContext(), unpackTypes(state.getMembers().getTypes()));
}

class CreateSimpleStateLowering : public OpConversionPattern<mlir::subop::CreateOp> {
   public:
   using OpConversionPattern<mlir::subop::CreateOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::subop::CreateOp createOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!createOp.getType().isa<mlir::subop::SimpleStateType>()) return failure();
      mlir::Value ref;
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         //todo: may not be valid in the long term
         rewriter.setInsertionPointToStart(&createOp->getParentOfType<mlir::func::FuncOp>().getFunctionBody().front());
         ref = rewriter.create<mlir::util::AllocaOp>(createOp->getLoc(), typeConverter->convertType(createOp.getType()), mlir::Value());
      }
      if (!createOp.getInitFn().empty()) {
         auto x = rewriter.create<mlir::tuples::ReturnOp>(createOp->getLoc());
         rewriter.mergeBlockBefore(&createOp.getInitFn().front().front(), x);
         auto terminator = mlir::cast<mlir::tuples::ReturnOp>(x->getPrevNode());
         auto packed = rewriter.create<mlir::util::PackOp>(createOp->getLoc(), terminator.getResults());
         rewriter.eraseOp(x);
         rewriter.eraseOp(terminator);
         rewriter.create<mlir::util::StoreOp>(createOp->getLoc(), packed, ref, mlir::Value());
      }
      rewriter.replaceOp(createOp, ref);
      return mlir::success();
   }
};
static mlir::TupleType getHtKVType(mlir::subop::HashMapType t) {
   auto keyTupleType = mlir::TupleType::get(t.getContext(), unpackTypes(t.getKeyMembers().getTypes()));
   auto valTupleType = mlir::TupleType::get(t.getContext(), unpackTypes(t.getValueMembers().getTypes()));
   return mlir::TupleType::get(t.getContext(), {keyTupleType, valTupleType});
}
static mlir::TupleType getHtEntryType(mlir::subop::HashMapType t) {
   auto i8PtrType = mlir::util::RefType::get(t.getContext(), IntegerType::get(t.getContext(), 8));

   return mlir::TupleType::get(t.getContext(), {i8PtrType, mlir::IndexType::get(t.getContext()), getHtKVType(t)});
}
class CreateHashMapLowering : public OpConversionPattern<mlir::subop::CreateOp> {
   public:
   using OpConversionPattern<mlir::subop::CreateOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::subop::CreateOp createOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!createOp.getType().isa<mlir::subop::HashMapType>()) return failure();
      auto t = createOp.getType().cast<mlir::subop::HashMapType>();

      auto typeSize = rewriter.create<mlir::util::SizeOfOp>(createOp->getLoc(), rewriter.getIndexType(), getHtEntryType(t));
      Value initialCapacity = rewriter.create<arith::ConstantIndexOp>(createOp->getLoc(), 4);
      auto ptr = rt::Hashtable::create(rewriter, createOp->getLoc())({typeSize, initialCapacity})[0];
      rewriter.replaceOp(createOp, ptr);
      return mlir::success();
   }
};
class CreateLazyMultiMapLowering : public OpConversionPattern<mlir::subop::CreateOp> {
   public:
   using OpConversionPattern<mlir::subop::CreateOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::subop::CreateOp createOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!createOp.getType().isa<mlir::subop::LazyMultiMapType>()) return failure();
      auto stateType = createOp.getType().cast<mlir::subop::LazyMultiMapType>();
      auto tupleType = mlir::TupleType::get(getContext(), unpackTypes(stateType.getValueMembers().getTypes()));
      auto entryType = mlir::TupleType::get(getContext(), {rewriter.getIndexType(), tupleType});
      Value typesize = rewriter.create<mlir::util::SizeOfOp>(createOp->getLoc(), rewriter.getIndexType(), typeConverter->convertType(entryType));
      Value ptr = rt::LazyJoinHashtable::create(rewriter, createOp->getLoc())(typesize)[0];
      rewriter.replaceOp(createOp, ptr);
      return mlir::success();
   }
};
class CreateTableLowering : public OpConversionPattern<mlir::subop::CreateOp> {
   public:
   using OpConversionPattern<mlir::subop::CreateOp>::OpConversionPattern;
   std::string arrowDescrFromType(mlir::Type type) const {
      if (type.isIndex()) {
         return "int[64]";
      } else if (isIntegerType(type, 1)) {
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
      if (!createOp.getType().isa<mlir::subop::ResultTableType>()) return failure();
      auto tableType = createOp.getType().cast<mlir::subop::ResultTableType>();
      std::string descr;
      auto columnNames = createOp.getDescrAttr().cast<mlir::ArrayAttr>();
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

class CreateBufferLowering : public OpConversionPattern<mlir::subop::CreateOp> {
   public:
   using OpConversionPattern<mlir::subop::CreateOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::subop::CreateOp createOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!createOp.getType().isa<mlir::subop::BufferType>()) return failure();
      auto loc = createOp->getLoc();
      Value initialCapacity = rewriter.create<arith::ConstantIndexOp>(loc, 1024);
      auto elementType = typeConverter->convertType(getTupleType(createOp.getType()));
      auto typeSize = rewriter.create<mlir::util::SizeOfOp>(loc, rewriter.getIndexType(), elementType);
      mlir::Value vector = rt::GrowingBuffer::create(rewriter, loc)({typeSize, initialCapacity})[0];
      for (auto& b : createOp.getInitFn()) {
         auto res = inlineBlock(&b.front(), rewriter, {});
         auto packed = rewriter.create<mlir::util::PackOp>(createOp->getLoc(), res);
         mlir::Value pointer = rt::GrowingBuffer::insert(rewriter, loc)({vector})[0];
         Value castedPointer = rewriter.create<mlir::util::GenericMemrefCastOp>(loc, mlir::util::RefType::get(getContext(), elementType), pointer);
         rewriter.create<util::StoreOp>(loc, packed, castedPointer, mlir::Value());
      }
      rewriter.replaceOp(createOp, vector);
      return mlir::success();
   }
};
class MaintainOpLowering : public OpConversionPattern<mlir::subop::MaintainOp> {
   public:
   using OpConversionPattern<mlir::subop::MaintainOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::subop::MaintainOp maintainOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rt::LazyJoinHashtable::finalize(rewriter, maintainOp->getLoc())(adaptor.getState());
      rewriter.eraseOp(maintainOp);
      return mlir::success();
   }
};
class SetResultOpLowering : public OpConversionPattern<mlir::subop::SetResultOp> {
   public:
   using OpConversionPattern<mlir::subop::SetResultOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::subop::SetResultOp setResultOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<mlir::dsa::SetResultOp>(setResultOp, setResultOp.getResultId(), adaptor.getState());
      return mlir::success();
   }
};
class SortLowering : public OpConversionPattern<mlir::subop::SortOp> {
   public:
   using OpConversionPattern<mlir::subop::SortOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::subop::SortOp sortOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      static size_t id = 0;
      auto vectorType = sortOp.getToSort().getType().cast<mlir::subop::BufferType>();
      std::unordered_map<std::string, size_t> memberPositions;
      for (auto i = 0ull; i < vectorType.getMembers().getTypes().size(); i++) {
         memberPositions.insert({vectorType.getMembers().getNames()[i].cast<mlir::StringAttr>().str(), i});
      }
      auto ptrType = mlir::util::RefType::get(getContext(), IntegerType::get(getContext(), 8));

      ModuleOp parentModule = sortOp->getParentOfType<ModuleOp>();
      mlir::TupleType elementType = getTupleType(sortOp.getToSort().getType());

      mlir::func::FuncOp funcOp;
      {
         OpBuilder::InsertionGuard insertionGuard(rewriter);
         rewriter.setInsertionPointToStart(parentModule.getBody());
         funcOp = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "dsa_sort_compare" + std::to_string(id++), rewriter.getFunctionType(TypeRange({ptrType, ptrType}), TypeRange(rewriter.getI1Type())));
         auto* funcBody = new Block;
         funcBody->addArguments(TypeRange({ptrType, ptrType}), {parentModule->getLoc(), parentModule->getLoc()});
         funcOp.getBody().push_back(funcBody);
         rewriter.setInsertionPointToStart(funcBody);
         Value left = funcBody->getArgument(0);
         Value right = funcBody->getArgument(1);

         Value genericMemrefLeft = rewriter.create<util::GenericMemrefCastOp>(sortOp.getLoc(), util::RefType::get(rewriter.getContext(), elementType), left);
         Value genericMemrefRight = rewriter.create<util::GenericMemrefCastOp>(sortOp.getLoc(), util::RefType::get(rewriter.getContext(), elementType), right);
         Value tupleLeft = rewriter.create<util::LoadOp>(sortOp.getLoc(), elementType, genericMemrefLeft, Value());
         Value tupleRight = rewriter.create<util::LoadOp>(sortOp.getLoc(), elementType, genericMemrefRight, Value());
         auto unpackedLeft = rewriter.create<mlir::util::UnPackOp>(sortOp->getLoc(), tupleLeft);
         auto unpackedRight = rewriter.create<mlir::util::UnPackOp>(sortOp->getLoc(), tupleRight);
         std::vector<mlir::Value> args;
         for (auto sortByMember : sortOp.getSortBy()) {
            args.push_back(unpackedLeft.getResult(memberPositions[sortByMember.cast<mlir::StringAttr>().str()]));
         }
         for (auto sortByMember : sortOp.getSortBy()) {
            args.push_back(unpackedRight.getResult(memberPositions[sortByMember.cast<mlir::StringAttr>().str()]));
         }
         auto terminator = rewriter.create<mlir::func::ReturnOp>(sortOp.getLoc());
         Block* sortLambda = &sortOp.getRegion().front();
         auto* sortLambdaTerminator = sortLambda->getTerminator();
         rewriter.mergeBlockBefore(sortLambda, terminator, args);
         mlir::tuples::ReturnOp returnOp = mlir::cast<mlir::tuples::ReturnOp>(terminator->getPrevNode());
         Value x = returnOp.getResults()[0];
         rewriter.create<mlir::func::ReturnOp>(sortOp.getLoc(), x);
         rewriter.eraseOp(sortLambdaTerminator);
         rewriter.eraseOp(terminator);
      }

      Value functionPointer = rewriter.create<mlir::func::ConstantOp>(sortOp->getLoc(), funcOp.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(funcOp.getSymName())));
      auto genericBuffer=rt::GrowingBuffer::sort(rewriter, sortOp->getLoc())({adaptor.getToSort(), functionPointer})[0];
      rewriter.replaceOpWithNewOp<mlir::util::BufferCastOp>(sortOp,typeConverter->convertType(sortOp.getType()),genericBuffer);
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
      assert(inFlightOp.getColumns().size() == inFlightOp.getValues().size());
      for (auto i = 0ul; i < inFlightOp.getColumns().size(); i++) {
         const auto* col = &inFlightOp.getColumns()[i].cast<mlir::tuples::ColumnDefAttr>().getColumn();
         auto val = inFlightOp.getValues()[i];
         mapping.insert(std::make_pair(col, val));
      }
   }
   ColumnMapping(mlir::subop::InFlightTupleOp inFlightOp) {
      assert(inFlightOp.getColumns().size() == inFlightOp.getValues().size());
      for (auto i = 0ul; i < inFlightOp.getColumns().size(); i++) {
         const auto* col = &inFlightOp.getColumns()[i].cast<mlir::tuples::ColumnDefAttr>().getColumn();
         auto val = inFlightOp.getValues()[i];
         mapping.insert(std::make_pair(col, val));
      }
   }
   void merge(mlir::subop::InFlightOp inFlightOp) {
      assert(inFlightOp.getColumns().size() == inFlightOp.getValues().size());
      for (auto i = 0ul; i < inFlightOp.getColumns().size(); i++) {
         const auto* col = &inFlightOp.getColumns()[i].cast<mlir::tuples::ColumnDefAttr>().getColumn();
         auto val = inFlightOp.getValues()[i];
         mapping.insert(std::make_pair(col, val));
      }
   }
   void merge(mlir::subop::InFlightTupleOp inFlightOp) {
      assert(inFlightOp.getColumns().size() == inFlightOp.getValues().size());
      for (auto i = 0ul; i < inFlightOp.getColumns().size(); i++) {
         const auto* col = &inFlightOp.getColumns()[i].cast<mlir::tuples::ColumnDefAttr>().getColumn();
         auto val = inFlightOp.getValues()[i];
         mapping.insert(std::make_pair(col, val));
      }
   }
   mlir::Value resolve(mlir::tuples::ColumnRefAttr ref) {
      if (!mapping.contains(&ref.getColumn())) {
         ref.dump();
      }
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

class ScanRefsTableLowering : public OpConversionPattern<mlir::subop::ScanRefsOp> {
   public:
   using OpConversionPattern<mlir::subop::ScanRefsOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::subop::ScanRefsOp scanOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!scanOp.getState().getType().isa<mlir::subop::TableType>()) return failure();
      auto loc = scanOp->getLoc();
      auto refType = scanOp.getRef().getColumn().type.cast<mlir::subop::TableEntryRefType>();
      std::string memberMapping = "[";
      std::vector<mlir::Type> accessedColumnTypes;
      auto members = refType.getMembers();
      for (auto i = 0ul; i < members.getTypes().size(); i++) {
         auto type = members.getTypes()[i].cast<mlir::TypeAttr>().getValue();
         auto name = members.getNames()[i].cast<mlir::StringAttr>().str();
         accessedColumnTypes.push_back(type);
         if (memberMapping.length() > 1) {
            memberMapping += ",";
         }
         memberMapping += "\"" + name + "\"";
      }
      memberMapping += "]";
      mlir::Value memberMappingValue = rewriter.create<mlir::util::CreateConstVarLen>(scanOp->getLoc(), mlir::util::VarLen32Type::get(rewriter.getContext()), memberMapping);
      mlir::Value iterator = rt::DataSourceIteration::init(rewriter, scanOp->getLoc())({adaptor.getState(), memberMappingValue})[0];
      ColumnMapping mapping;

      auto baseTypes = [](mlir::TypeRange arr) {
         std::vector<Type> res;
         for (auto x : arr) { res.push_back(getBaseType(x)); }
         return res;
      };
      auto* ctxt = rewriter.getContext();
      auto tupleType = mlir::TupleType::get(ctxt, baseTypes(accessedColumnTypes));
      auto recordBatchType = mlir::dsa::RecordBatchType::get(ctxt, tupleType);

      auto whileOp = rewriter.create<mlir::scf::WhileOp>(scanOp->getLoc(), mlir::TypeRange{}, mlir::ValueRange{});
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         Block* before = new Block;
         rewriter.setInsertionPointToStart(before);
         whileOp.getBefore().push_back(before);
         mlir::Value iteratorValid = rt::DataSourceIteration::isValid(rewriter, loc)({iterator})[0];
         rewriter.create<mlir::scf::ConditionOp>(loc, iteratorValid, ValueRange{});
      }
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         Block* after = new Block;
         rewriter.setInsertionPointToStart(after);
         whileOp.getAfter().push_back(after);
         mlir::Value recordBatchPointer;
         {
            mlir::OpBuilder::InsertionGuard guard2(rewriter);
            rewriter.setInsertionPointToStart(&scanOp->getParentOfType<mlir::func::FuncOp>().getBody().front());
            recordBatchPointer = rewriter.create<mlir::util::AllocaOp>(loc, mlir::util::RefType::get(rewriter.getContext(), recordBatchType), mlir::Value());
         }
         rt::DataSourceIteration::access(rewriter, loc)({iterator, recordBatchPointer});
         mlir::Value recordBatch = rewriter.create<mlir::util::LoadOp>(loc, recordBatchPointer, mlir::Value());

         auto forOp2 = rewriter.create<mlir::dsa::ForOp>(scanOp->getLoc(), mlir::TypeRange{}, recordBatch, mlir::ValueRange{});
         {
            mlir::Block* block2 = new mlir::Block;
            block2->addArgument(recordBatchType.getElementType(), scanOp->getLoc());
            forOp2.getBodyRegion().push_back(block2);
            mlir::OpBuilder::InsertionGuard guard2(rewriter);
            rewriter.setInsertionPointToStart(block2);
            mapping.define(scanOp.getRef(), forOp2.getInductionVar());
            mlir::Value newInFlight = mapping.createInFlight(rewriter);
            rewriter.replaceOp(scanOp, newInFlight);
            rewriter.create<mlir::dsa::YieldOp>(scanOp->getLoc());
         }

         rt::DataSourceIteration::next(rewriter, loc)({iterator});
         rewriter.create<mlir::scf::YieldOp>(loc);
      }
      rt::DataSourceIteration::end(rewriter, loc)({iterator});

      return success();
   }
};

class ScanRefsSimpleStateLowering : public OpConversionPattern<mlir::subop::ScanRefsOp> {
   public:
   using OpConversionPattern<mlir::subop::ScanRefsOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::subop::ScanRefsOp scanOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!scanOp.getState().getType().isa<mlir::subop::SimpleStateType>()) return failure();
      ColumnMapping mapping;
      mapping.define(scanOp.getRef(), adaptor.getState());
      mlir::Value newInFlight = mapping.createInFlight(rewriter);
      rewriter.replaceOp(scanOp, newInFlight);
      return success();
   }
};
void implementBufferIteration(mlir::Value bufferIterator, mlir::Type entryType, mlir::Location loc, mlir::ConversionPatternRewriter& rewriter, std::function<void(mlir::ConversionPatternRewriter& rewriter, mlir::Value)> fn) {
   auto whileOp = rewriter.create<mlir::scf::WhileOp>(loc, mlir::TypeRange{}, mlir::ValueRange{});
   Block* before = new Block;
   Block* after = new Block;
   whileOp.getBefore().push_back(before);
   whileOp.getAfter().push_back(after);

   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(before);
      mlir::Value cond = rt::BufferIterator::isIteratorValid(rewriter, loc)({bufferIterator})[0];
      rewriter.create<mlir::scf::ConditionOp>(loc, cond, ValueRange{});
   }
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(after);
      auto buffer = rt::BufferIterator::iteratorGetCurrentBuffer(rewriter, loc)({bufferIterator})[0];

      auto castedBuffer = rewriter.create<mlir::util::BufferCastOp>(loc, mlir::util::BufferType::get(rewriter.getContext(), entryType), buffer);

      auto forOp = rewriter.create<mlir::dsa::ForOp>(loc, mlir::TypeRange{}, castedBuffer, mlir::ValueRange{});
      mlir::Block* block = new mlir::Block;
      block->addArgument(mlir::util::RefType::get(rewriter.getContext(), entryType), loc);
      forOp.getBodyRegion().push_back(block);
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(block);
         fn(rewriter, forOp.getInductionVar());
         rewriter.create<mlir::dsa::YieldOp>(loc);
      }
      rt::BufferIterator::iteratorNext(rewriter, loc)({bufferIterator});
      rewriter.create<mlir::scf::YieldOp>(loc);
   }
   rt::BufferIterator::destroy(rewriter, loc)({bufferIterator});
}
class ScanLazyMultiMapLowering : public OpConversionPattern<mlir::subop::ScanOp> {
   public:
   using OpConversionPattern<mlir::subop::ScanOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::subop::ScanOp scanOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!scanOp.getState().getType().isa<mlir::subop::LazyMultiMapType>()) return failure();
      auto columns = scanOp.getState().getType().cast<mlir::subop::LazyMultiMapType>().getValueMembers();
      ColumnMapping mapping;
      auto stateType = scanOp.getState().getType().cast<mlir::subop::LazyMultiMapType>();
      auto tupleType = mlir::TupleType::get(getContext(), unpackTypes(stateType.getValueMembers().getTypes()));
      auto entryType = mlir::TupleType::get(getContext(), {rewriter.getIndexType(), tupleType});
      auto iterator = rt::LazyJoinHashtable::createIterator(rewriter, scanOp->getLoc())(adaptor.getState())[0];
      implementBufferIteration(iterator, entryType, scanOp->getLoc(), rewriter, [&](ConversionPatternRewriter& rewriter, mlir::Value ptr) {
         auto loaded = rewriter.create<mlir::util::LoadOp>(scanOp->getLoc(), ptr);
         auto unpackedEntry = rewriter.create<mlir::util::UnPackOp>(scanOp->getLoc(), loaded);
         auto unpackedValues = rewriter.create<mlir::util::UnPackOp>(scanOp->getLoc(), unpackedEntry.getResult(1)).getResults();

         for (auto i = 0ul; i < columns.getTypes().size(); i++) {
            auto name = columns.getNames()[i].cast<mlir::StringAttr>().str();
            if (scanOp.getMapping().contains(name)) {
               auto columnDefAttr = scanOp.getMapping().get(name).cast<mlir::tuples::ColumnDefAttr>();
               mapping.define(columnDefAttr, unpackedValues[i]);
            }
         }

         mlir::Value newInFlight = mapping.createInFlight(rewriter);
         rewriter.replaceOp(scanOp, newInFlight);
      });
      return success();
   }
};
class ScanRefsVectorLowering : public OpConversionPattern<mlir::subop::ScanRefsOp> {
   public:
   using OpConversionPattern<mlir::subop::ScanRefsOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::subop::ScanRefsOp scanOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!scanOp.getState().getType().isa<mlir::subop::BufferType>()) return failure();
      ColumnMapping mapping;
      auto elementType =getTupleType(scanOp.getState().getType());

      auto iterator=rt::GrowingBuffer::createIterator(rewriter,scanOp->getLoc())(adaptor.getState())[0];
      implementBufferIteration(iterator,elementType,scanOp->getLoc(),rewriter,[&](ConversionPatternRewriter& rewriter,mlir::Value ptr){
         mapping.define(scanOp.getRef(), ptr);
         mlir::Value newInFlight = mapping.createInFlight(rewriter);
         rewriter.replaceOp(scanOp, newInFlight);
      });
      return success();
   }
};

class ScanRefsSortedViewLowering : public OpConversionPattern<mlir::subop::ScanRefsOp> {
   public:
   using OpConversionPattern<mlir::subop::ScanRefsOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::subop::ScanRefsOp scanOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!scanOp.getState().getType().isa<mlir::subop::SortedViewType>()) return failure();
      ColumnMapping mapping;
      auto elementType = mlir::util::RefType::get(getContext(), getTupleType(scanOp.getState().getType()));
      auto forOp = rewriter.create<mlir::dsa::ForOp>(scanOp->getLoc(), mlir::TypeRange{}, adaptor.getState(), mlir::ValueRange{});
      mlir::Block* block = new mlir::Block;
      block->addArgument(elementType, scanOp->getLoc());
      forOp.getBodyRegion().push_back(block);
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(block);
         mapping.define(scanOp.getRef(), forOp.getInductionVar());

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
      if (!scanOp.getState().getType().isa<mlir::subop::HashMapType>()) return failure();
      auto hashMapType = scanOp.getState().getType().cast<mlir::subop::HashMapType>();
      auto keyMembers = hashMapType.getKeyMembers();
      auto valMembers = hashMapType.getValueMembers();
      ColumnMapping mapping;
      auto loc = scanOp->getLoc();
      auto it = rt::Hashtable::createIterator(rewriter, loc)({adaptor.getState()})[0];
      auto kvPtrType = mlir::util::RefType::get(getContext(), getHtKVType(hashMapType));
      implementBufferIteration(it, getHtEntryType(hashMapType), scanOp->getLoc(), rewriter, [&](ConversionPatternRewriter& rewriter, mlir::Value ptr) {
         auto kvPtr = rewriter.create<mlir::util::TupleElementPtrOp>(loc, kvPtrType, ptr, 2);
         auto kvLoaded = rewriter.create<mlir::util::LoadOp>(loc, kvPtr, mlir::Value());
         auto kv = rewriter.create<mlir::util::UnPackOp>(scanOp->getLoc(), kvLoaded).getResults();
         auto unpackedKeys = rewriter.create<mlir::util::UnPackOp>(scanOp->getLoc(), kv[0]).getResults();
         auto unpackedVals = rewriter.create<mlir::util::UnPackOp>(scanOp->getLoc(), kv[1]).getResults();

         for (auto i = 0ul; i < keyMembers.getTypes().size(); i++) {
            auto name = keyMembers.getNames()[i].cast<mlir::StringAttr>().str();
            if (scanOp.getMapping().contains(name)) {
               auto columnDefAttr = scanOp.getMapping().get(name).cast<mlir::tuples::ColumnDefAttr>();
               mapping.define(columnDefAttr, unpackedKeys[i]);
            }
         }
         for (auto i = 0ul; i < valMembers.getTypes().size(); i++) {
            auto name = valMembers.getNames()[i].cast<mlir::StringAttr>().str();
            if (scanOp.getMapping().contains(name)) {
               auto columnDefAttr = scanOp.getMapping().get(name).cast<mlir::tuples::ColumnDefAttr>();
               mapping.define(columnDefAttr, unpackedVals[i]);
            }
         }

         mlir::Value newInFlight = mapping.createInFlight(rewriter);
         rewriter.replaceOp(scanOp, newInFlight);
      });
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
      auto unpacked = rewriter.create<mlir::util::UnPackOp>(loc, adaptor.getList());
      auto ptr = unpacked.getResult(0);
      auto hash = unpacked.getResult(1);
      auto iteratorType = ptr.getType();
      auto referenceType = scanOp.getList().getType().cast<mlir::subop::ListType>().getT();

      auto whileOp = rewriter.create<mlir::scf::WhileOp>(loc, iteratorType, ptr);
      Block* before = new Block;
      Block* after = new Block;
      whileOp.getBefore().push_back(before);
      whileOp.getAfter().push_back(after);

      mlir::Value beforePtr = before->addArgument(iteratorType, loc);
      mlir::Value afterPtr = after->addArgument(iteratorType, loc);
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(before);
         mlir::Value cond = rewriter.create<mlir::util::IsRefValidOp>(loc, rewriter.getI1Type(), beforePtr);
         rewriter.create<mlir::scf::ConditionOp>(loc, cond, beforePtr);
      }
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(after);
         auto tupleType = mlir::TupleType::get(getContext(), unpackTypes(referenceType.getMembers().getTypes()));
         auto i8PtrType = mlir::util::RefType::get(getContext(), rewriter.getI8Type());
         Value castedPtr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(), mlir::TupleType::get(getContext(), {i8PtrType, tupleType})), afterPtr);
         Value valuePtr = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(getContext(), tupleType), castedPtr, 1);

         mapping.define(scanOp.getElem(), valuePtr);
         mlir::Value newInFlight = mapping.createInFlight(rewriter);
         rewriter.replaceOp(scanOp, newInFlight);
         Value nextPtr = rewriter.create<util::TupleElementPtrOp>(loc, mlir::util::RefType::get(getContext(), i8PtrType), castedPtr, 0);
         mlir::Value next = rewriter.create<mlir::util::LoadOp>(loc, nextPtr, mlir::Value());
         next = rewriter.create<mlir::util::FilterTaggedPtr>(loc, next.getType(), next, hash);
         rewriter.create<mlir::scf::YieldOp>(loc, next);
      }

      return success();
   }
};
class GenerateEmitLowering : public OpConversionPattern<mlir::subop::GenerateEmitOp> {
   public:
   using OpConversionPattern<mlir::subop::GenerateEmitOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::subop::GenerateEmitOp generateOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.eraseOp(generateOp);
      return success();
   }
};

class GenerateLowering : public OpConversionPattern<mlir::subop::GenerateOp> {
   public:
   using OpConversionPattern<mlir::subop::GenerateOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::subop::GenerateOp generateOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      ColumnMapping mapping;
      std::vector<mlir::subop::GenerateEmitOp> emitOps;
      generateOp.getRegion().walk([&](mlir::subop::GenerateEmitOp emitOp) {
         emitOps.push_back(emitOp);
      });
      std::vector<mlir::Value> streams;
      for (auto emitOp : emitOps) {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointAfter(emitOp);
         ColumnMapping mapping;
         mapping.define(generateOp.getGeneratedColumns(), emitOp.getValues());
         mlir::Value newInFlight = mapping.createInFlight(rewriter);
         streams.push_back(newInFlight);
      }
      {
         auto* b = &generateOp.getRegion().front();
         auto* terminator = b->getTerminator();
         mlir::BlockAndValueMapping mapper;
         for (auto& x : b->getOperations()) {
            if (&x != terminator) {
               rewriter.clone(x, mapper);
            }
         }
         std::vector<mlir::Value> res;
         for (auto& s : streams) {
            s = mapper.lookup(s);
         }
      }
      rewriter.replaceOpWithNewOp<mlir::subop::UnionOp>(generateOp, streams);
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
      for (auto x : adaptor.getStreams()) {
         if (auto nestedUnion = mlir::dyn_cast_or_null<mlir::subop::UnionOp>(x.getDefiningOp())) {
            newStreams.insert(newStreams.end(), nestedUnion.getStreams().begin(), nestedUnion.getStreams().end());
         } else {
            newStreams.push_back(x);
         }
      }
      rewriter.replaceOpWithNewOp<mlir::subop::UnionOp>(unionOp, newStreams);
      return mlir::success();
   }
};
template <class T, unsigned benefit = 1>
class TupleStreamConsumerLowering : public OpConversionPattern<T> {
   TupleStreamConsumerLowering(MLIRContext* context, PatternBenefit b = benefit) : OpConversionPattern<T>(context, b) {}
   virtual LogicalResult matchAndRewriteConsumer(T subOp, typename OpConversionPattern<T>::OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const = 0;
   LogicalResult matchAndRewrite(T subOp, typename OpConversionPattern<T>::OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (auto inFlightOp = mlir::dyn_cast_or_null<mlir::subop::InFlightOp>(adaptor.getStream().getDefiningOp())) {
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

      if (auto unionOp = mlir::dyn_cast_or_null<mlir::subop::UnionOp>(adaptor.getStream().getDefiningOp())) {
         std::vector<mlir::subop::InFlightOp> inFlightOps;
         for (auto x : unionOp.getStreams()) {
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

   public:
   TupleStreamConsumerLowering(TypeConverter& typeConverter, MLIRContext* context, PatternBenefit b = benefit) : OpConversionPattern<T>(typeConverter, context, b) {}
};
class FilterLowering : public TupleStreamConsumerLowering<mlir::subop::FilterOp> {
   public:
   using TupleStreamConsumerLowering<mlir::subop::FilterOp>::TupleStreamConsumerLowering;
   LogicalResult matchAndRewriteConsumer(mlir::subop::FilterOp filterOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const override {
      mlir::Value cond = rewriter.create<mlir::db::AndOp>(filterOp.getLoc(), mapping.resolve(filterOp.getConditions()));
      cond = rewriter.create<mlir::db::DeriveTruth>(filterOp.getLoc(), cond);
      if (filterOp.getFilterSemantic() == mlir::subop::FilterSemantic::none_true) {
         cond = rewriter.create<mlir::db::NotOp>(filterOp->getLoc(), cond);
      }
      rewriter.create<mlir::scf::IfOp>(
         filterOp->getLoc(), mlir::TypeRange{}, cond, [&](mlir::OpBuilder& builder1, mlir::Location) {
               newStream=mapping.createInFlight(builder1);
               builder1.create<mlir::scf::YieldOp>(filterOp->getLoc()); });

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
      auto parameters = mapping.resolve(nestedMapOp.getParameters());
      args.insert(args.end(), parameters.begin(), parameters.end());
      for (size_t i = 0; i < args.size(); i++) {
         if (args[i].getType() != nestedMapOp.getRegion().front().getArgument(i).getType()) {
            args[i] = rewriter.create<mlir::UnrealizedConversionCastOp>(nestedMapOp->getLoc(), nestedMapOp.getRegion().front().getArgument(i).getType(), args[i]).getResult(0);
         }
      }
      auto results = inlineBlock(&nestedMapOp.getRegion().front(), rewriter, args);
      if (!results.empty()) {
         newStream = rewriter.create<mlir::subop::CombineTupleOp>(nestedMapOp->getLoc(), rewriter.getRemappedValue(results[0]), inFlightTuple);
      }
      return success();
   }
};
class CombineInFlightLowering : public TupleStreamConsumerLowering<mlir::subop::CombineTupleOp> {
   public:
   using TupleStreamConsumerLowering<mlir::subop::CombineTupleOp>::TupleStreamConsumerLowering;

   LogicalResult matchAndRewriteConsumer(mlir::subop::CombineTupleOp combineInFlightOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const override {
      if (auto inFlightOpRight = mlir::dyn_cast_or_null<mlir::subop::InFlightTupleOp>(adaptor.getRight().getDefiningOp())) {
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
      for (mlir::Attribute attr : renamingOp.getColumns()) {
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

         mlir::Block* source = &cloned.getFn().front();
         auto* terminator = source->getTerminator();

         source->walk([&](mlir::tuples::GetColumnOp getColumnOp) {
            getColumnOp.replaceAllUsesWith(mapping.resolve(getColumnOp.getAttr()));
            toErase.push_back(getColumnOp.getOperation());
         });
         for (auto* op : toErase) {
            op->dropAllUses();
            op->erase();
         }
         auto returnOp = mlir::cast<mlir::tuples::ReturnOp>(terminator);
         std::vector<Value> res(returnOp.getResults().begin(), returnOp.getResults().end());
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
         mapping.define(mapOp.getComputedCols(), res);
      }
      newStream = mapping.createInFlight(rewriter);
      return success();
   }
};

class MaterializeTableLowering : public TupleStreamConsumerLowering<mlir::subop::MaterializeOp> {
   public:
   using TupleStreamConsumerLowering<mlir::subop::MaterializeOp>::TupleStreamConsumerLowering;

   LogicalResult matchAndRewriteConsumer(mlir::subop::MaterializeOp materializeOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const override {
      if (!materializeOp.getState().getType().isa<mlir::subop::ResultTableType>()) return failure();
      auto stateType = materializeOp.getState().getType().cast<mlir::subop::ResultTableType>();
      auto state = adaptor.getState();
      for (size_t i = 0; i < stateType.getMembers().getTypes().size(); i++) {
         auto memberName = stateType.getMembers().getNames()[i].cast<mlir::StringAttr>().str();
         auto attribute = materializeOp.getMapping().get(memberName).cast<mlir::tuples::ColumnRefAttr>();
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
      if (!materializeOp.getState().getType().isa<mlir::subop::BufferType>()) return failure();
      auto stateType = materializeOp.getState().getType().cast<mlir::subop::BufferType>();
      std::vector<mlir::Value> values;
      for (size_t i = 0; i < stateType.getMembers().getTypes().size(); i++) {
         auto memberName = stateType.getMembers().getNames()[i].cast<mlir::StringAttr>().str();
         auto attribute = materializeOp.getMapping().get(memberName).cast<mlir::tuples::ColumnRefAttr>();
         auto val = mapping.resolve(attribute);
         values.push_back(val);
      }
      mlir::Value packed = rewriter.create<mlir::util::PackOp>(materializeOp->getLoc(), values);
      mlir::Value pointer = rt::GrowingBuffer::insert(rewriter, materializeOp->getLoc())({adaptor.getState()})[0];
      Value castedPointer = rewriter.create<mlir::util::GenericMemrefCastOp>(materializeOp->getLoc(), mlir::util::RefType::get(getContext(), packed.getType()), pointer);
      rewriter.create<util::StoreOp>(materializeOp->getLoc(), packed, castedPointer, mlir::Value());
      return mlir::success();
   }
};
class MaterializeLazyMultiMapLowering : public TupleStreamConsumerLowering<mlir::subop::MaterializeOp> {
   public:
   using TupleStreamConsumerLowering<mlir::subop::MaterializeOp>::TupleStreamConsumerLowering;

   LogicalResult matchAndRewriteConsumer(mlir::subop::MaterializeOp materializeOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const override {
      if (!materializeOp.getState().getType().isa<mlir::subop::LazyMultiMapType>()) return failure();
      auto stateType = materializeOp.getState().getType().cast<mlir::subop::LazyMultiMapType>();
      auto loc = materializeOp->getLoc();
      std::vector<mlir::Value> keys;
      std::vector<mlir::Value> values;
      for (size_t i = 0; i < stateType.getKeyMembers().getTypes().size(); i++) {
         auto memberName = stateType.getKeyMembers().getNames()[i].cast<mlir::StringAttr>().str();
         auto attribute = materializeOp.getMapping().get(memberName).cast<mlir::tuples::ColumnRefAttr>();
         auto val = mapping.resolve(attribute);
         keys.push_back(val);
      }
      for (size_t i = 0; i < stateType.getValueMembers().getTypes().size(); i++) {
         auto memberName = stateType.getValueMembers().getNames()[i].cast<mlir::StringAttr>().str();
         auto attribute = materializeOp.getMapping().get(memberName).cast<mlir::tuples::ColumnRefAttr>();
         auto val = mapping.resolve(attribute);
         values.push_back(val);
      }
      mlir::Value hashed = keys[0];
      mlir::Value packedValues = rewriter.create<mlir::util::PackOp>(materializeOp->getLoc(), values);
      mlir::Value pointer = rt::LazyJoinHashtable::insert(rewriter, loc)({adaptor.getState(), hashed})[0];
      Value castedPointer = rewriter.create<mlir::util::GenericMemrefCastOp>(loc, mlir::util::RefType::get(getContext(), packedValues.getType()), pointer);
      rewriter.create<util::StoreOp>(loc, packedValues, castedPointer, mlir::Value());
      return mlir::success();
   }
};

class LookupSimpleStateLowering : public TupleStreamConsumerLowering<mlir::subop::LookupOp> {
   public:
   using TupleStreamConsumerLowering<mlir::subop::LookupOp>::TupleStreamConsumerLowering;
   LogicalResult matchAndRewriteConsumer(mlir::subop::LookupOp lookupOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const override {
      if (!lookupOp.getState().getType().isa<mlir::subop::SimpleStateType>()) return failure();
      mapping.define(lookupOp.getRef(), adaptor.getState());
      newStream = mapping.createInFlight(rewriter);
      return mlir::success();
   }
};
class LookupLazyMultiMapLowering : public TupleStreamConsumerLowering<mlir::subop::LookupOp> {
   public:
   using TupleStreamConsumerLowering<mlir::subop::LookupOp>::TupleStreamConsumerLowering;
   LogicalResult matchAndRewriteConsumer(mlir::subop::LookupOp lookupOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const override {
      if (!lookupOp.getState().getType().isa<mlir::subop::LazyMultiMapType>()) return failure();
      auto loc = lookupOp->getLoc();
      mlir::Value hash = mapping.resolve(lookupOp.getKeys())[0];
      auto* context = getContext();
      auto indexType = rewriter.getIndexType();
      auto htType = util::RefType::get(context, mlir::util::RefType::get(context, rewriter.getI8Type()));

      Value castedPointer = rewriter.create<mlir::util::GenericMemrefCastOp>(loc, util::RefType::get(context, TupleType::get(context, {htType, indexType, indexType, indexType})), adaptor.getState());

      auto loaded = rewriter.create<util::LoadOp>(loc, castedPointer.getType().cast<mlir::util::RefType>().getElementType(), castedPointer, Value());
      auto unpacked = rewriter.create<mlir::util::UnPackOp>(loc, loaded);
      Value ht = unpacked.getResult(0);
      Value htMask = unpacked.getResult(1);
      Value buckedPos = rewriter.create<arith::AndIOp>(loc, htMask, hash);
      Value ptr = rewriter.create<util::LoadOp>(loc, mlir::util::RefType::get(getContext(), rewriter.getI8Type()), ht, buckedPos);
      //optimization
      ptr = rewriter.create<mlir::util::FilterTaggedPtr>(loc, ptr.getType(), ptr, hash);
      Value matches = rewriter.create<mlir::util::PackOp>(loc, ValueRange{ptr, hash});

      mapping.define(lookupOp.getRef(), matches);
      newStream = mapping.createInFlight(rewriter);
      return mlir::success();
   }
};

class LookupHashMapLowering : public TupleStreamConsumerLowering<mlir::subop::LookupOrInsertOp> {
   public:
   using TupleStreamConsumerLowering<mlir::subop::LookupOrInsertOp>::TupleStreamConsumerLowering;
   LogicalResult matchAndRewriteConsumer(mlir::subop::LookupOrInsertOp lookupOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const override {
      if (!lookupOp.getState().getType().isa<mlir::subop::HashMapType>()) return failure();
      mlir::subop::HashMapType htStateType = lookupOp.getState().getType().cast<mlir::subop::HashMapType>();
      auto packed = rewriter.create<mlir::util::PackOp>(lookupOp->getLoc(), mapping.resolve(lookupOp.getKeys()));

      mlir::Value hash = rewriter.create<mlir::db::Hash>(lookupOp->getLoc(), packed); //todo: external hash
      mlir::Value hashTable = adaptor.getState();

      auto equalFnBuilder = [&lookupOp](mlir::OpBuilder& rewriter, Value left, Value right) -> Value {
         auto unpackedLeft = rewriter.create<mlir::util::UnPackOp>(lookupOp->getLoc(), left).getResults();
         auto unpackedRight = rewriter.create<mlir::util::UnPackOp>(lookupOp->getLoc(), right).getResults();
         std::vector<mlir::Value> arguments;
         arguments.insert(arguments.end(), unpackedLeft.begin(), unpackedLeft.end());
         arguments.insert(arguments.end(), unpackedRight.begin(), unpackedRight.end());
         auto res = inlineBlock(&lookupOp.getEqFn().front(), rewriter, arguments);
         return res[0];
      };
      auto initValBuilder = [&lookupOp](mlir::OpBuilder& rewriter) -> mlir::Value {
         auto res = inlineBlock(&lookupOp.getInitFn().front(), rewriter, {});
         return rewriter.create<mlir::util::PackOp>(lookupOp->getLoc(), res);
      };
      auto loc = lookupOp->getLoc();
      Value hashed = hash;

      auto keyType = packed.getType();
      auto* context = rewriter.getContext();
      auto entryType = getHtEntryType(htStateType);
      auto i8PtrType = mlir::util::RefType::get(context, IntegerType::get(context, 8));
      auto i8PtrPtrType = mlir::util::RefType::get(context, i8PtrType);

      auto idxType = rewriter.getIndexType();
      auto idxPtrType = util::RefType::get(rewriter.getContext(), idxType);
      auto kvType = getHtKVType(htStateType);
      auto kvPtrType = mlir::util::RefType::get(context, kvType);
      auto keyPtrType = mlir::util::RefType::get(context, keyType);
      auto entryPtrType = mlir::util::RefType::get(context, entryType);
      auto htType = mlir::util::RefType::get(context, entryPtrType);

      Value castedState = rewriter.create<util::GenericMemrefCastOp>(loc, mlir::util::RefType::get(getContext(), mlir::TupleType::get(getContext(), {htType, rewriter.getIndexType()})), adaptor.getState());
      Value htAddress = rewriter.create<util::TupleElementPtrOp>(loc, mlir::util::RefType::get(rewriter.getContext(), htType), castedState, 0);
      Value htMaskAddress = rewriter.create<util::TupleElementPtrOp>(loc, mlir::util::RefType::get(rewriter.getContext(), rewriter.getIndexType()), castedState, 1);
      Value ht = rewriter.create<util::LoadOp>(loc, htType, htAddress);
      Value htMask = rewriter.create<util::LoadOp>(loc, rewriter.getIndexType(), htMaskAddress);

      Value trueValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

      //position = hash & hashTableMask
      Value position = rewriter.create<arith::AndIOp>(loc, htMask, hashed);
      // ptr = &hashtable[position]
      Type bucketPtrType = util::RefType::get(context, entryType);
      Type doneType = rewriter.getI1Type();
      Value ptr = rewriter.create<util::LoadOp>(loc, bucketPtrType, ht, position);

      auto whileOp = rewriter.create<scf::WhileOp>(loc, bucketPtrType, ValueRange({ptr}));
      Block* before = rewriter.createBlock(&whileOp.getBefore(), {}, bucketPtrType, {loc});
      Block* after = rewriter.createBlock(&whileOp.getAfter(), {}, bucketPtrType, {loc});

      // The conditional block of the while loop.
      {
         rewriter.setInsertionPointToStart(&whileOp.getBefore().front());
         Value ptr = before->getArgument(0);

         Value currEntryPtr = ptr;
         //    if (*ptr != nullptr){
         Value cmp = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), currEntryPtr);
         auto ifOp = rewriter.create<scf::IfOp>(
            loc, TypeRange({doneType, bucketPtrType}), cmp,
            [&](OpBuilder& b, Location loc) {

               Value hashAddress=rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, currEntryPtr, 1);
               Value entryHash = rewriter.create<util::LoadOp>(loc, idxType, hashAddress);
               Value hashMatches = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, entryHash,hashed);
               auto ifOpH = b.create<scf::IfOp>(
                  loc, TypeRange({doneType,bucketPtrType}), hashMatches, [&](OpBuilder& b, Location loc) {
               Value kvAddress = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, currEntryPtr, 2);
               Value entryKeyAddress = rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, kvAddress, 0);
               Value entryKey = rewriter.create<util::LoadOp>(loc, keyType, entryKeyAddress);

               Value keyMatches = equalFnBuilder(b, entryKey, packed);
               auto ifOp2 = b.create<scf::IfOp>(
                  loc, TypeRange({doneType, bucketPtrType}), keyMatches, [&](OpBuilder& b, Location loc) {


                     b.create<scf::YieldOp>(loc, ValueRange{falseValue, ptr}); }, [&](OpBuilder& b, Location loc) {
                           Value nextPtrAddr=rewriter.create<util::TupleElementPtrOp>(loc, i8PtrPtrType, currEntryPtr, 0);
                              //          ptr = &entry.next
                              Value newEntryPtr=b.create<util::LoadOp>(loc,nextPtrAddr,mlir::Value());
                              Value newPtr=b.create<util::GenericMemrefCastOp>(loc, bucketPtrType,newEntryPtr);
                              //          yield ptr,done=false
                              b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr }); });
               b.create<scf::YieldOp>(loc, ifOp2.getResults());
                     }, [&](OpBuilder& b, Location loc) {
                     Value nextPtrAddr=rewriter.create<util::TupleElementPtrOp>(loc, i8PtrPtrType, currEntryPtr, 0);
                     //          ptr = &entry.next
                     Value newEntryPtr=b.create<util::LoadOp>(loc,nextPtrAddr,mlir::Value());
                     Value newPtr=b.create<util::GenericMemrefCastOp>(loc, bucketPtrType,newEntryPtr);
                     //          yield ptr,done=false
                     b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr });});
               b.create<scf::YieldOp>(loc, ifOpH.getResults()); }, [&](OpBuilder& b, Location loc) {
               Value initialVal = initValBuilder(b);
               Value newKVPair = b.create<util::PackOp>(loc,ValueRange({packed, initialVal}));
               //       %newEntry = ...
               Value newValueLocPtr=rt::Hashtable::insert(b,loc)({hashTable,hashed})[0];
               Value castedNewValueLocPtr= rewriter.create<util::GenericMemrefCastOp>(loc, bucketPtrType, newValueLocPtr);
               Value kvAddress = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, castedNewValueLocPtr, 2);

               //       append(vec,newEntry)
               b.create<util::StoreOp>(loc, newKVPair, kvAddress,Value());

               b.create<scf::YieldOp>(loc, ValueRange{falseValue, castedNewValueLocPtr}); });
         //       if(compare(entry.key,key)){

         Value done = ifOp.getResult(0);
         Value newPtr = ifOp.getResult(1);
         rewriter.create<scf::ConditionOp>(loc, done, ValueRange({newPtr}));
      }

      // The body of the while loop: shift right until reaching a value of 0.
      {
         rewriter.setInsertionPointToStart(&whileOp.getAfter().front());
         rewriter.create<scf::YieldOp>(loc, after->getArguments());
      }

      rewriter.setInsertionPointAfter(whileOp);

      Value currEntryPtr = whileOp.getResult(0);
      Value kvAddress = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, currEntryPtr, 2);
      mapping.define(lookupOp.getRef(), rewriter.create<mlir::util::TupleElementPtrOp>(lookupOp->getLoc(), mlir::util::RefType::get(getContext(), kvType.getType(1)), kvAddress, 1));
      newStream = mapping.createInFlight(rewriter);
      return mlir::success();
   }
};

class DefaultGatherOpLowering : public TupleStreamConsumerLowering<mlir::subop::GatherOp> {
   public:
   using TupleStreamConsumerLowering<mlir::subop::GatherOp>::TupleStreamConsumerLowering;

   LogicalResult matchAndRewriteConsumer(mlir::subop::GatherOp gatherOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const override {
      auto columns = gatherOp.getRef().getColumn().type.cast<mlir::subop::StateEntryReference>().getMembers();
      auto loaded = rewriter.create<mlir::util::LoadOp>(gatherOp->getLoc(), mapping.resolve(gatherOp.getRef()));
      auto unpackedValues = rewriter.create<mlir::util::UnPackOp>(gatherOp->getLoc(), loaded).getResults();
      std::unordered_map<std::string, mlir::Value> values;
      for (auto i = 0ul; i < columns.getTypes().size(); i++) {
         auto name = columns.getNames()[i].cast<mlir::StringAttr>().str();
         values[name] = unpackedValues[i];
      }
      for (auto x : gatherOp.getMapping()) {
         mapping.define(x.getValue().cast<mlir::tuples::ColumnDefAttr>(), values[x.getName().str()]);
      }
      newStream = mapping.createInFlight(rewriter);
      return success();
   }
};
class TableRefGatherOpLowering : public TupleStreamConsumerLowering<mlir::subop::GatherOp, 2> {
   public:
   using TupleStreamConsumerLowering<mlir::subop::GatherOp, 2>::TupleStreamConsumerLowering;

   LogicalResult matchAndRewriteConsumer(mlir::subop::GatherOp gatherOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const override {
      auto refType = gatherOp.getRef().getColumn().type;
      if (!refType.isa<mlir::subop::TableEntryRefType>()) { return failure(); }
      auto columns = refType.cast<mlir::subop::TableEntryRefType>().getMembers();
      auto currRow = mapping.resolve(gatherOp.getRef());
      for (size_t i = 0; i < columns.getTypes().size(); i++) {
         auto memberName = columns.getNames()[i].cast<mlir::StringAttr>().str();
         if (gatherOp.getMapping().contains(memberName)) {
            auto columnDefAttr = gatherOp.getMapping().get(memberName).cast<mlir::tuples::ColumnDefAttr>();
            auto type = columnDefAttr.getColumn().type;
            size_t accessOffset = i;
            std::vector<mlir::Type> types;
            types.push_back(getBaseType(type));
            if (type.isa<mlir::db::NullableType>()) {
               types.push_back(rewriter.getI1Type());
            }
            auto atOp = rewriter.create<mlir::dsa::At>(gatherOp->getLoc(), types, currRow, accessOffset);
            if (type.isa<mlir::db::NullableType>()) {
               mlir::Value isNull = rewriter.create<mlir::db::NotOp>(gatherOp->getLoc(), atOp.getValid());
               mlir::Value val = rewriter.create<mlir::db::AsNullableOp>(gatherOp->getLoc(), type, atOp.getVal(), isNull);
               mapping.define(columnDefAttr, val);
            } else {
               mapping.define(columnDefAttr, atOp.getVal());
            }
         }
      }
      newStream = mapping.createInFlight(rewriter);
      return success();
   }
};
class ScatterOpLowering : public TupleStreamConsumerLowering<mlir::subop::ScatterOp> {
   public:
   using TupleStreamConsumerLowering<mlir::subop::ScatterOp>::TupleStreamConsumerLowering;

   LogicalResult matchAndRewriteConsumer(mlir::subop::ScatterOp scatterOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const override {
      auto columns = scatterOp.getRef().getColumn().type.cast<mlir::subop::StateEntryReference>().getMembers();

      auto loaded = rewriter.create<mlir::util::LoadOp>(scatterOp->getLoc(), mapping.resolve(scatterOp.getRef()));
      auto unpackedValues = rewriter.create<mlir::util::UnPackOp>(scatterOp->getLoc(), loaded).getResults();
      std::unordered_map<std::string, mlir::Value> values;
      for (auto i = 0ul; i < columns.getTypes().size(); i++) {
         auto name = columns.getNames()[i].cast<mlir::StringAttr>().str();
         values[name] = unpackedValues[i];
      }
      for (auto x : scatterOp.getMapping()) {
         values[x.getName().str()] = mapping.resolve(x.getValue().cast<mlir::tuples::ColumnRefAttr>());
      }
      std::vector<mlir::Value> toStore;
      for (auto i = 0ul; i < columns.getTypes().size(); i++) {
         auto name = columns.getNames()[i].cast<mlir::StringAttr>().str();
         toStore.push_back(values[name]);
      }
      mlir::Value packed = rewriter.create<mlir::util::PackOp>(scatterOp->getLoc(), toStore);
      rewriter.create<mlir::util::StoreOp>(scatterOp->getLoc(), packed, mapping.resolve(scatterOp.getRef()), mlir::Value());
      return success();
   }
};
class GetSingleValLowering : public OpConversionPattern<mlir::subop::GetSingleValOp> {
   using OpConversionPattern<mlir::subop::GetSingleValOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::subop::GetSingleValOp getSingleValOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (auto inFlightOp = mlir::dyn_cast_or_null<mlir::subop::InFlightOp>(adaptor.getStream().getDefiningOp())) {
         rewriter.setInsertionPointAfter(inFlightOp);
         ColumnMapping mapping(inFlightOp);
         rewriter.replaceOp(getSingleValOp, mapping.resolve(getSingleValOp.getColumn()));
         return success();
      }
      return failure();
   }
};
class UnrealizedConversionCastLowering : public OpConversionPattern<mlir::UnrealizedConversionCastOp> {
   using OpConversionPattern<mlir::UnrealizedConversionCastOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::UnrealizedConversionCastOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      SmallVector<Type> convertedTypes;
      if (succeeded(typeConverter->convertTypes(op.getOutputs().getTypes(),
                                                convertedTypes)) &&
          convertedTypes == adaptor.getInputs().getTypes()) {
         rewriter.replaceOp(op, adaptor.getInputs());
         return success();
      }
      convertedTypes.clear();
      if (succeeded(typeConverter->convertTypes(adaptor.getInputs().getTypes(),
                                                convertedTypes)) &&
          convertedTypes == op.getOutputs().getType()) {
         rewriter.replaceOp(op, adaptor.getInputs());
         return success();
      }
      return failure();
   }
};

class ReduceOpLowering : public TupleStreamConsumerLowering<mlir::subop::ReduceOp> {
   public:
   using TupleStreamConsumerLowering<mlir::subop::ReduceOp>::TupleStreamConsumerLowering;

   LogicalResult matchAndRewriteConsumer(mlir::subop::ReduceOp reduceOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter, mlir::Value& newStream, ColumnMapping& mapping) const override {
      auto columns = reduceOp.getRef().getColumn().type.cast<mlir::subop::StateEntryReference>().getMembers();

      auto loaded = rewriter.create<mlir::util::LoadOp>(reduceOp->getLoc(), mapping.resolve(reduceOp.getRef()));
      auto unpackedValues = rewriter.create<mlir::util::UnPackOp>(reduceOp->getLoc(), loaded).getResults();
      std::unordered_map<std::string, mlir::Value> values;
      for (auto i = 0ul; i < columns.getTypes().size(); i++) {
         auto name = columns.getNames()[i].cast<mlir::StringAttr>().str();
         values[name] = unpackedValues[i];
      }
      std::vector<mlir::Value> arguments;
      for (auto attr : reduceOp.getColumns()) {
         arguments.push_back(mapping.resolve(attr.cast<mlir::tuples::ColumnRefAttr>()));
      }
      for (auto member : reduceOp.getMembers()) {
         arguments.push_back(values.at(member.cast<mlir::StringAttr>().str()));
      }
      auto updated = inlineBlock(&reduceOp.getRegion().front(), rewriter, arguments);
      for (size_t i = 0; i < reduceOp.getMembers().size(); i++) {
         values[reduceOp.getMembers()[i].cast<mlir::StringAttr>().str()] = updated[i];
      }
      std::vector<mlir::Value> toStore;
      for (auto i = 0ul; i < columns.getTypes().size(); i++) {
         auto name = columns.getNames()[i].cast<mlir::StringAttr>().str();
         toStore.push_back(values[name]);
      }
      mlir::Value packed = rewriter.create<mlir::util::PackOp>(reduceOp->getLoc(), toStore);
      rewriter.create<mlir::util::StoreOp>(reduceOp->getLoc(), packed, mapping.resolve(reduceOp.getRef()), mlir::Value());
      return success();
   }
};
template <class T>
static std::vector<T> repeat(T val, size_t times) {
   std::vector<T> res{};
   for (auto i = 0ul; i < times; i++) res.push_back(val);
   return res;
}

class LoopLowering : public OpConversionPattern<mlir::subop::LoopOp> {
   public:
   using OpConversionPattern<mlir::subop::LoopOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(mlir::subop::LoopOp loopOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = loopOp->getLoc();
      auto* b = &loopOp.getBody().front();
      auto* terminator = b->getTerminator();
      auto continueOp = mlir::cast<mlir::subop::LoopContinueOp>(terminator);
      mlir::Value trueValue = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, rewriter.getI1Type());
      mlir::Value falseValue = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, rewriter.getI1Type());
      std::vector<mlir::Type> iterTypes;
      std::vector<mlir::Value> iterArgs;

      iterTypes.push_back(rewriter.getI1Type());
      for (auto argumentType : loopOp.getBody().getArgumentTypes()) {
         iterTypes.push_back(typeConverter->convertType(argumentType));
      }
      iterArgs.push_back(trueValue);
      iterArgs.insert(iterArgs.end(), adaptor.getArgs().begin(), adaptor.getArgs().end());
      auto whileOp = rewriter.create<mlir::scf::WhileOp>(loc, iterTypes, iterArgs);
      {
         auto* before = new Block;
         before->addArguments(iterTypes, repeat(loc, iterTypes.size()));
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(before);
         whileOp.getBefore().push_back(before);
         rewriter.create<mlir::scf::ConditionOp>(loc, before->getArgument(0), before->getArguments());
      }
      {
         auto* after = new Block;
         after->addArguments(iterTypes, repeat(loc, iterTypes.size()));

         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(after);
         whileOp.getAfter().push_back(after);
         mlir::BlockAndValueMapping mapper;
         for (size_t i = 0; i < loopOp.getBody().getNumArguments(); i++) {
            mlir::Value whileArg = after->getArguments()[i + 1];
            mlir::Value newArg = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, loopOp.getBody().getArgument(i).getType(), whileArg).getResult(0);
            mapper.map(loopOp.getBody().getArgument(i), newArg);
            rewriter.replaceUsesOfBlockArgument(loopOp.getBody().getArgument(i), newArg);
         }
         for (auto& x : b->getOperations()) {
            if (&x != terminator) {
               auto* clonedWithoutRegions = rewriter.cloneWithoutRegions(x, mapper);
               for (auto i = 0ul; i < x.getNumRegions(); i++) {
                  rewriter.inlineRegionBefore(x.getRegion(i), clonedWithoutRegions->getRegion(i),
                                              clonedWithoutRegions->getRegion(i).end());
               }
            }
         }
         std::vector<mlir::Value> res;
         res.push_back(falseValue);
         for (size_t i = 0; i < continueOp.getValues().size(); i++) {
            res.push_back(rewriter.create<mlir::UnrealizedConversionCastOp>(loc, typeConverter->convertType(continueOp.getValues()[i].getType()), mapper.lookup(continueOp.getValues()[i])).getResult(0));
         }
         res[0] = rewriter.create<mlir::subop::GetSingleValOp>(loc, rewriter.getI1Type(), mapper.lookup(continueOp.getCondStream()), continueOp.getCondColumnAttr());
         rewriter.create<mlir::scf::YieldOp>(loc, res);
      }
      rewriter.replaceOp(loopOp, whileOp.getResults().drop_front());
      rewriter.eraseBlock(b);
      return success();
   }
};

void SubOpToControlFlowLoweringPass::runOnOperation() {
   auto module = getOperation();
   getContext().getLoadedDialect<mlir::util::UtilDialect>()->getFunctionHelper().setParentModule(module);

   // Define Conversion Target
   ConversionTarget target(getContext());
   target.addLegalOp<ModuleOp>();
   //target.addLegalOp<UnrealizedConversionCastOp>();
   target.addIllegalDialect<subop::SubOperatorDialect>();
   target.addLegalDialect<db::DBDialect>();
   target.addLegalDialect<dsa::DSADialect>();

   target.addLegalDialect<tuples::TupleStreamDialect>();
   target.addLegalDialect<func::FuncDialect>();
   target.addLegalDialect<memref::MemRefDialect>();
   target.addLegalDialect<arith::ArithDialect>();
   target.addLegalDialect<cf::ControlFlowDialect>();
   target.addLegalDialect<scf::SCFDialect>();
   target.addLegalDialect<util::UtilDialect>();
   target.addLegalOp<subop::InFlightOp>();
   target.addLegalOp<subop::InFlightTupleOp>();
   target.addDynamicallyLegalOp<subop::GenerateEmitOp>([](mlir::subop::GenerateEmitOp generateEmitOp) -> bool {
      return generateEmitOp->getParentOfType<mlir::subop::GenerateOp>() != nullptr;
   });
   target.addDynamicallyLegalOp<subop::UnionOp>([](mlir::subop::UnionOp unionOp) -> bool {
      bool res = llvm::all_of(unionOp.getStreams(), [](mlir::Value v) { return mlir::isa_and_nonnull<mlir::subop::InFlightOp>(v.getDefiningOp()); });
      return res;
   });
   auto* ctxt = &getContext();

   TypeConverter typeConverter;
   typeConverter.addConversion([](mlir::Type t) { return t; });
   typeConverter.addConversion([&](mlir::subop::TableType t) -> Type {
      return mlir::util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
   });
   typeConverter.addConversion([&](mlir::subop::ResultTableType t) -> Type {
      auto tupleType = mlir::TupleType::get(ctxt, unpackTypes(t.getMembers().getTypes()));
      return mlir::dsa::ResultTableType::get(ctxt, tupleType);
   });
   typeConverter.addConversion([&](mlir::subop::BufferType t) -> Type {
      return mlir::util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
   });
   typeConverter.addConversion([&](mlir::subop::SortedViewType t) -> Type {
      return mlir::util::BufferType::get(t.getContext(), getTupleType(t.getBasedOn()));
   });
   typeConverter.addConversion([&](mlir::subop::SimpleStateType t) -> Type {
      auto tupleType = mlir::TupleType::get(ctxt, unpackTypes(t.getMembers().getTypes()));
      return mlir::util::RefType::get(t.getContext(), tupleType);
   });
   typeConverter.addConversion([&](mlir::subop::HashMapType t) -> Type {
      return mlir::util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
   });
   typeConverter.addConversion([&](mlir::subop::LazyMultiMapType t) -> Type {
      return mlir::util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8));
   });
   typeConverter.addConversion([&](mlir::subop::ListType t) -> Type {
      return mlir::TupleType::get(t.getContext(), {mlir::util::RefType::get(t.getContext(), mlir::IntegerType::get(ctxt, 8)), mlir::IndexType::get(t.getContext())});
   });
   RewritePatternSet patterns(&getContext());
   patterns.insert<FilterLowering>(typeConverter, ctxt);
   patterns.insert<RenameLowering>(typeConverter, ctxt);
   patterns.insert<MapLowering>(typeConverter, ctxt);
   patterns.insert<NestedMapLowering>(typeConverter, ctxt);
   patterns.insert<GetExternalTableLowering>(typeConverter, ctxt);
   patterns.insert<ScanRefsTableLowering>(typeConverter, ctxt);
   patterns.insert<ScanLazyMultiMapLowering>(typeConverter, ctxt);

   patterns.insert<ScanRefsVectorLowering>(typeConverter, ctxt);
   patterns.insert<ScanRefsSortedViewLowering>(typeConverter, ctxt);
   patterns.insert<CreateTableLowering>(typeConverter, ctxt);
   patterns.insert<CreateBufferLowering>(typeConverter, ctxt);
   patterns.insert<CreateHashMapLowering>(typeConverter, ctxt);
   patterns.insert<CreateLazyMultiMapLowering>(typeConverter, ctxt);
   patterns.insert<CreateSimpleStateLowering>(typeConverter, ctxt);
   patterns.insert<ScanRefsSimpleStateLowering>(typeConverter, ctxt);
   patterns.insert<ScanListLowering>(typeConverter, ctxt);
   patterns.insert<MaterializeTableLowering>(typeConverter, ctxt);
   patterns.insert<MaterializeVectorLowering>(typeConverter, ctxt);
   patterns.insert<MaterializeLazyMultiMapLowering>(typeConverter, ctxt);
   patterns.insert<MaintainOpLowering>(typeConverter, ctxt);
   patterns.insert<SetResultOpLowering>(typeConverter, ctxt);
   patterns.insert<LoopLowering>(typeConverter, ctxt);
   patterns.insert<GetSingleValLowering>(typeConverter, ctxt);
   patterns.insert<UnrealizedConversionCastLowering>(typeConverter, ctxt);

   patterns.insert<SortLowering>(typeConverter, ctxt);
   patterns.insert<CombineInFlightLowering>(typeConverter, ctxt);
   patterns.insert<LookupSimpleStateLowering>(typeConverter, ctxt);
   patterns.insert<LookupHashMapLowering>(typeConverter, ctxt);
   patterns.insert<LookupLazyMultiMapLowering>(typeConverter, ctxt);
   patterns.insert<ScanHashMapLowering>(typeConverter, ctxt);
   patterns.insert<ReduceOpLowering>(typeConverter, ctxt);
   patterns.insert<ScatterOpLowering>(typeConverter, ctxt);
   patterns.insert<DefaultGatherOpLowering>(typeConverter, ctxt);
   patterns.insert<TableRefGatherOpLowering>(typeConverter, ctxt);
   patterns.insert<UnionLowering>(typeConverter, ctxt);
   patterns.insert<GenerateLowering>(typeConverter, ctxt);
   patterns.insert<GenerateEmitLowering>(typeConverter, ctxt);
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
   pm.addPass(mlir::subop::createNormalizeSubOpPass());
   pm.addPass(mlir::subop::createPullGatherUpPass());
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