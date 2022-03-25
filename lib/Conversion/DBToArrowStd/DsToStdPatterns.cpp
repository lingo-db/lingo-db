#include "mlir/Conversion/DBToArrowStd/DBToArrowStd.h"

#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"
using namespace mlir;
namespace {
class VectorHelper2 {
   Type elementType;
   Location loc;

   public:
   static Type createSType(MLIRContext* context, Type elementType) {
      return mlir::TupleType::get(context, {IndexType::get(context), IndexType::get(context), mlir::util::RefType::get(context, elementType)});
   }
   static mlir::util::RefType createType(MLIRContext* context, Type elementType) {
      return mlir::util::RefType::get(context, createSType(context, elementType));
   }
   VectorHelper2(Type elementType, Location loc) : elementType(elementType), loc(loc) {
   }
   Value create(mlir::OpBuilder& builder, Value initialCapacity, mlir::db::codegen::FunctionRegistry& functionRegistry) {
      auto typeSize = builder.create<mlir::util::SizeOfOp>(loc, builder.getIndexType(), elementType);
      auto ptr = functionRegistry.call(builder, loc, mlir::db::codegen::FunctionRegistry::FunctionId::VecCreate, ValueRange({typeSize, initialCapacity}))[0];
      return builder.create<mlir::util::GenericMemrefCastOp>(loc, createType(builder.getContext(), elementType), ptr);
   }
   void insert(mlir::OpBuilder& builder, Value vec, Value newVal, mlir::db::codegen::FunctionRegistry& functionRegistry) {
      auto idxType = builder.getIndexType();
      auto idxPtrType = util::RefType::get(builder.getContext(), idxType);
      auto valuesType = mlir::util::RefType::get(builder.getContext(), elementType);
      Value lenAddress = builder.create<util::TupleElementPtrOp>(loc, idxPtrType, vec, 0);
      Value capacityAddress = builder.create<util::TupleElementPtrOp>(loc, idxPtrType, vec, 1);

      auto len = builder.create<mlir::util::LoadOp>(loc, idxType, lenAddress, Value());
      auto capacity = builder.create<mlir::util::LoadOp>(loc, idxType, capacityAddress, Value());
      Value cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, len, capacity);
      builder.create<scf::IfOp>(
         loc, TypeRange({}), cmp, [&](OpBuilder& b, Location loc) { b.create<scf::YieldOp>(loc); }, [&](OpBuilder& b, Location loc) {
            Value downCasted = b.create<util::GenericMemrefCastOp>(loc, mlir::util::RefType::get(b.getContext(),b.getI8Type()), vec);
            functionRegistry.call(b,loc,mlir::db::codegen::FunctionRegistry::FunctionId::VecResize,ValueRange{downCasted});
            b.create<scf::YieldOp>(loc); });
      Value valuesAddress = builder.create<util::TupleElementPtrOp>(loc, util::RefType::get(builder.getContext(), valuesType), vec, 2);
      auto values = builder.create<mlir::util::LoadOp>(loc, valuesType, valuesAddress, Value());

      builder.create<util::StoreOp>(loc, newVal, values, len);
      Value one = builder.create<arith::ConstantIndexOp>(loc, 1);

      Value newLen = builder.create<arith::AddIOp>(loc, len, one);

      builder.create<mlir::util::StoreOp>(loc, newLen, lenAddress, Value());
   }
};
class CreateDsLowering : public ConversionPattern {
   mlir::db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit CreateDsLowering(TypeConverter& typeConverter, MLIRContext* context, db::codegen::FunctionRegistry& functionRegistry)
      : ConversionPattern(typeConverter, mlir::db::CreateDS::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto createOp = mlir::cast<mlir::db::CreateDS>(op);
      if (auto joinHtType = createOp.ds().getType().dyn_cast<mlir::db::JoinHashtableType>()) {
         auto entryType = mlir::TupleType::get(rewriter.getContext(), {joinHtType.getKeyType(), joinHtType.getValType()});
         auto tupleType = mlir::TupleType::get(rewriter.getContext(), {rewriter.getIndexType(), entryType});
         Value typesize = rewriter.create<mlir::util::SizeOfOp>(op->getLoc(), rewriter.getIndexType(), typeConverter->convertType(tupleType));
         Value ptr = functionRegistry.call(rewriter, op->getLoc(), mlir::db::codegen::FunctionRegistry::FunctionId::JoinHtCreate, typesize)[0];
         rewriter.replaceOpWithNewOp<util::GenericMemrefCastOp>(op, typeConverter->convertType(joinHtType), ptr);
         return success();
      } else if (auto vecType = createOp.ds().getType().dyn_cast<mlir::db::VectorType>()) {
         Value initialCapacity = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 1024);
         auto elementType = typeConverter->convertType(vecType.getElementType());
         VectorHelper2 vectorHelper(elementType, op->getLoc());
         rewriter.replaceOp(op, vectorHelper.create(rewriter, initialCapacity, functionRegistry));
         return success();
      }
      return failure();
   }
};
class HashtableInsertLowering : public ConversionPattern {
   mlir::db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit HashtableInsertLowering(TypeConverter& typeConverter, MLIRContext* context, db::codegen::FunctionRegistry& functionRegistry)
      : ConversionPattern(typeConverter, mlir::db::HashtableInsert::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      mlir::db::HashtableInsertAdaptor adaptor(operands);
      Value hashed = rewriter.create<mlir::db::Hash>(op->getLoc(), rewriter.getIndexType(), adaptor.key());
      mlir::Value val = adaptor.val();
      if (!val) {
         val = rewriter.create<mlir::util::UndefTupleOp>(op->getLoc(), mlir::TupleType::get(getContext()));
      }
      auto entry = rewriter.create<mlir::util::PackOp>(op->getLoc(), mlir::ValueRange({adaptor.key(), val}));
      auto bucket = rewriter.create<mlir::util::PackOp>(op->getLoc(), mlir::ValueRange({hashed, entry}));
      auto loc = op->getLoc();
      auto idxType = rewriter.getIndexType();
      auto idxPtrType = util::RefType::get(rewriter.getContext(), idxType);
      auto valuesType = mlir::util::RefType::get(rewriter.getContext(), bucket.getType());
      Value lenAddress = rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, adaptor.ht(), 2);
      Value capacityAddress = rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, adaptor.ht(), 3);

      auto len = rewriter.create<mlir::util::LoadOp>(loc, idxType, lenAddress, Value());
      auto capacity = rewriter.create<mlir::util::LoadOp>(loc, idxType, capacityAddress, Value());
      Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, len, capacity);
      rewriter.create<scf::IfOp>(
         loc, TypeRange({}), cmp, [&](OpBuilder& b, Location loc) { b.create<scf::YieldOp>(loc); }, [&](OpBuilder& b, Location loc) {
            Value downCasted = b.create<util::GenericMemrefCastOp>(loc, mlir::util::RefType::get(b.getContext(),b.getI8Type()), adaptor.ht());
            functionRegistry.call(b,loc,mlir::db::codegen::FunctionRegistry::FunctionId::JoinHtResize,ValueRange{downCasted});
            b.create<scf::YieldOp>(loc); });
      Value valuesAddress = rewriter.create<util::TupleElementPtrOp>(loc, mlir::util::RefType::get(getContext(), adaptor.ht().getType().cast<mlir::util::RefType>().getElementType().cast<mlir::TupleType>().getType(4)), adaptor.ht(), 4);
      Value castedValuesAddress = rewriter.create<mlir::util::GenericMemrefCastOp>(loc, mlir::util::RefType::get(getContext(), valuesType), valuesAddress);
      auto values = rewriter.create<mlir::util::LoadOp>(loc, valuesType, castedValuesAddress, Value());
      rewriter.create<util::StoreOp>(loc, bucket, values, len);
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

      Value newLen = rewriter.create<arith::AddIOp>(loc, len, one);

      rewriter.create<mlir::util::StoreOp>(loc, newLen, lenAddress, Value());
      rewriter.eraseOp(op);
      return success();
   }
};
class HashtableFinalizeLowering : public ConversionPattern {
   mlir::db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit HashtableFinalizeLowering(TypeConverter& typeConverter, MLIRContext* context, db::codegen::FunctionRegistry& functionRegistry)
      : ConversionPattern(typeConverter, mlir::db::HashtableFinalize::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      mlir::db::HashtableFinalizeAdaptor adaptor(operands);
      Value downCasted = rewriter.create<util::GenericMemrefCastOp>(op->getLoc(), mlir::util::RefType::get(rewriter.getContext(), rewriter.getI8Type()), adaptor.ht());
      functionRegistry.call(rewriter, op->getLoc(), mlir::db::codegen::FunctionRegistry::FunctionId::JoinHtFinalize, ValueRange{downCasted});
      rewriter.eraseOp(op);
      return success();
   }
};
class DSAppendLowering : public ConversionPattern {
   mlir::db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit DSAppendLowering(TypeConverter& typeConverter, MLIRContext* context, db::codegen::FunctionRegistry& functionRegistry)
      : ConversionPattern(typeConverter, mlir::db::Append::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto appendOp=mlir::cast<mlir::db::Append>(op);
      mlir::db::AppendAdaptor adaptor(operands);
      Value builderVal = adaptor.ds();
      Value v = adaptor.val();
      auto convertedElementType = typeConverter->convertType(appendOp.ds().getType().cast<mlir::db::VectorType>().getElementType());
      VectorHelper2 helper(convertedElementType, op->getLoc());
      helper.insert(rewriter, builderVal, v, functionRegistry);
      rewriter.eraseOp(op);
      return success();
   }
};
} // end namespace
namespace mlir::db {
void populateDsToStdPatterns(mlir::db::codegen::FunctionRegistry& functionRegistry, mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns) {
   patterns.insert<CreateDsLowering>(typeConverter, patterns.getContext(), functionRegistry);
   patterns.insert<HashtableInsertLowering>(typeConverter, patterns.getContext(), functionRegistry);
   patterns.insert<HashtableFinalizeLowering>(typeConverter, patterns.getContext(), functionRegistry);
   patterns.insert<DSAppendLowering>(typeConverter, patterns.getContext(), functionRegistry);
}
} // end namespace mlir::db