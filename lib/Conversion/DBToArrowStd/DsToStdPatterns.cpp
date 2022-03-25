#include "mlir/Conversion/DBToArrowStd/DBToArrowStd.h"

#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"
using namespace mlir;
namespace {

class CreateHashtableCreateLowering : public ConversionPattern {
   mlir::db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit CreateHashtableCreateLowering(TypeConverter& typeConverter, MLIRContext* context, db::codegen::FunctionRegistry& functionRegistry)
      : ConversionPattern(typeConverter, mlir::db::HashtableCreate::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto createOp = mlir::cast<mlir::db::HashtableCreate>(op);
      auto builderType = createOp.ht().getType().cast<mlir::db::JoinHashtableType>();
      auto entryType = mlir::TupleType::get(rewriter.getContext(), {builderType.getKeyType(), builderType.getValType()});
      auto tupleType = mlir::TupleType::get(rewriter.getContext(), {rewriter.getIndexType(), entryType});
      Value typesize = rewriter.create<mlir::util::SizeOfOp>(op->getLoc(), rewriter.getIndexType(), typeConverter->convertType(tupleType));
      Value ptr = functionRegistry.call(rewriter, op->getLoc(), mlir::db::codegen::FunctionRegistry::FunctionId::JoinHtCreate, typesize)[0];
      rewriter.replaceOpWithNewOp<util::GenericMemrefCastOp>(op, typeConverter->convertType(createOp.ht().getType()), ptr);

      return success();
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
} // end namespace
namespace mlir::db {
void populateDsToStdPatterns(mlir::db::codegen::FunctionRegistry& functionRegistry, mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns) {
   patterns.insert<CreateHashtableCreateLowering>(typeConverter, patterns.getContext(), functionRegistry);
   patterns.insert<HashtableInsertLowering>(typeConverter, patterns.getContext(), functionRegistry);
   patterns.insert<HashtableFinalizeLowering>(typeConverter, patterns.getContext(), functionRegistry);
}
} // end namespace mlir::db