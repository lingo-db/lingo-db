#include "llvm/ADT/Sequence.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/util/Passes.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToLLVM RewritePatterns
//===----------------------------------------------------------------------===//

namespace {
/// Return a value representing an access into a global string with the given
/// name, creating the string if necessary.
static Value getOrCreateGlobalString(Location loc, OpBuilder& builder, StringRef value,
                                     ModuleOp module) {
   static std::unordered_map<std::string, std::string> names;
   if (!names.count(value.str())) {
      names[value.str()] = std::string("print_str_") + std::to_string(names.size());
   }
   StringRef name = StringRef(names[value.str()]);

   // Create the global at the entry of the module.
   LLVM::GlobalOp global;
   if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMArrayType::get(
         builder.getIntegerType(8), value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                              LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value));
   }

   // Get the pointer to the first character in the global string.
   Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
   Value cst0 = builder.create<LLVM::ConstantOp>(
      loc, builder.getI64Type(),
      builder.getIntegerAttr(builder.getIndexType(), 0));
   return builder.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(builder.getIntegerType(8)), globalPtr,
      ArrayRef<Value>({cst0, cst0}));
}
static mlir::LLVM::LLVMStructType convertTuple(TupleType tupleType, TypeConverter& typeConverter) {
   std::vector<Type> types;
   for (auto t : tupleType.getTypes()) {
      t.dump();
      types.push_back(typeConverter.convertType(t));
   }
   return mlir::LLVM::LLVMStructType::getLiteral(tupleType.getContext(), types);
}
/// Lowers `toy.print` to a loop nest calling `printf` on each of the individual
/// elements of the array.
class StringConstOpLowering : public ConversionPattern {
   public:
   explicit StringConstOpLowering(MLIRContext* context)
      : ConversionPattern(mlir::util::StringConstantOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto constop = mlir::dyn_cast_or_null<mlir::util::StringConstantOp>(op);
      auto ptr = getOrCreateGlobalString(rewriter.getUnknownLoc(), rewriter, constop.val(), op->getParentOfType<ModuleOp>());
      auto len = rewriter.create<mlir::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIntegerAttr(rewriter.getI32Type(), constop.val().size()));
      constop.ptr().replaceAllUsesWith(ptr);
      constop.len().replaceAllUsesWith(len);
      rewriter.eraseOp(op);
      return success();
   }
};
class CombineOpLowering : public ConversionPattern {
   public:
   explicit CombineOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::CombineOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto constop = mlir::dyn_cast_or_null<mlir::util::CombineOp>(op);
      auto tupleType = constop.tuple().getType().dyn_cast_or_null<TupleType>();
      auto structType = convertTuple(tupleType, *typeConverter);
      Value tpl = rewriter.create<LLVM::UndefOp>(rewriter.getUnknownLoc(), structType);
      unsigned pos = 0;
      for (auto val : constop.vals()) {
         tpl = rewriter.create<LLVM::InsertValueOp>(rewriter.getUnknownLoc(), tpl, val,
                                                    rewriter.getI64ArrayAttr(pos++));
      }
      rewriter.replaceOp(op, tpl);
      return success();
   }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ToyToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace {
struct UtilToLLVMLoweringPass
   : public PassWrapper<UtilToLLVMLoweringPass, OperationPass<ModuleOp>> {
   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
   }
   void runOnOperation() final;
};
} // end anonymous namespace

void mlir::util::populateUtilToLLVMConversionPatterns(LLVMTypeConverter& typeConverter, RewritePatternSet& patterns) {
   typeConverter.addConversion([&](mlir::TupleType tupleType) {
      return convertTuple(tupleType, typeConverter);
   });

   patterns.add<StringConstOpLowering>(patterns.getContext());
   patterns.add<CombineOpLowering>(typeConverter, patterns.getContext());
}
