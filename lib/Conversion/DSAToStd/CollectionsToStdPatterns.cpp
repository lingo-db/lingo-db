#include "mlir/Conversion/DSAToStd/CollectionIteration.h"
#include "mlir/Conversion/DSAToStd/DSAToStd.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
namespace {

class ForOpLowering : public OpConversionPattern<mlir::dsa::ForOp> {
   public:
   using OpConversionPattern<mlir::dsa::ForOp>::OpConversionPattern;
   std::vector<Value> remap(std::vector<Value> values, ConversionPatternRewriter& builder) const {
      for (size_t i = 0; i < values.size(); i++) {
         values[i] = builder.getRemappedValue(values[i]);
      }
      return values;
   }

   LogicalResult matchAndRewrite(mlir::dsa::ForOp forOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      std::vector<Type> argumentTypes;
      std::vector<Location> argumentLocs;
      for (auto t : forOp.getRegion().getArgumentTypes()) {
         argumentTypes.push_back(t);
         argumentLocs.push_back(forOp->getLoc());
      }
      auto collectionType = mlir::dyn_cast_or_null<mlir::util::CollectionType>(forOp.getCollection().getType());
      auto iterator = mlir::dsa::CollectionIterationImpl::getImpl(collectionType, adaptor.getCollection());

      ModuleOp parentModule = forOp->getParentOfType<ModuleOp>();
      std::vector<Value> results = iterator->implementLoop(forOp->getLoc(), adaptor.getInitArgs(), *typeConverter, rewriter, parentModule, [&](std::function<Value(OpBuilder & b)> getElem, ValueRange iterargs, OpBuilder builder) {
         auto yieldOp = cast<mlir::dsa::YieldOp>(forOp.getBody()->getTerminator());
         std::vector<Type> resTypes;
         std::vector<Location> locs;
         for (auto t : yieldOp.getResults()) {
            resTypes.push_back(typeConverter->convertType(t.getType()));
            locs.push_back(forOp->getLoc());
         }
         std::vector<Value> values;
         values.push_back(getElem(builder));
         values.insert(values.end(), iterargs.begin(), iterargs.end());
         auto term = builder.create<mlir::scf::YieldOp>(forOp->getLoc());
         builder.setInsertionPoint(term);
         rewriter.inlineBlockBefore(forOp.getBody(), &*builder.getInsertionPoint(), values);

         std::vector<Value> results(yieldOp.getResults().begin(), yieldOp.getResults().end());
         rewriter.eraseOp(yieldOp);
         rewriter.eraseOp(term);

         return results;
      });
      {
         OpBuilder::InsertionGuard insertionGuard(rewriter);

         forOp.getRegion().push_back(new Block());
         forOp.getRegion().front().addArguments(argumentTypes, argumentLocs);
         rewriter.setInsertionPointToStart(&forOp.getRegion().front());
         rewriter.create<mlir::dsa::YieldOp>(forOp.getLoc());
      }

      rewriter.replaceOp(forOp, results);
      return success();
   }
};
} // namespace

void mlir::dsa::populateCollectionsToStdPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns) {
   auto* context = patterns.getContext();

   patterns.insert<ForOpLowering>(typeConverter, context);


}