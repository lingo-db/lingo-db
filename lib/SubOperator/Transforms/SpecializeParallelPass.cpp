#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/SubOperator/SubOperatorDialect.h"
#include "mlir/Dialect/SubOperator/SubOperatorOps.h"
#include "mlir/Dialect/SubOperator/Transforms/ColumnUsageAnalysis.h"
#include "mlir/Dialect/SubOperator/Transforms/Passes.h"
#include "mlir/Dialect/SubOperator/Transforms/StateUsageTransformer.h"
#include "mlir/Dialect/SubOperator/Transforms/SubOpDependencyAnalysis.h"
#include "mlir/Dialect/TupleStream/TupleStreamDialect.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace {

class IntroducePreAggrHt : public mlir::RewritePattern {
   const mlir::subop::ColumnUsageAnalysis& analysis;

   public:
   IntroducePreAggrHt(mlir::MLIRContext* context, mlir::subop::ColumnUsageAnalysis& analysis)
      : RewritePattern(mlir::subop::CreateThreadLocalOp::getOperationName(), 1, context), analysis(analysis) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto threadLocalParent = mlir::cast<mlir::subop::CreateThreadLocalOp>(op);
      auto returnOp = mlir::dyn_cast_or_null<mlir::tuples::ReturnOp>(threadLocalParent.getInitFn().front().getTerminator());
      auto createOp = mlir::dyn_cast_or_null<mlir::subop::GenericCreateOp>(returnOp.getResults()[0].getDefiningOp());
      if (!createOp) {
         return mlir::failure();
      }
      auto mapType = mlir::dyn_cast_or_null<mlir::subop::HashMapType>(createOp.getType());
      if (!mapType) {
         return mlir::failure();
      }
      auto resIdx = threadLocalParent.getRes().getUses().begin()->getOperandNumber();
      auto producedState = threadLocalParent->getParentOfType<mlir::subop::ExecutionStepOp>().getResult(resIdx);

      std::vector<mlir::subop::LookupOrInsertOp> lookupOrInsertOps;
      mlir::subop::MergeOp mergeOp;
      mlir::subop::ExecutionStepOp mergeStep;
      std::vector<mlir::Value> localExecutionSteps;
      for (auto& threadLocalUse : producedState.getUses()) {
         if (auto executionStep = mlir::dyn_cast_or_null<mlir::subop::ExecutionStepOp>(threadLocalUse.getOwner())) {
            auto val = executionStep.getSubOps().getArgument(threadLocalUse.getOperandNumber());
            auto localUses = executionStep.getSubOps().getArgument(threadLocalUse.getOperandNumber()).getUses();

            if (!localUses.empty()) {
               if (auto merge = mlir::dyn_cast<mlir::subop::MergeOp>(localUses.begin()->getOwner())) {
                  mergeOp = merge;
                  mergeStep = executionStep;


                  std::function<bool(mlir::Value)> checkUse = [&](mlir::Value v) -> bool {

                     for (auto& use : v.getUses()) {
                        if (mlir::isa<mlir::subop::ScanRefsOp, mlir::subop::LookupOp>(use.getOwner())) {
                        }else if (auto executionStep = mlir::dyn_cast<mlir::subop::ExecutionStepOp>(use.getOwner())) {
                           if(!checkUse(executionStep.getSubOps().getArgument(use.getOperandNumber()))){
                              return false;
                           }
                        } else {
                           return false;
                        }
                     }
                     return true;
                  };
                  if(!checkUse(mergeStep.getResult(0))){
                     return mlir::failure();
                  }
                  /*for (auto* mergedUser : executionStep.getResult(0).getUsers()) {
                     if (mlir::isa<mlir::subop::ScanRefsOp, mlir::subop::LookupOp>(mergedUser)) {
                     } else {
                        //todo
                        return mlir::failure();
                     }
                  }*/
               } else {
                  std::function<bool(mlir::Value)> checkUse = [&](mlir::Value v) -> bool {

                     for (auto& use : v.getUses()) {
                        if (mlir::isa<mlir::subop::LookupOrInsertOp>(use.getOwner())) {
                        }else if (auto executionStep = mlir::dyn_cast<mlir::subop::ExecutionStepOp>(use.getOwner())) {
                           if(!checkUse(executionStep.getSubOps().getArgument(use.getOperandNumber()))){
                              return false;
                           }
                        } else {
                           return false;
                        }
                     }
                     return true;
                  };
                  if (!checkUse(val)) {
                     return mlir::failure();
                  }
                  localExecutionSteps.push_back(val);
               }
            }
         } else {
            return mlir::failure();
         }
      }
      auto optimisticHt = mlir::subop::PreAggrHtType::get(getContext(), mapType.getKeyMembers(), mapType.getValueMembers());
      auto optimisticHtFragment = mlir::subop::PreAggrHtFragmentType::get(getContext(), mapType.getKeyMembers(), mapType.getValueMembers());
      auto threadLocalType = mlir::subop::ThreadLocalType::get(getContext(), optimisticHtFragment);
      mlir::TypeConverter fragmentTypeConverter;

      fragmentTypeConverter.addConversion([&](mlir::subop::LookupEntryRefType lookupRefType) {
         return mlir::subop::LookupEntryRefType::get(lookupRefType.getContext(), mlir::cast<mlir::subop::LookupAbleState>(fragmentTypeConverter.convertType(lookupRefType.getState())));
      });
      fragmentTypeConverter.addConversion([&](mlir::subop::HashMapType mapType) {
         return optimisticHtFragment;
      });
      mlir::TypeConverter htTypeConverter;
      htTypeConverter.addConversion([&](mlir::subop::HashMapType mapType) {
         return optimisticHt;
      });
      htTypeConverter.addConversion([&](mlir::subop::HashMapEntryRefType refType) {
         return mlir::subop::PreAggrHTEntryRefType::get(refType.getContext(), optimisticHt);
      });
      htTypeConverter.addConversion([&](mlir::subop::LookupEntryRefType lookupRefType) {
         return mlir::subop::LookupEntryRefType::get(lookupRefType.getContext(), mlir::cast<mlir::subop::LookupAbleState>(htTypeConverter.convertType(lookupRefType.getState())));
      });

      htTypeConverter.addConversion([&](mlir::subop::ListType listType) {
         return mlir::subop::ListType::get(listType.getContext(), mlir::cast<mlir::subop::StateEntryReference>(htTypeConverter.convertType(listType.getT())));
      });
      htTypeConverter.addConversion([&](mlir::subop::OptionalType optionalType) {
         return mlir::subop::OptionalType::get(optionalType.getContext(), mlir::cast<mlir::subop::StateEntryReference>(htTypeConverter.convertType(optionalType.getT())));
      });
      mlir::subop::SubOpStateUsageTransformer fragmentTransformer(analysis, getContext(), [&](mlir::Operation* op, mlir::Type type) -> mlir::Type {
         return fragmentTypeConverter.convertType(type);
      });
      mlir::subop::SubOpStateUsageTransformer htTransformer(analysis, getContext(), [&](mlir::Operation* op, mlir::Type type) -> mlir::Type {
         return htTypeConverter.convertType(type);
      });
      for (auto v:localExecutionSteps) {
         fragmentTransformer.updateValue(v, optimisticHtFragment);
         v.setType(optimisticHtFragment);
      }
      htTransformer.updateValue(mergeStep.getResult(0), optimisticHt);
      mergeStep.getResult(0).setType(optimisticHt);
      mergeStep.getSubOps().getArgument(0).setType(threadLocalType);

      producedState.setType(threadLocalType);
      auto loc = op->getLoc();
      auto newCreateThreadLocal = rewriter.create<mlir::subop::CreateThreadLocalOp>(loc, threadLocalType);
      auto* newBlock = new mlir::Block;
      newCreateThreadLocal.getInitFn().push_back(newBlock);
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(newBlock);
         rewriter.create<mlir::tuples::ReturnOp>(loc, rewriter.create<mlir::subop::GenericCreateOp>(loc, optimisticHtFragment).getRes());
      }

      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPoint(mergeOp);
         auto newMergeOp = rewriter.create<mlir::subop::MergeOp>(mergeOp->getLoc(), optimisticHt, mergeStep.getSubOps().getArgument(0));
         if (!mergeOp.getCombineFn().empty()) {
            mlir::IRMapping mapping;
            mergeOp.getCombineFn().cloneInto(&newMergeOp.getCombineFn(), mapping);
         }
         if (!mergeOp.getEqFn().empty()) {
            mlir::IRMapping mapping;
            mergeOp.getEqFn().cloneInto(&newMergeOp.getEqFn(), mapping);
         }
         rewriter.replaceOp(mergeOp, newMergeOp.getRes());
      }
      rewriter.replaceOp(threadLocalParent, newCreateThreadLocal.getRes());
      return mlir::success();
   }
};
class SpecializeParallelSubOpPass : public mlir::PassWrapper<SpecializeParallelSubOpPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SpecializeParallelSubOpPass)
   virtual llvm::StringRef getArgument() const override { return "subop-specialize-parallel"; }

   SpecializeParallelSubOpPass() {}
   void getDependentDialects(mlir::DialectRegistry& registry) const override {
      registry.insert<mlir::util::UtilDialect, mlir::db::DBDialect>();
   }
   void runOnOperation() override {
      //transform "standalone" aggregation functions
      auto columnUsageAnalysis = getAnalysis<mlir::subop::ColumnUsageAnalysis>();

      mlir::RewritePatternSet patterns(&getContext());
      patterns.insert<IntroducePreAggrHt>(&getContext(), columnUsageAnalysis);

      if (mlir::applyPatternsAndFoldGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
         assert(false && "should not happen");
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
mlir::subop::createSpecializeParallelPass() { return std::make_unique<SpecializeParallelSubOpPass>(); }