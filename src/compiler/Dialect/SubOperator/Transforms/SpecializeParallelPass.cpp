#include "llvm/ADT/TypeSwitch.h"

#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnUsageAnalysis.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/StateUsageTransformer.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/SubOpDependencyAnalysis.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace {
using namespace lingodb::compiler::dialect;

class IntroducePreAggrHt : public mlir::RewritePattern {
   const subop::ColumnUsageAnalysis& analysis;

   public:
   IntroducePreAggrHt(mlir::MLIRContext* context, subop::ColumnUsageAnalysis& analysis)
      : RewritePattern(subop::CreateThreadLocalOp::getOperationName(), 1, context), analysis(analysis) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto threadLocalParent = mlir::cast<subop::CreateThreadLocalOp>(op);
      auto returnOp = mlir::dyn_cast_or_null<tuples::ReturnOp>(threadLocalParent.getInitFn().front().getTerminator());
      auto createOp = mlir::dyn_cast_or_null<subop::GenericCreateOp>(returnOp.getResults()[0].getDefiningOp());
      if (!createOp) {
         return mlir::failure();
      }
      auto mapType = mlir::dyn_cast_or_null<subop::HashMapType>(createOp.getType());
      if (!mapType) {
         return mlir::failure();
      }
      auto resIdx = threadLocalParent.getRes().getUses().begin()->getOperandNumber();
      auto producedState = threadLocalParent->getParentOfType<subop::ExecutionStepOp>().getResult(resIdx);

      std::vector<subop::LookupOrInsertOp> lookupOrInsertOps;
      subop::MergeOp mergeOp;
      subop::ExecutionStepOp mergeStep;
      std::vector<mlir::Value> localExecutionSteps;
      for (auto& threadLocalUse : producedState.getUses()) {
         if (auto executionStep = mlir::dyn_cast_or_null<subop::ExecutionStepOp>(threadLocalUse.getOwner())) {
            auto val = executionStep.getSubOps().getArgument(threadLocalUse.getOperandNumber());
            auto localUses = executionStep.getSubOps().getArgument(threadLocalUse.getOperandNumber()).getUses();

            if (!localUses.empty()) {
               if (auto merge = mlir::dyn_cast<subop::MergeOp>(localUses.begin()->getOwner())) {
                  mergeOp = merge;
                  mergeStep = executionStep;

                  std::function<bool(mlir::Value)> checkUse = [&](mlir::Value v) -> bool {
                     for (auto& use : v.getUses()) {
                        if (mlir::isa<subop::ScanRefsOp, subop::LookupOp>(use.getOwner())) {
                        } else if (auto executionStep = mlir::dyn_cast<subop::ExecutionStepOp>(use.getOwner())) {
                           if (!checkUse(executionStep.getSubOps().getArgument(use.getOperandNumber()))) {
                              return false;
                           }
                        } else {
                           return false;
                        }
                     }
                     return true;
                  };
                  if (!checkUse(mergeStep.getResult(0))) {
                     return mlir::failure();
                  }
                  /*for (auto* mergedUser : executionStep.getResult(0).getUsers()) {
                     if (mlir::isa<subop::ScanRefsOp, subop::LookupOp>(mergedUser)) {
                     } else {
                        //todo
                        return mlir::failure();
                     }
                  }*/
               } else {
                  std::function<bool(mlir::Value)> checkUse = [&](mlir::Value v) -> bool {
                     for (auto& use : v.getUses()) {
                        if (mlir::isa<subop::LookupOrInsertOp>(use.getOwner())) {
                        } else if (auto executionStep = mlir::dyn_cast<subop::ExecutionStepOp>(use.getOwner())) {
                           if (!checkUse(executionStep.getSubOps().getArgument(use.getOperandNumber()))) {
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
      auto optimisticHt = subop::PreAggrHtType::get(getContext(), mapType.getKeyMembers(), mapType.getValueMembers());
      auto optimisticHtFragment = subop::PreAggrHtFragmentType::get(getContext(), mapType.getKeyMembers(), mapType.getValueMembers());
      auto threadLocalType = subop::ThreadLocalType::get(getContext(), optimisticHtFragment);
      mlir::TypeConverter fragmentTypeConverter;

      fragmentTypeConverter.addConversion([&](subop::LookupEntryRefType lookupRefType) {
         return subop::LookupEntryRefType::get(lookupRefType.getContext(), mlir::cast<subop::LookupAbleState>(fragmentTypeConverter.convertType(lookupRefType.getState())));
      });
      fragmentTypeConverter.addConversion([&](subop::HashMapType mapType) {
         return optimisticHtFragment;
      });
      mlir::TypeConverter htTypeConverter;
      htTypeConverter.addConversion([&](subop::HashMapType mapType) {
         return optimisticHt;
      });
      htTypeConverter.addConversion([&](subop::HashMapEntryRefType refType) {
         return subop::PreAggrHTEntryRefType::get(refType.getContext(), optimisticHt);
      });
      htTypeConverter.addConversion([&](subop::LookupEntryRefType lookupRefType) {
         return subop::LookupEntryRefType::get(lookupRefType.getContext(), mlir::cast<subop::LookupAbleState>(htTypeConverter.convertType(lookupRefType.getState())));
      });

      htTypeConverter.addConversion([&](subop::ListType listType) {
         return subop::ListType::get(listType.getContext(), mlir::cast<subop::StateEntryReference>(htTypeConverter.convertType(listType.getT())));
      });
      htTypeConverter.addConversion([&](subop::OptionalType optionalType) {
         return subop::OptionalType::get(optionalType.getContext(), mlir::cast<subop::StateEntryReference>(htTypeConverter.convertType(optionalType.getT())));
      });
      subop::SubOpStateUsageTransformer fragmentTransformer(analysis, getContext(), [&](mlir::Operation* op, mlir::Type type) -> mlir::Type {
         return fragmentTypeConverter.convertType(type);
      });
      subop::SubOpStateUsageTransformer htTransformer(analysis, getContext(), [&](mlir::Operation* op, mlir::Type type) -> mlir::Type {
         return htTypeConverter.convertType(type);
      });
      for (auto v : localExecutionSteps) {
         fragmentTransformer.updateValue(v, optimisticHtFragment);
         v.setType(optimisticHtFragment);
      }
      htTransformer.updateValue(mergeStep.getResult(0), optimisticHt);
      mergeStep.getResult(0).setType(optimisticHt);
      mergeStep.getSubOps().getArgument(0).setType(threadLocalType);

      producedState.setType(threadLocalType);
      auto loc = op->getLoc();
      auto newCreateThreadLocal = rewriter.create<subop::CreateThreadLocalOp>(loc, threadLocalType);
      auto* newBlock = new mlir::Block;
      newCreateThreadLocal.getInitFn().push_back(newBlock);
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(newBlock);
         rewriter.create<tuples::ReturnOp>(loc, rewriter.create<subop::GenericCreateOp>(loc, optimisticHtFragment).getRes());
      }

      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPoint(mergeOp);
         auto newMergeOp = rewriter.create<subop::MergeOp>(mergeOp->getLoc(), optimisticHt, mergeStep.getSubOps().getArgument(0));
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
      registry.insert<util::UtilDialect, db::DBDialect>();
   }
   void runOnOperation() override {
      //transform "standalone" aggregation functions
      auto columnUsageAnalysis = getAnalysis<subop::ColumnUsageAnalysis>();

      mlir::RewritePatternSet patterns(&getContext());
      patterns.insert<IntroducePreAggrHt>(&getContext(), columnUsageAnalysis);

      if (mlir::applyPatternsGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
         assert(false && "should not happen");
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
subop::createSpecializeParallelPass() { return std::make_unique<SpecializeParallelSubOpPass>(); }