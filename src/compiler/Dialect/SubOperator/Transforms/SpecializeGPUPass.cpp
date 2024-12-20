#include "llvm/ADT/TypeSwitch.h"

#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnUsageAnalysis.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/StateUsageTransformer.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/SubOpDependencyAnalysis.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace {
using namespace lingodb::compiler::dialect;
/*

*/
class IntroducePreAggrHt : public mlir::RewritePattern {
   const subop::ColumnUsageAnalysis& analysis;

   public:
   IntroducePreAggrHt(mlir::MLIRContext* context, subop::ColumnUsageAnalysis& analysis)
      : RewritePattern(subop::GenericCreateOp::getOperationName(), 1, context), analysis(analysis) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      subop::GenericCreateOp creationOp = mlir::cast<subop::GenericCreateOp>(op);
      auto mapType = mlir::dyn_cast_or_null<subop::HashMapType>(creationOp.getType());
      if (!mapType) {
         return mlir::failure();
      }
      if (mlir::dyn_cast_or_null<subop::CreateKernelLocalOp>(creationOp->getParentOp())) {
         // Do not start from kernel local.
         return mlir::failure();
      }
      auto resIdx = creationOp.getRes().getUses().begin()->getOperandNumber();
      auto stateAsStepResult = creationOp->getParentOfType<subop::ExecutionStepOp>().getResult(resIdx);

      std::vector<mlir::Value> stepLocalUsesOfMap;
      std::vector<mlir::Value> contextSwitchesAfterCreation; // creation will be replaced anyways
      std::vector<subop::CreateKernelLocalOp> kernelLocalsCreation;

      std::pair<mlir::Value, subop::CreateKernelLocalOp> stepRegionArgToKernelLocals;
      subop::MergeIntoOp mergeIntoOp;
      subop::ExecutionStepOp stepWithMerge;
      subop::ExecutionStepOp lastInserterStep;
      for (auto& usage : stateAsStepResult.getUses()) {
         if (auto userExecutionStep = mlir::dyn_cast_or_null<subop::ExecutionStepOp>(usage.getOwner())) {
            mlir::Value stepRegionArg = userExecutionStep.getSubOps().getArgument(usage.getOperandNumber());
            if (auto stepMergeIntoOp = mlir::dyn_cast<subop::MergeIntoOp>(stepRegionArg.getUses().begin()->getOwner())) {
               mergeIntoOp = stepMergeIntoOp; // remember op for outlining
               stepWithMerge = userExecutionStep;
               lastInserterStep = userExecutionStep;
               stepLocalUsesOfMap.push_back(stepRegionArg);
               stepLocalUsesOfMap.push_back(mergeIntoOp.getKernelLocal());
               stepRegionArgToKernelLocals.first = stepRegionArg;
               stepRegionArgToKernelLocals.second = mlir::cast<subop::CreateKernelLocalOp>(mergeIntoOp.getKernelLocal().getDefiningOp());
            }
         } else {
            return mlir::failure();
         }
      }
      auto optimisticHtType = subop::PreAggrHtType::get(getContext(), mapType.getKeyMembers(), mapType.getValueMembers(), false);
      auto optimisticHtFragmentType = subop::PreAggrHtFragmentType::get(getContext(), mapType.getKeyMembers(), mapType.getValueMembers(), false);
      mlir::TypeConverter fragmentTypeConverter;
      fragmentTypeConverter.addConversion([&](subop::LookupEntryRefType lookupRefType) {
         return subop::LookupEntryRefType::get(lookupRefType.getContext(), mlir::cast<subop::LookupAbleState>(fragmentTypeConverter.convertType(lookupRefType.getState())));
      });
      fragmentTypeConverter.addConversion([&](subop::HashMapType mapType) {
         return optimisticHtFragmentType;
      });
      mlir::TypeConverter htTypeConverter;
      htTypeConverter.addConversion([&](subop::HashMapType mapType) {
         return optimisticHtType;
      });
      htTypeConverter.addConversion([&](subop::HashMapEntryRefType refType) {
         return subop::PreAggrHTEntryRefType::get(refType.getContext(), optimisticHtType);
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

      for (auto usage : stepLocalUsesOfMap) {
         // *Usages* of SSA values: region argument and kernel local which are assumed to be hashmap before this.
         fragmentTransformer.updateValue(usage, optimisticHtFragmentType);
         usage.setType(optimisticHtFragmentType);
      }
      auto loc = creationOp->getLoc();
      // State creation before GPU step: create optHtFrag instead of hashmap
      subop::ExecutionStepOp optHtFragCreationOp;
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointAfter(creationOp->getParentOfType<subop::ExecutionStepOp>());
         llvm::SmallVector<bool> threadLocals(1, false);
         optHtFragCreationOp = rewriter.create<subop::ExecutionStepOp>(loc, mlir::TypeRange{optimisticHtFragmentType}, mlir::ValueRange{}, rewriter.getBoolArrayAttr({}), subop::DeviceType::CPU);
         auto* block = new mlir::Block;
         optHtFragCreationOp.getSubOps().push_back(block);
         rewriter.setInsertionPointToStart(block);
         mlir::Value cpuOptHtFrag = rewriter.create<subop::GenericCreateOp>(loc, optimisticHtFragmentType);
         mlir::Value optHtFrag = rewriter.create<subop::StateContextSwitchOp>(loc, mlir::TypeRange{optimisticHtFragmentType}, cpuOptHtFrag, subop::DataMovementDirection::toDevice);
         rewriter.create<subop::ExecutionStepReturnOp>(loc, mlir::ValueRange{optHtFrag});
      }
      rewriter.replaceOp(creationOp->getParentOfType<subop::ExecutionStepOp>(), optHtFragCreationOp);

      // Update optHtFrag usages as step region argument type (hashmap -> optHTFrag) and its kernel local creation : affects all usages inside step region.
      auto kernelLocalCreationOp = stepRegionArgToKernelLocals.second;
      stepRegionArgToKernelLocals.first.setType(optimisticHtFragmentType);
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointAfter(kernelLocalCreationOp);
         auto kernelLocalType = subop::KernelLocalType::get(rewriter.getContext(), mlir::cast<subop::State>(optimisticHtFragmentType));
         mlir::Value newKernelLocal = rewriter.create<subop::CreateKernelLocalOp>(kernelLocalCreationOp->getLoc(), kernelLocalType, kernelLocalCreationOp.getLocality());
         auto* block = new mlir::Block;
         mlir::cast<subop::CreateKernelLocalOp>(newKernelLocal.getDefiningOp()).getInitFn().push_back(block);
         rewriter.setInsertionPointToStart(block);
         rewriter.create<tuples::ReturnOp>(loc, rewriter.create<subop::GenericCreateOp>(loc, optimisticHtFragmentType).getRes());
         rewriter.replaceOp(kernelLocalCreationOp, newKernelLocal);
         // insert_or_update now use new kernel local, merge_into merges new kernel local to regionArg.setType().
         // the step argument itself is updated via the above the loop transformation
      }

      // We need to make a new execution step after the merge that will allocate enough memory for ht partitions directories according to the sizes of fragments partitions.
      subop::ExecutionStepOp createOptHTAndMergeFrag;
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointAfter(lastInserterStep);
         createOptHTAndMergeFrag = rewriter.create<subop::ExecutionStepOp>(loc, mlir::TypeRange{optimisticHtType}, mlir::ValueRange{optHtFragCreationOp.getResult(0)}, rewriter.getBoolArrayAttr({false}), subop::DeviceType::GPU);
         auto* block = new mlir::Block;
         createOptHTAndMergeFrag.getSubOps().push_back(block);
         rewriter.setInsertionPointToStart(block);
         mlir::Value fragArg = block->addArgument(optimisticHtFragmentType, loc);
         // Merge logic is implemented in runtime, fragArg is already on GPU in its reduced form
         subop::MergeOneToOneOp mergeFragToOptHT = rewriter.create<subop::MergeOneToOneOp>(loc, optimisticHtType, fragArg);
         if (!mergeIntoOp.getCombineFn().empty()) {
            mlir::IRMapping mapping;
            mergeIntoOp.getCombineFn().cloneInto(&mergeFragToOptHT.getCombineFn(), mapping);
         }
         if (!mergeIntoOp.getEqFn().empty()) {
            mlir::IRMapping mapping;
            mergeIntoOp.getEqFn().cloneInto(&mergeFragToOptHT.getEqFn(), mapping);
         }
         rewriter.create<subop::ExecutionStepReturnOp>(loc, mlir::ValueRange{mergeFragToOptHT->getResult(0)});
         // Previous users of "merged fragment" should now refer to the optHT.
         for (auto& use : optHtFragCreationOp.getResult(0).getUses()) {
            if (use.getOwner()->isBeforeInBlock(createOptHTAndMergeFrag) || use.getOwner() == createOptHTAndMergeFrag) {
               continue;
            }
            if (auto userExecutionStep = mlir::dyn_cast_or_null<subop::ExecutionStepOp>(use.getOwner())) {
               userExecutionStep.updateStateType(htTransformer, optHtFragCreationOp.getResult(0), optimisticHtType);
            }
            use.set(createOptHTAndMergeFrag.getResult(0));
         }
      }
      return mlir::success();
   }
};
class SpecializeGPUPassSubOpPass : public mlir::PassWrapper<SpecializeGPUPassSubOpPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SpecializeGPUPassSubOpPass)
   virtual llvm::StringRef getArgument() const override { return "subop-specialize-gpu"; }

   SpecializeGPUPassSubOpPass() {}
   void getDependentDialects(mlir::DialectRegistry& registry) const override {
      registry.insert<util::UtilDialect, db::DBDialect>();
   }
   void runOnOperation() override {
      //transform "standalone" aggregation functions
      auto columnUsageAnalysis = getAnalysis<subop::ColumnUsageAnalysis>();

      mlir::RewritePatternSet patterns(&getContext());
      patterns.insert<IntroducePreAggrHt>(&getContext(), columnUsageAnalysis);
      if (mlir::applyPatternsAndFoldGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
         assert(false && "should not happen");
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
subop::createSpecializeGPUPass() { return std::make_unique<SpecializeGPUPassSubOpPass>(); }