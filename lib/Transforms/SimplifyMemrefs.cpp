#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <iostream>

#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
class FoldLoadGlobal : public mlir::RewritePattern {
   public:
   FoldLoadGlobal(mlir::MLIRContext* context)
      : RewritePattern(mlir::memref::LoadOp::getOperationName(), 1, context) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      mlir::memref::LoadOp loadOp = mlir::cast<mlir::memref::LoadOp>(op);
      std::vector<size_t> indices;
      for (auto i : loadOp.getIndices()) {
         if (auto *idxDefiningOp = i.getDefiningOp()) {
            if (auto constantOp = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(idxDefiningOp)) {
               if (auto integerAttr = constantOp.getValue().dyn_cast_or_null<mlir::IntegerAttr>()) {
                  indices.push_back(integerAttr.getInt());
                  continue;
               }
            }
         }
         return mlir::failure();
      }
      if (auto *memrefDefiningOp = loadOp.memref().getDefiningOp()) {
         if (auto getGlobalOp = mlir::dyn_cast_or_null<mlir::memref::GetGlobalOp>(memrefDefiningOp)) {
            if (auto *symbolTableOp = mlir::SymbolTable::getNearestSymbolTable(op)) {
               auto *resolvedOp = mlir::SymbolTable::lookupSymbolIn(symbolTableOp, getGlobalOp.name());
               if (auto globalOp = mlir::dyn_cast_or_null<mlir::memref::GlobalOp>(resolvedOp)) {
                  if (!globalOp.constant()) return mlir::failure();
                  if (globalOp.isExternal()) return mlir::failure();
                  if (!globalOp.initial_value().hasValue()) return mlir::failure();
                  auto initialValue = globalOp.initial_value().getValue();
                  if (auto denseAttr = initialValue.dyn_cast_or_null<mlir::DenseIntOrFPElementsAttr>()) {
                     auto it = denseAttr.getValues<float>();
                     auto res = it[indices[0]];
                     std::cout << res << std::endl;
                     mlir::Value resConstant = rewriter.create<mlir::arith::ConstantOp>(loadOp->getLoc(), loadOp.getType(), rewriter.getF32FloatAttr(res));
                     rewriter.replaceOp(loadOp, resConstant);
                     return mlir::success();
                  }
               }
            }
         }
      }
      return mlir::failure();
   }
};
class FoldLocalLoadStores : public mlir::RewritePattern {
   public:
   FoldLocalLoadStores(mlir::MLIRContext* context)
      : RewritePattern(mlir::memref::LoadOp::getOperationName(), 1, context) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      mlir::memref::LoadOp loadOp = mlir::cast<mlir::memref::LoadOp>(op);
      if (auto *memrefDefiningOp = loadOp.memref().getDefiningOp()) {
         std::vector<mlir::Operation*> sameBlockUsers;
         for (auto *u : memrefDefiningOp->getUsers()) {
            if (u->getBlock() == loadOp->getBlock() && u->isBeforeInBlock(loadOp) && !mlir::isa<mlir::memref::LoadOp>(u)) {
               sameBlockUsers.push_back(u);
            }
         }
         std::sort(sameBlockUsers.begin(), sameBlockUsers.end(), [](mlir::Operation* a, mlir::Operation* b) { return a->isBeforeInBlock(b); });
         auto *lastUser = sameBlockUsers.back();
         if (auto storeOp = mlir::dyn_cast_or_null<mlir::memref::StoreOp>(lastUser)) {
            if (storeOp.indices() == loadOp.indices()) {
               rewriter.replaceOp(op, storeOp.value());
               return mlir::success();
            }
         }
      }
      return mlir::failure();
   }
};
class SimplifyMemrefs : public mlir::PassWrapper<SimplifyMemrefs, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "simplify-memrefs"; }

   public:
   void runOnOperation() override {
      //transform "standalone" aggregation functions
      {
         mlir::RewritePatternSet patterns(&getContext());
         patterns.insert<FoldLoadGlobal>(&getContext());
         patterns.insert<FoldLocalLoadStores>(&getContext());

         if (mlir::applyPatternsAndFoldGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
            assert(false && "should not happen");
         }
      }
   }
};
} // end anonymous namespace

namespace mlir {

std::unique_ptr<Pass> createSimplifyMemrefsPass() { return std::make_unique<SimplifyMemrefs>(); }

} // end namespace mlir