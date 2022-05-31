#include "mlir/Transforms/CustomPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <queue>
namespace {
class SinkOp : public mlir::PassWrapper<SinkOp, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "sinkop"; }
   static bool isSideEffectFree(mlir::Operation* op) {
      if (auto memInterface = dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
         // If the op has side-effects, it cannot be moved.
         if (!memInterface.hasNoEffect())
            return false;
         // If the op does not have recursive side effects, then it can be moved.
         if (!op->hasTrait<mlir::OpTrait::HasRecursiveSideEffects>())
            return true;
      } else if (!op->hasTrait<mlir::OpTrait::HasRecursiveSideEffects>()) {
         // Otherwise, if the op does not implement the memory effect interface and
         // it does not have recursive side effects, then it cannot be known that the
         // op is moveable.
         return false;
      }

      // Recurse into the regions and ensure that all nested ops can also be moved.
      for (mlir::Region& region : op->getRegions())
         for (mlir::Operation& op : region.getOps())
            if (!isSideEffectFree(&op))
               return false;
      return true;
   }
   mlir::Block* canSink(mlir::Operation* op) {
      auto users = op->getUsers();
      if (users.empty()) return nullptr;
      auto *firstUser = *users.begin();
      if (firstUser->getParentRegion() != op->getParentRegion()) return nullptr;

      if (op->getBlock()==firstUser->getBlock()) return nullptr;
      for(auto *user:users){
         if(user->getBlock()!=firstUser->getBlock()){
            return nullptr;
         }
      }
      if (!isSideEffectFree(op)) return nullptr;
      return firstUser->getBlock();
   }

   public:
   void runOnOperation() override {
      std::queue<mlir::Operation*> q;
      //todo: better performance
      getOperation().walk([&](mlir::Operation* op) {
         if (canSink(op))
            q.push(op);
      });
      while (!q.empty()) {
         mlir::Operation* curr = q.front();
         q.pop();
         auto *sinkBefore = canSink(curr);
         if (!sinkBefore) continue;
         curr->moveBefore(sinkBefore,sinkBefore->begin());
         for (auto operand : curr->getOperands()) {
            if (auto *definingOp = operand.getDefiningOp()) {
               if (canSink(definingOp)) {
                  q.push(definingOp);
               }
            }
         }
      }
   }
};
} // end anonymous namespace
std::unique_ptr<mlir::Pass> mlir::createSinkOpPass() {
   return std::make_unique<SinkOp>();
}