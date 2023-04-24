#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SubOperator/SubOperatorOps.h"
#include "mlir/Dialect/SubOperator/Transforms/ColumnUsageAnalysis.h"
#include "mlir/Dialect/SubOperator/Transforms/Passes.h"
#include "mlir/Dialect/SubOperator/Transforms/SubOpDependencyAnalysis.h"
#include "mlir/Dialect/TupleStream/TupleStreamDialect.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace {

class ParallelizePass : public mlir::PassWrapper<ParallelizePass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ParallelizePass)
   virtual llvm::StringRef getArgument() const override { return "subop-parallelize"; }
   void collectPipelineOperations(std::vector<mlir::Operation*>& ops, mlir::Operation* op, bool start = false) {
      if (op->getDialect()->getNamespace() == "subop" && !start) {
         ops.push_back(op);
      }
      for (auto res : op->getResults()) {
         if (res.getType().isa<mlir::tuples::TupleStreamType>()) {
            for (auto* user : res.getUsers()) {
               collectPipelineOperations(ops, user);
            }
         }
      }
      op->walk([&](mlir::Operation* nested) {
         if (nested != op) {
            auto isStreamType = [](mlir::Type t) { return t.isa<mlir::tuples::TupleStreamType>(); };
            if (llvm::none_of(nested->getOperandTypes(), isStreamType) && llvm::any_of(nested->getResultTypes(), isStreamType)) {
               collectPipelineOperations(ops, nested);
            }
         }
      });
   }

   void runOnOperation() override {
      std::unordered_map<mlir::Operation*, std::vector<mlir::Operation*>> toThreadLocals;

      getOperation()->walk([&](mlir::Operation* op) {
         if (auto scanRefsOp = mlir::dyn_cast_or_null<mlir::subop::ScanRefsOp>(op)) {
            if (auto funcOp = mlir::dyn_cast_or_null<mlir::func::FuncOp>(scanRefsOp->getParentOp())) {
               std::vector<mlir::Operation*> pipelineOps;
               collectPipelineOperations(pipelineOps, op, true);
               bool canBeParallel = true;
               auto isStreamTypeOrNested = [&](mlir::Value v) {
                  if (v.getType().isa<mlir::tuples::TupleStreamType>()) return true;
                  if (auto* def = v.getDefiningOp()) {
                     if (def->getParentOp() != funcOp.getOperation() && funcOp->isAncestor(def)) {
                        return true;
                     }
                  }
                  return false;
               };

               for (auto* pipelineOp : pipelineOps) {
                  if (llvm::all_of(pipelineOp->getOperands(), isStreamTypeOrNested)) {
                     //ignore: do not interact with states
                     continue;
                  } else if (auto materializeOp = mlir::dyn_cast_or_null<mlir::subop::MaterializeOp>(pipelineOp)) {
                     if (auto* createOp = materializeOp.getState().getDefiningOp()) {
                        toThreadLocals[createOp].push_back(materializeOp);
                        continue;
                     }
                  }
                  canBeParallel = false;
                  pipelineOp->dump();
               }
               //finally: mark as parallel
               if (canBeParallel) {
                  scanRefsOp->setAttr("parallel", mlir::UnitAttr::get(&getContext()));
               }
            }
         }
      });
      for (auto toThreadLocal : toThreadLocals) {
         auto* createOp = toThreadLocal.first;
         mlir::OpBuilder builder(&getContext());
         builder.setInsertionPoint(createOp);
         auto threadLocalType = mlir::subop::ThreadLocalType::get(builder.getContext(), createOp->getResultTypes()[0]);
         auto createThreadLocal = builder.create<mlir::subop::CreateThreadLocalOp>(createOp->getLoc(), threadLocalType);
         auto* block = new mlir::Block;
         createThreadLocal.getInitFn().push_back(block);
         builder.setInsertionPointToStart(block);
         builder.create<mlir::tuples::ReturnOp>(createOp->getLoc(), builder.clone(*createOp)->getResult(0));
         std::unordered_set<mlir::Operation*> localUsers(toThreadLocal.second.begin(), toThreadLocal.second.end());
         std::vector<mlir::Operation*> otherUsers;
         for (auto* user : createOp->getUsers()) {
            if (!localUsers.contains(user)) {
               otherUsers.push_back(user); //todo fix behavior for nested
            }
         }
         std::sort(otherUsers.begin(), otherUsers.end(), [&](mlir::Operation* left, mlir::Operation* right) {
            return left->isBeforeInBlock(right);
         });
         assert(otherUsers.size() > 0);
         builder.setInsertionPoint(otherUsers[0]);
         mlir::Value merged = builder.create<mlir::subop::MergeOp>(createOp->getLoc(), createOp->getResultTypes()[0], createThreadLocal.getResult());
         for (auto localUser : localUsers) {
            builder.setInsertionPoint(localUser);
            mlir::Value local = builder.create<mlir::subop::GetLocal>(createOp->getLoc(), createOp->getResultTypes()[0], createThreadLocal.getResult());
            localUser->replaceUsesOfWith(createOp->getResult(0), local);
         }
         createOp->replaceAllUsesWith(mlir::ValueRange{merged});
         createOp->erase();
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
mlir::subop::createParallelizePass() { return std::make_unique<ParallelizePass>(); }