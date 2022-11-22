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

class PullGatherUpPass : public mlir::PassWrapper<PullGatherUpPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PullGatherUpPass)
   virtual llvm::StringRef getArgument() const override { return "subop-pull-gather-up"; }

   void runOnOperation() override {
      //transform "standalone" aggregation functions
      auto columnUsageAnalysis = getAnalysis<mlir::subop::ColumnUsageAnalysis>();

      std::vector<mlir::subop::GatherOp> gatherOps;
      getOperation()->walk([&](mlir::subop::GatherOp gatherOp) {
         gatherOps.push_back(gatherOp);
      });
      for (auto gatherOp : gatherOps) {
         std::vector<mlir::NamedAttribute> remaining = gatherOp.getMapping().getValue();
         gatherOp.getRes().replaceAllUsesWith(gatherOp.getStream());
         mlir::Value currStream = gatherOp.getStream();
         gatherOp->setOperand(0, gatherOp.getResult());
         mlir::Operation* currentParent;
         mlir::Value lastStream;
         while (currStream) {
            auto users = currStream.getUsers();
            if (users.begin() == users.end()) break;
            auto second = users.begin();
            second++;
            if (second != users.end()) break;
            currentParent = *users.begin();
            bool otherStreams = false;
            for (auto v : currentParent->getOperands()) {
               otherStreams |= v != currStream && v.getType().isa<mlir::tuples::TupleStreamType>();
            }
            if (otherStreams) break;
            auto usedColumns = columnUsageAnalysis.getUsedColumns(currentParent);
            std::vector<mlir::NamedAttribute> usedByCurrent;
            std::vector<mlir::NamedAttribute> notUsedByCurrent;
            for (auto x : remaining) {
               if (usedColumns.contains(&x.getValue().cast<mlir::tuples::ColumnDefAttr>().getColumn())) {
                  usedByCurrent.push_back(x);
               } else {
                  notUsedByCurrent.push_back(x);
               }
            }
            if (!usedByCurrent.empty()) {
               mlir::OpBuilder builder(currentParent);
               auto newGatherOp = builder.create<mlir::subop::GatherOp>(gatherOp->getLoc(), currStream, gatherOp.getRef(), builder.getDictionaryAttr(usedByCurrent));
               currStream.replaceAllUsesWith(newGatherOp.getResult());
               newGatherOp->setOperand(0, currStream);
               lastStream = newGatherOp.getResult();
               //newGatherOp->dump();
            } else {
               lastStream = currStream;
            }
            remaining = std::move(notUsedByCurrent);

            currStream = currentParent->getNumResults() == 1 ? currentParent->getResult(0) : mlir::Value();
         }
         if (!currStream && currentParent) {
            if (mlir::isa<mlir::tuples::ReturnOp>(currentParent)) {
               currStream = lastStream;
            }
         }
         if (!remaining.empty() && currStream) {
            mlir::OpBuilder builder(currStream.getContext());
            builder.setInsertionPointAfter(currStream.getDefiningOp());
            if (!currStream.getUsers().empty()) {
               auto newGatherOp = builder.create<mlir::subop::GatherOp>(gatherOp->getLoc(), currStream, gatherOp.getRef(), gatherOp.getMapping());
               currStream.replaceAllUsesWith(newGatherOp.getResult());
               newGatherOp->setOperand(0, currStream);
            }
         }
         gatherOp->dropAllReferences();
         gatherOp->erase();
      }

      //getOperation()->dump();
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
mlir::subop::createPullGatherUpPass() { return std::make_unique<PullGatherUpPass>(); }