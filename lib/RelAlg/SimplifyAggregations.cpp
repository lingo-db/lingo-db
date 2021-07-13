#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
class SimplifyAggregations : public mlir::PassWrapper<SimplifyAggregations, mlir::FunctionPass> {
   public:

   void runOnFunction() override {
      getFunction().walk([&](mlir::relalg::AggregationOp aggregationOp) {
         mlir::Value arg=aggregationOp.aggr_func().front().getArgument(0);
         std::vector<mlir::Operation*> users(arg.getUsers().begin(),arg.getUsers().end());
         if(users.size()==1){
            if(auto projectionOp=mlir::dyn_cast_or_null<mlir::relalg::ProjectionOp>(users[0])){
               if(projectionOp.set_semantic()==mlir::relalg::SetSemantic::distinct){
                  mlir::OpBuilder builder(aggregationOp);
                  auto attrs=mlir::relalg::Attributes::fromArrayAttr(aggregationOp.group_by_attrs());
                  attrs.insert(mlir::relalg::Attributes::fromArrayAttr(projectionOp.attrs()));
                  auto newProj = builder.create<mlir::relalg::ProjectionOp>(builder.getUnknownLoc(), mlir::relalg::TupleStreamType::get(&getContext()), mlir::relalg::SetSemantic::distinct, aggregationOp.rel(), attrs.asRefArrayAttr(&getContext()));
                  aggregationOp.setOperand(newProj);
                  projectionOp.replaceAllUsesWith(arg);
                  projectionOp->remove();
                  projectionOp->dropAllUses();
                  projectionOp->dropAllReferences();
                  projectionOp->destroy();
               }
            }
         }
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createSimplifyAggregationsPass() { return std::make_unique<SimplifyAggregations>(); }
} // end namespace relalg
} // end namespace mlir