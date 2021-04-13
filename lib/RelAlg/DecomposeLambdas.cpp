#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
class DecomposeLambdas : public mlir::PassWrapper<DecomposeLambdas, mlir::FunctionPass> {
   public:
   void decomposeSelection(mlir::Value v, mlir::Value& tree) {
      auto currentSel = mlir::dyn_cast_or_null<mlir::relalg::SelectionOp>(v.getDefiningOp()->getParentOp());
      using namespace mlir;
      if (auto andop = dyn_cast_or_null<mlir::db::AndOp>(v.getDefiningOp())) {
         for (auto operand : andop.vals()) {
            decomposeSelection(operand, tree);
         }
      } else {
         OpBuilder builder(currentSel);
         mlir::BlockAndValueMapping mapping;
         auto newsel = builder.create<relalg::SelectionOp>(builder.getUnknownLoc(), mlir::relalg::RelationType::get(builder.getContext()), tree);
         tree = newsel;
         newsel.initPredicate();
         mapping.map(currentSel.getPredicateArgument(), newsel.getPredicateArgument());
         builder.setInsertionPointToStart(&newsel.predicate().front());
         mlir::relalg::detail::inlineOpIntoBlock(v.getDefiningOp(), v.getDefiningOp()->getParentOp(), newsel.getOperation(), &newsel.getPredicateBlock(), mapping);
         builder.create<mlir::relalg::ReturnOp>(builder.getUnknownLoc(), mapping.lookup(v));
         auto* terminator = newsel.getLambdaBlock().getTerminator();
         terminator->remove();
         terminator->destroy();
      }
   }
   void decomposeMap(mlir::relalg::MapOp currentMap, mlir::Value& tree) {
      using namespace mlir;
      currentMap->walk([&](mlir::relalg::AddAttrOp addAttrOp) {
         OpBuilder builder(currentMap);
         mlir::BlockAndValueMapping mapping;
         auto newmap = builder.create<relalg::MapOp>(builder.getUnknownLoc(), mlir::relalg::RelationType::get(builder.getContext()), currentMap.sym_name(), tree);
         tree = newmap;
         newmap.predicate().push_back(new Block);
         newmap.predicate().addArgument(mlir::relalg::TupleType::get(builder.getContext()));
         builder.setInsertionPointToStart(&newmap.predicate().front());
         builder.create<relalg::ReturnOp>(builder.getUnknownLoc());
         mapping.map(currentMap.getLambdaArgument(), newmap.getLambdaArgument());
         mlir::relalg::detail::inlineOpIntoBlock(addAttrOp.getOperation(), addAttrOp->getParentOp(), newmap.getOperation(), &newmap.getLambdaBlock(), mapping);
      });
   }
   void runOnFunction() override {
      getFunction().walk([&](mlir::relalg::SelectionOp op) {
         auto* terminator = op.getRegion().front().getTerminator();
         auto retval = terminator->getOperand(0);
         mlir::Value val = op.rel();
         decomposeSelection(retval, val);
         op.replaceAllUsesWith(val);
         op->remove();
         op->destroy();
      });
      getFunction().walk([&](mlir::relalg::MapOp op) {
         mlir::Value val = op.rel();
         decomposeMap(op, val);
         op.replaceAllUsesWith(val);
         op->remove();
         op->destroy();
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createDecomposeLambdasPass() { return std::make_unique<DecomposeLambdas>(); }
} // end namespace relalg
} // end namespace mlir