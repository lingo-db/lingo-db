
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"

#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <iostream>
#include <list>
#include <queue>
#include <unordered_set>

namespace {
class ImplicitToExplicitJoins : public mlir::PassWrapper<ImplicitToExplicitJoins, mlir::FunctionPass> {
   void runOnFunction() override {
      using namespace  mlir;
      getFunction().walk([&](mlir::Operation* op) {
        TupleLamdaOperator surrounding_operator=op->getParentOfType<TupleLamdaOperator>();
         if(!surrounding_operator){
            return;
         }

         Value tree_val=surrounding_operator->getOperand(0);
        if(auto existsop=mlir::dyn_cast_or_null<mlir::relalg::ExistsOp>(op)){
            OpBuilder builder(surrounding_operator);
           auto relationalAttribute = std::make_shared<mlir::relalg::RelationalAttribute>(db::BoolType::get(builder.getContext()));
           relalg::RelationalAttributeDefAttr defAttr=relalg::RelationalAttributeDefAttr::get(builder.getContext(),"mj1mark",relationalAttribute,Attribute());
           relalg::RelationalAttributeRefAttr refAttr=relalg::RelationalAttributeRefAttr::get(builder.getContext(),builder.getSymbolRefAttr("mj1"),std::shared_ptr<relalg::RelationalAttribute>());

           //::mlir::Type result, ::mlir::StringAttr sym_name, ::mlir::relalg::RelationalAttributeDefAttr markattr, ::mlir::Value left, ::mlir::Value right
            auto mjop=builder.create<relalg::MarkJoinOp>(builder.getUnknownLoc(),mlir::relalg::RelationType::get(builder.getContext()),"mj1",defAttr,tree_val,existsop.rel());
            mjop.getRegion().push_back(new Block);
            builder.setInsertionPointToStart(&mjop.getRegion().front());
            builder.create<relalg::ReturnOp>(builder.getUnknownLoc());
            builder.setInsertionPoint(existsop);
            auto replacement=builder.create<relalg::GetAttrOp>(builder.getUnknownLoc(),db::BoolType::get(builder.getContext()),refAttr,surrounding_operator.getLambdaRegion().getArgument(0));
            existsop->replaceAllUsesWith(replacement);
            existsop->remove();
            existsop->destroy();
            tree_val=mjop;
         }else if(auto getscalarop=mlir::dyn_cast_or_null<mlir::relalg::GetScalarOp>(op)){
           OpBuilder builder(surrounding_operator);

           //::mlir::Type result, ::mlir::StringAttr sym_name, ::mlir::relalg::RelationalAttributeDefAttr markattr, ::mlir::Value left, ::mlir::Value right
           auto mjop=builder.create<relalg::SingleJoinOp>(builder.getUnknownLoc(),mlir::relalg::RelationType::get(builder.getContext()),"mj1",defAttr,tree_val,existsop.rel());
           mjop.getRegion().push_back(new Block);
           builder.setInsertionPointToStart(&mjop.getRegion().front());
           builder.create<relalg::ReturnOp>(builder.getUnknownLoc());
           builder.setInsertionPoint(existsop);
           auto replacement=builder.create<relalg::GetAttrOp>(builder.getUnknownLoc(),db::BoolType::get(builder.getContext()),refAttr,surrounding_operator.getLambdaRegion().getArgument(0));
           existsop->replaceAllUsesWith(replacement);
           existsop->remove();
           existsop->destroy();
           tree_val=mjop;
         }
         surrounding_operator->setOperand(0,tree_val);

      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createImplicitToExplicitJoinsPass() { return std::make_unique<ImplicitToExplicitJoins>(); }
} // end namespace relalg
} // end namespace mlir