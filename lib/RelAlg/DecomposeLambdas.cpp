
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
class DecomposeLambdas : public mlir::PassWrapper<DecomposeLambdas, mlir::FunctionPass> {
   public:
   void addRequirements(mlir::Operation* op,mlir::Block* b,llvm::SmallVector<mlir::Operation*,8>& extracted, llvm::SmallPtrSet<mlir::Operation*,8>& alreadyPresent){
      if(!op)
         return;
      if(op->getBlock()!=b)
         return;
      if(alreadyPresent.contains(op))
         return;
      for(auto operand:op->getOperands()) {
         addRequirements(operand.getDefiningOp(), b, extracted, alreadyPresent);
      }
      alreadyPresent.insert(op);
      extracted.push_back(op);
   }
   void decomposeSelection(mlir::Value v,mlir::Value& tree ){
      auto currentSel=mlir::dyn_cast_or_null<mlir::relalg::SelectionOp>(v.getDefiningOp()->getParentOp());
      using namespace mlir;
      if(auto andop=dyn_cast_or_null<mlir::db::AndOp>(v.getDefiningOp())){
         Value toReturn;
         for(auto operand:andop.vals()){
            decomposeSelection(operand, tree);
         }
      }else{
         llvm::SmallVector<mlir::Operation*,8> extracted;
         llvm::SmallPtrSet<mlir::Operation*,8> alreadyPresent;
         addRequirements(v.getDefiningOp(),v.getDefiningOp()->getBlock(),extracted,alreadyPresent);
         OpBuilder builder(currentSel);
         mlir::BlockAndValueMapping mapping;
         auto newsel=builder.create<relalg::SelectionOp>(builder.getUnknownLoc(),mlir::relalg::RelationType::get(builder.getContext()),tree);
         tree=newsel;
         newsel.predicate().push_back(new Block);
         newsel.predicate().addArgument(mlir::relalg::TupleType::get(builder.getContext()));
         mapping.map(currentSel.getLambdaArgument(),newsel.getLambdaArgument());
         builder.setInsertionPointToStart(&newsel.predicate().front());
         auto returnop=builder.create<relalg::ReturnOp>(builder.getUnknownLoc());
         builder.setInsertionPointToStart(&newsel.predicate().front());
         for(auto *op:extracted){
            auto *cloneOp=builder.clone(*op,mapping);
            cloneOp->moveBefore(returnop);
         }
         builder.create<mlir::relalg::ReturnOp>(builder.getUnknownLoc(),mapping.lookup(v));
         returnop->remove();
         returnop->destroy();

      }
   }
   void decomposeMap(mlir::relalg::MapOp currentMap,mlir::Value& tree ){
      using namespace mlir;
      currentMap->walk([&](mlir::relalg::AddAttrOp addAttrOp){
        llvm::SmallVector<mlir::Operation*,8> extracted;
        llvm::SmallPtrSet<mlir::Operation*,8> alreadyPresent;
        addRequirements(addAttrOp.getOperation(),addAttrOp->getBlock(),extracted,alreadyPresent);
        OpBuilder builder(currentMap);
        mlir::BlockAndValueMapping mapping;
        auto newmap=builder.create<relalg::MapOp>(builder.getUnknownLoc(),mlir::relalg::RelationType::get(builder.getContext()),currentMap.sym_name(),tree);
        tree=newmap;
        newmap.predicate().push_back(new Block);
        newmap.predicate().addArgument(mlir::relalg::TupleType::get(builder.getContext()));
        mapping.map(currentMap.getLambdaArgument(),newmap.getLambdaArgument());
        builder.setInsertionPointToStart(&newmap.predicate().front());
        auto returnop=builder.create<relalg::ReturnOp>(builder.getUnknownLoc());
        builder.setInsertionPointToStart(&newmap.predicate().front());
        for(auto *op:extracted){
           auto *cloneOp=builder.clone(*op,mapping);
           cloneOp->moveBefore(returnop);
        }
      });


   }
   void runOnFunction() override {
      getFunction().walk([&](mlir::relalg::SelectionOp op) {
         auto *terminator=op.getRegion().front().getTerminator();
         auto retval=terminator->getOperand(0);
         mlir::Value val=op.rel();
         decomposeSelection(retval, val);
         op.replaceAllUsesWith(val);
         op->remove();
         op->destroy();
      });
      getFunction().walk([&](mlir::relalg::MapOp op) {
        mlir::Value val=op.rel();
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