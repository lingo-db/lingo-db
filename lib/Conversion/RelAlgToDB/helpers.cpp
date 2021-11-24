#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
using namespace mlir::relalg;
std::vector<mlir::Value> mlir::relalg::mergeRelationalBlock(mlir::Block* dest,mlir::Operation* op,mlir::function_ref<mlir::Block*(mlir::Operation*)> getBlockFn, LoweringContext& context, LoweringContext::AttributeResolverScope& scope) {

   // Splice the operations of the 'source' block into the 'dest' block and erase
   // it.
   llvm::iplist<mlir::Operation> translated;
   std::vector<mlir::Operation*> toErase;
   auto cloned=op->clone();
   mlir::Block* source=getBlockFn(cloned);
   auto* terminator = source->getTerminator();

   source->walk([&](mlir::relalg::GetAttrOp getAttrOp) {
      getAttrOp.replaceAllUsesWith(context.getValueForAttribute(&getAttrOp.attr().getRelationalAttribute()));
      toErase.push_back(getAttrOp.getOperation());
   });
   for (auto addAttrOp : source->getOps<mlir::relalg::AddAttrOp>()) {
      context.setValueForAttribute(scope, &addAttrOp.attr().getRelationalAttribute(), addAttrOp.val());
      toErase.push_back(addAttrOp.getOperation());
   }

   dest->getOperations().splice(dest->end(), source->getOperations());
   for (auto* op : toErase) {
      op->dropAllUses();
      op->erase();
   }
   auto returnOp=mlir::cast<mlir::relalg::ReturnOp>(terminator);
   std::vector<Value> res(returnOp.results().begin(),returnOp.results().end());
   terminator->erase();
    return res;
}