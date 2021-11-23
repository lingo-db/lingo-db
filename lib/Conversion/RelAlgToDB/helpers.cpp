#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
using namespace mlir::relalg;
void mlir::relalg::mergeRelatinalBlock(mlir::Block* dest,mlir::Block* source, LoweringContext& context, LoweringContext::AttributeResolverScope& scope) {

   // Splice the operations of the 'source' block into the 'dest' block and erase
   // it.
   llvm::iplist<mlir::Operation> translated;
   std::vector<mlir::Operation*> toErase;
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
}