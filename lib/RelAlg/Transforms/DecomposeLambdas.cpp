#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
class DecomposeLambdas : public mlir::PassWrapper<DecomposeLambdas, mlir::FunctionPass> {
   virtual llvm::StringRef getArgument() const override { return "relalg-decompose-lambdas"; }

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
         auto newsel = builder.create<relalg::SelectionOp>(currentSel->getLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), tree);
         tree = newsel;
         newsel.initPredicate();
         mapping.map(currentSel.getPredicateArgument(), newsel.getPredicateArgument());
         builder.setInsertionPointToStart(&newsel.predicate().front());
         mlir::relalg::detail::inlineOpIntoBlock(v.getDefiningOp(), v.getDefiningOp()->getParentOp(), newsel.getOperation(), &newsel.getPredicateBlock(), mapping);
         builder.create<mlir::relalg::ReturnOp>(currentSel->getLoc(), mapping.lookup(v));
         auto* terminator = newsel.getLambdaBlock().getTerminator();
         terminator->remove();
         terminator->destroy();
      }
   }
   static llvm::DenseMap<mlir::Value, mlir::relalg::Attributes> analyze(mlir::Block* block, mlir::relalg::Attributes availableLeft, mlir::relalg::Attributes availableRight) {
      llvm::DenseMap<mlir::Value, mlir::relalg::Attributes> required;
      mlir::relalg::Attributes leftKeys, rightKeys;
      std::vector<mlir::Type> types;
      block->walk([&](mlir::Operation* op) {
         if (auto getAttr = mlir::dyn_cast_or_null<mlir::relalg::GetAttrOp>(op)) {
            required.insert({getAttr.getResult(), mlir::relalg::Attributes::from(getAttr.attr())});
         } else {
            mlir::relalg::Attributes attributes;
            for (auto operand : op->getOperands()) {
               if (required.count(operand)) {
                  attributes.insert(required[operand]);
               }
            }
            for (auto result : op->getResults()) {
               required.insert({result, attributes});
            }
         }
      });
      return required;
   }
   mlir::Value decomposeOuterJoin(mlir::Value v, mlir::relalg::Attributes availableLeft, mlir::relalg::Attributes availableRight, llvm::DenseMap<mlir::Value, mlir::relalg::Attributes> required) {
      auto currentJoinOp = mlir::dyn_cast_or_null<mlir::relalg::OuterJoinOp>(v.getDefiningOp()->getParentOp());
      using namespace mlir;
      if (auto andop = dyn_cast_or_null<mlir::db::AndOp>(v.getDefiningOp())) {
         std::vector<Value> vals;
         for (auto operand : andop.vals()) {
            auto val=decomposeOuterJoin(operand, availableLeft,availableRight,required);
            if(val){
               vals.push_back(val);
            }
         }
         OpBuilder builder(andop);
         auto newAndOp=builder.create<mlir::db::AndOp>(andop->getLoc(),vals);
         andop->remove();
         andop->dropAllReferences();
         //andop->destroy();
         return newAndOp;
      } else {
         if (required[v].isSubsetOf(availableRight)) {
            auto children = currentJoinOp.getChildren();
            OpBuilder builder(currentJoinOp);
            mlir::BlockAndValueMapping mapping;
            auto newsel = builder.create<relalg::SelectionOp>(currentJoinOp->getLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), children[1].asRelation());
            newsel.initPredicate();
            mapping.map(currentJoinOp.getPredicateArgument(), newsel.getPredicateArgument());
            builder.setInsertionPointToStart(&newsel.predicate().front());
            mlir::relalg::detail::inlineOpIntoBlock(v.getDefiningOp(), v.getDefiningOp()->getParentOp(), newsel.getOperation(), &newsel.getPredicateBlock(), mapping);
            builder.create<mlir::relalg::ReturnOp>(currentJoinOp->getLoc(), mapping.lookup(v));
            auto* terminator = newsel.getLambdaBlock().getTerminator();
            terminator->remove();
            terminator->destroy();
            currentJoinOp.setChildren({children[0], newsel});
            return Value();
         }
         return v;
      }
   }
   void decomposeMap(mlir::relalg::MapOp currentMap, mlir::Value& tree) {
      using namespace mlir;

      auto *terminator=currentMap.predicate().front().getTerminator();
      if(auto returnOp=mlir::dyn_cast_or_null<mlir::relalg::ReturnOp>(terminator)){
         assert(returnOp.results().size()==1);
         auto *definingOp=returnOp.results()[0].getDefiningOp();
         mlir::relalg::AddAttrOp addAttrOp=definingOp?mlir::dyn_cast_or_null<mlir::relalg::AddAttrOp>(definingOp):relalg::AddAttrOp();
         while(mlir::isa_and_nonnull<mlir::relalg::AddAttrOp>(addAttrOp)){
            OpBuilder builder(currentMap);
            mlir::BlockAndValueMapping mapping;
            auto newmap = builder.create<relalg::MapOp>(currentMap->getLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), currentMap.sym_name(), tree);
            tree = newmap;
            newmap.predicate().push_back(new Block);
            newmap.predicate().addArgument(mlir::relalg::TupleType::get(builder.getContext()));
            builder.setInsertionPointToStart(&newmap.predicate().front());
            auto ret1=builder.create<relalg::ReturnOp>(currentMap->getLoc());
            mapping.map(currentMap.getLambdaArgument(), newmap.getLambdaArgument());
            mapping.map(addAttrOp.tuple(), newmap.getLambdaArgument());
            mlir::relalg::detail::inlineOpIntoBlock(addAttrOp.getOperation(), addAttrOp->getParentOp(), newmap.getOperation(), &newmap.getLambdaBlock(), mapping);
            builder.create<relalg::ReturnOp>(currentMap->getLoc(),mapping.lookup(addAttrOp.tuple_out()));
            ret1->remove();
            ret1->dropAllReferences();
            ret1->destroy();
            //go to next in chain:
            definingOp=addAttrOp.tuple().getDefiningOp();
            addAttrOp=definingOp?mlir::dyn_cast_or_null<mlir::relalg::AddAttrOp>(definingOp):relalg::AddAttrOp();
         }
      }
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
      getFunction().walk([&](mlir::relalg::OuterJoinOp op) {
        auto* terminator = op.getRegion().front().getTerminator();
        auto retval = terminator->getOperand(0);
         auto availableLeft = op.getChildren()[0].getAvailableAttributes();
         auto availableRight = op.getChildren()[1].getAvailableAttributes();
         auto mapped=analyze(&op.getPredicateBlock(),availableLeft,availableRight);
         auto val=decomposeOuterJoin(retval,availableLeft,availableRight,mapped);
         mlir::OpBuilder builder(terminator);
         builder.create<mlir::relalg::ReturnOp>(terminator->getLoc(),val?mlir::ValueRange{val}:mlir::ValueRange{});
         terminator->remove();
         terminator->destroy();

      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createDecomposeLambdasPass() { return std::make_unique<DecomposeLambdas>(); }
} // end namespace relalg
} // end namespace mlir