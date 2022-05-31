#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <unordered_set>

namespace {
class DecomposeLambdas : public mlir::PassWrapper<DecomposeLambdas, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-decompose-lambdas"; }

   public:
   bool checkRestriction(std::string& str, mlir::Value v) {
      auto *op = v.getDefiningOp();
      if (!op) return true;
      if (auto refOp = mlir::dyn_cast_or_null<mlir::relalg::GetColumnOp>(op)) {
         auto scope = refOp.attr().getName().getRootReference().str();
         if (str.empty() || str == scope) {
            str = scope;
            return true;
         } else {
            return false;
         }
      }
      for (auto operand : op->getOperands()) {
         if (!checkRestriction(str, operand)) {
            return false;
         }
      }
      return true;
   }
   std::unordered_map<std::string, std::vector<mlir::Value>> deriveRestrictionsFromOrAnd(mlir::db::AndOp andOp) {
      std::unordered_map<std::string, std::vector<mlir::Value>> restrictions;
      for (auto operand : andOp->getOperands()) {
         std::string scope = "";
         if (checkRestriction(scope, operand)) {
            restrictions[scope].push_back(operand);
         }
      }
      return restrictions;
   }

   void deriveRestrictionsFromOr(mlir::db::OrOp orOp, mlir::Value& tree) {
      auto currentSel = mlir::dyn_cast_or_null<mlir::relalg::SelectionOp>(orOp->getParentOp());

      std::vector<std::unordered_map<std::string, std::vector<mlir::Value>>> restrictions;
      std::unordered_set<std::string> availableScopes;
      for (auto v : orOp.vals()) {
         if (auto andOp = dyn_cast_or_null<mlir::db::AndOp>(v.getDefiningOp())) {
            restrictions.push_back(deriveRestrictionsFromOrAnd(andOp));
            for (const auto& p : restrictions[restrictions.size() - 1]) {
               availableScopes.insert(p.first);
            }
         }else{
            return;
         }
      }
      for (auto scope : availableScopes) {
         bool availableInAll = true;
         for (auto& m : restrictions) {
            availableInAll &= m.contains(scope);
         }
         if (availableInAll) {
            mlir::OpBuilder builder(currentSel);
            mlir::BlockAndValueMapping mapping;
            auto newsel = builder.create<mlir::relalg::SelectionOp>(currentSel->getLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), tree);
            tree = newsel;
            newsel.initPredicate();
            mapping.map(currentSel.getPredicateArgument(), newsel.getPredicateArgument());
            builder.setInsertionPointToStart(&newsel.predicate().front());
            auto* terminator = newsel.getLambdaBlock().getTerminator();

            std::vector<mlir::Value> c2;
            for (auto& m : restrictions) {
               std::vector<mlir::Value> c1;
               for (auto v : m[scope]) {
                  mlir::relalg::detail::inlineOpIntoBlock(v.getDefiningOp(), v.getDefiningOp()->getParentOp(), newsel.getOperation(), &newsel.getPredicateBlock(), mapping,terminator);
                  c1.push_back(mapping.lookup(v));
               }
               if (c1.size() == 1) {
                  c2.push_back(c1[0]);
               } else {
                  c2.push_back(builder.create<mlir::db::AndOp>(orOp->getLoc(), c1));
               }
            }
            mlir::Value ored = builder.create<mlir::db::OrOp>(orOp->getLoc(), c2);
            builder.create<mlir::relalg::ReturnOp>(currentSel->getLoc(), ored);
            terminator->erase();
         }
      }
   }
   void decomposeSelection(mlir::Value v, mlir::Value& tree) {
      auto currentSel = mlir::dyn_cast_or_null<mlir::relalg::SelectionOp>(v.getDefiningOp()->getParentOp());
      using namespace mlir;
      if (auto andop = dyn_cast_or_null<mlir::db::AndOp>(v.getDefiningOp())) {
         for (auto operand : andop.vals()) {
            decomposeSelection(operand, tree);
         }
      } else {
         if (auto orOp = dyn_cast_or_null<mlir::db::OrOp>(v.getDefiningOp())) {
            //todo: fix potential dominator problem...
            deriveRestrictionsFromOr(orOp,tree);
         }
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
         terminator->erase();
      }
   }
   static llvm::DenseMap<mlir::Value, mlir::relalg::ColumnSet> analyze(mlir::Block* block, mlir::relalg::ColumnSet availableLeft, mlir::relalg::ColumnSet availableRight) {
      llvm::DenseMap<mlir::Value, mlir::relalg::ColumnSet> required;
      mlir::relalg::ColumnSet leftKeys, rightKeys;
      std::vector<mlir::Type> types;
      block->walk([&](mlir::Operation* op) {
         if (auto getAttr = mlir::dyn_cast_or_null<mlir::relalg::GetColumnOp>(op)) {
            required.insert({getAttr.getResult(), mlir::relalg::ColumnSet::from(getAttr.attr())});
         } else {
            mlir::relalg::ColumnSet attributes;
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
   mlir::Value decomposeOuterJoin(mlir::Value v, mlir::relalg::ColumnSet availableLeft, mlir::relalg::ColumnSet availableRight, llvm::DenseMap<mlir::Value, mlir::relalg::ColumnSet> required) {
      auto currentJoinOp = mlir::dyn_cast_or_null<mlir::relalg::OuterJoinOp>(v.getDefiningOp()->getParentOp());
      using namespace mlir;
      if (auto andop = dyn_cast_or_null<mlir::db::AndOp>(v.getDefiningOp())) {
         std::vector<Value> vals;
         for (auto operand : andop.vals()) {
            auto val = decomposeOuterJoin(operand, availableLeft, availableRight, required);
            if (val) {
               vals.push_back(val);
            }
         }
         OpBuilder builder(andop);
         auto newAndOp = builder.create<mlir::db::AndOp>(andop->getLoc(), vals);
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

      auto* terminator = currentMap.predicate().front().getTerminator();
      if (auto returnOp = mlir::dyn_cast_or_null<mlir::relalg::ReturnOp>(terminator)) {
         assert(returnOp.results().size() == currentMap.computed_cols().size());
         auto computedValRange = returnOp.results();
         for(size_t i=0;i<computedValRange.size();i++){
            OpBuilder builder(currentMap);
            mlir::BlockAndValueMapping mapping;
            auto currentAttr=currentMap.computed_cols()[i].cast<mlir::relalg::ColumnDefAttr>();
            mlir::Value currentVal=computedValRange[i];
            auto newmap = builder.create<relalg::MapOp>(currentMap->getLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), tree,builder.getArrayAttr({currentAttr}));
            tree = newmap;
            newmap.predicate().push_back(new Block);
            newmap.predicate().addArgument(mlir::relalg::TupleType::get(builder.getContext()),currentMap->getLoc());
            builder.setInsertionPointToStart(&newmap.predicate().front());
            auto ret1 = builder.create<relalg::ReturnOp>(currentMap->getLoc());
            mapping.map(currentMap.getLambdaArgument(), newmap.getLambdaArgument());
            mlir::relalg::detail::inlineOpIntoBlock(currentVal.getDefiningOp(), currentVal.getDefiningOp()->getParentOp(), newmap.getOperation(), &newmap.getLambdaBlock(), mapping);
            builder.create<relalg::ReturnOp>(currentMap->getLoc(), mapping.lookup(currentVal));
            ret1->remove();
            ret1->dropAllReferences();
            ret1->destroy();
         }
      }
   }
   void runOnOperation() override {
      getOperation().walk([&](mlir::relalg::SelectionOp op) {
         auto* terminator = op.getRegion().front().getTerminator();
         mlir::Value val = op.rel();
         decomposeSelection(terminator->getOperand(0), val);
         op.replaceAllUsesWith(val);
         op->erase();
      });
      getOperation().walk([&](mlir::relalg::MapOp op) {
         mlir::Value val = op.rel();
         decomposeMap(op, val);
         op.replaceAllUsesWith(val);
         op->erase();
      });
      getOperation().walk([&](mlir::relalg::OuterJoinOp op) {
         auto* terminator = op.getRegion().front().getTerminator();
         auto retval = terminator->getOperand(0);
         auto availableLeft = op.getChildren()[0].getAvailableColumns();
         auto availableRight = op.getChildren()[1].getAvailableColumns();
         auto mapped = analyze(&op.getPredicateBlock(), availableLeft, availableRight);
         auto val = decomposeOuterJoin(retval, availableLeft, availableRight, mapped);
         mlir::OpBuilder builder(terminator);
         builder.create<mlir::relalg::ReturnOp>(terminator->getLoc(), val ? mlir::ValueRange{val} : mlir::ValueRange{});
         terminator->erase();
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createDecomposeLambdasPass() { return std::make_unique<DecomposeLambdas>(); }
} // end namespace relalg
} // end namespace mlir