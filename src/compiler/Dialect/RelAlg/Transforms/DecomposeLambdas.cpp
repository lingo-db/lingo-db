#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <unordered_set>

namespace {
using namespace lingodb::compiler::dialect;
class DecomposeInnerJoin : public mlir::RewritePattern {
   public:
   DecomposeInnerJoin(mlir::MLIRContext* context)
      : RewritePattern(relalg::InnerJoinOp::getOperationName(), 1, context) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto innerJoin = mlir::cast<relalg::InnerJoinOp>(op);
      auto cp = rewriter.create<relalg::CrossProductOp>(op->getLoc(), innerJoin.getLeft(), innerJoin.getRight());
      auto sel = rewriter.create<relalg::SelectionOp>(op->getLoc(), cp);
      rewriter.inlineRegionBefore(innerJoin.getPredicate(), sel.getPredicate(), sel.getPredicate().end());
      rewriter.replaceOp(op,sel.getResult());
      return mlir::success();
   }
};
class DecomposeLambdas : public mlir::PassWrapper<DecomposeLambdas, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-decompose-lambdas"; }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DecomposeLambdas)

   bool checkRestriction(std::string& str, mlir::Value v) {
      auto* op = v.getDefiningOp();
      if (!op) return true;
      if (auto refOp = mlir::dyn_cast_or_null<tuples::GetColumnOp>(op)) {
         auto scope = refOp.getAttr().getName().getRootReference().str();
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
   std::unordered_map<std::string, std::vector<mlir::Value>> deriveRestrictionsFromOrAnd(db::AndOp andOp) {
      std::unordered_map<std::string, std::vector<mlir::Value>> restrictions;
      for (auto operand : andOp->getOperands()) {
         std::string scope = "";
         if (checkRestriction(scope, operand)) {
            restrictions[scope].push_back(operand);
         }
      }
      return restrictions;
   }

   void deriveRestrictionsFromOr(db::OrOp orOp, mlir::Value& tree) {
      auto currentSel = mlir::dyn_cast_or_null<relalg::SelectionOp>(orOp->getParentOp());

      std::vector<std::unordered_map<std::string, std::vector<mlir::Value>>> restrictions;
      std::unordered_set<std::string> availableScopes;
      for (auto v : orOp.getVals()) {
         if (auto andOp = dyn_cast_or_null<db::AndOp>(v.getDefiningOp())) {
            restrictions.push_back(deriveRestrictionsFromOrAnd(andOp));
            for (const auto& p : restrictions[restrictions.size() - 1]) {
               availableScopes.insert(p.first);
            }
         } else {
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
            mlir::IRMapping mapping;
            auto newsel = builder.create<relalg::SelectionOp>(currentSel->getLoc(), tuples::TupleStreamType::get(builder.getContext()), tree);
            tree = newsel;
            newsel.initPredicate();
            mapping.map(currentSel.getPredicateArgument(), newsel.getPredicateArgument());
            builder.setInsertionPointToStart(&newsel.getPredicate().front());
            auto* terminator = newsel.getLambdaBlock().getTerminator();

            std::vector<mlir::Value> c2;
            for (auto& m : restrictions) {
               std::vector<mlir::Value> c1;
               for (auto v : m[scope]) {
                  relalg::detail::inlineOpIntoBlock(v.getDefiningOp(), v.getDefiningOp()->getParentOp(), &newsel.getPredicateBlock(), mapping, terminator);
                  c1.push_back(mapping.lookup(v));
               }
               if (c1.size() == 1) {
                  c2.push_back(c1[0]);
               } else {
                  c2.push_back(builder.create<db::AndOp>(orOp->getLoc(), c1));
               }
            }
            mlir::Value ored = builder.create<db::OrOp>(orOp->getLoc(), c2);
            builder.create<tuples::ReturnOp>(currentSel->getLoc(), ored);
            terminator->erase();
         }
      }
   }
   void decomposeSelection(mlir::Value v, mlir::Value& tree) {
      auto currentSel = mlir::dyn_cast_or_null<relalg::SelectionOp>(v.getDefiningOp()->getParentOp());
      using namespace mlir;
      if (auto andop = dyn_cast_or_null<db::AndOp>(v.getDefiningOp())) {
         for (auto operand : andop.getVals()) {
            decomposeSelection(operand, tree);
         }
      } else {
         if (auto orOp = dyn_cast_or_null<db::OrOp>(v.getDefiningOp())) {
            //todo: fix potential dominator problem...
            deriveRestrictionsFromOr(orOp, tree);
         }
         OpBuilder builder(currentSel);
         mlir::IRMapping mapping;
         auto newsel = builder.create<relalg::SelectionOp>(currentSel->getLoc(), tuples::TupleStreamType::get(builder.getContext()), tree);
         tree = newsel;
         newsel.initPredicate();
         mapping.map(currentSel.getPredicateArgument(), newsel.getPredicateArgument());
         builder.setInsertionPointToStart(&newsel.getPredicate().front());
         relalg::detail::inlineOpIntoBlock(v.getDefiningOp(), v.getDefiningOp()->getParentOp(), &newsel.getPredicateBlock(), mapping);
         builder.create<tuples::ReturnOp>(currentSel->getLoc(), mapping.lookup(v));
         auto* terminator = newsel.getLambdaBlock().getTerminator();
         terminator->erase();
      }
   }
   static llvm::DenseMap<mlir::Value, relalg::ColumnSet> analyze(mlir::Block* block, relalg::ColumnSet availableLeft, relalg::ColumnSet availableRight) {
      llvm::DenseMap<mlir::Value, relalg::ColumnSet> required;
      relalg::ColumnSet leftKeys, rightKeys;
      std::vector<mlir::Type> types;
      block->walk([&](mlir::Operation* op) {
         if (auto getAttr = mlir::dyn_cast_or_null<tuples::GetColumnOp>(op)) {
            required.insert({getAttr.getResult(), relalg::ColumnSet::from(getAttr.getAttr())});
         } else {
            relalg::ColumnSet attributes;
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
   mlir::Value decomposeOuterJoin(mlir::Value v, relalg::ColumnSet availableLeft, relalg::ColumnSet availableRight, llvm::DenseMap<mlir::Value, relalg::ColumnSet> required) {
      auto currentJoinOp = mlir::dyn_cast_or_null<relalg::OuterJoinOp>(v.getDefiningOp()->getParentOp());
      using namespace mlir;
      if (auto andop = dyn_cast_or_null<db::AndOp>(v.getDefiningOp())) {
         std::vector<Value> vals;
         for (auto operand : andop.getVals()) {
            auto val = decomposeOuterJoin(operand, availableLeft, availableRight, required);
            if (val) {
               vals.push_back(val);
            }
         }
         OpBuilder builder(andop);
         auto newAndOp = builder.create<db::AndOp>(andop->getLoc(), vals);
         andop->remove();
         andop->dropAllReferences();
         //andop->destroy();
         return newAndOp;
      } else {
         if (required[v].isSubsetOf(availableRight)) {
            auto children = currentJoinOp.getChildren();
            OpBuilder builder(currentJoinOp);
            mlir::IRMapping mapping;
            auto newsel = builder.create<relalg::SelectionOp>(currentJoinOp->getLoc(), tuples::TupleStreamType::get(builder.getContext()), children[1].asRelation());
            newsel.initPredicate();
            mapping.map(currentJoinOp.getPredicateArgument(), newsel.getPredicateArgument());
            builder.setInsertionPointToStart(&newsel.getPredicate().front());
            relalg::detail::inlineOpIntoBlock(v.getDefiningOp(), v.getDefiningOp()->getParentOp(), &newsel.getPredicateBlock(), mapping);
            builder.create<tuples::ReturnOp>(currentJoinOp->getLoc(), mapping.lookup(v));
            auto* terminator = newsel.getLambdaBlock().getTerminator();
            terminator->remove();
            terminator->destroy();
            currentJoinOp.setChildren({children[0], newsel});
            return Value();
         }
         return v;
      }
   }
   void decomposeMap(relalg::MapOp currentMap, mlir::Value& tree) {
      using namespace mlir;

      auto* terminator = currentMap.getPredicate().front().getTerminator();
      if (auto returnOp = mlir::dyn_cast_or_null<tuples::ReturnOp>(terminator)) {
         assert(returnOp.getResults().size() == currentMap.getComputedCols().size());
         auto computedValRange = returnOp.getResults();
         for (size_t i = 0; i < computedValRange.size(); i++) {
            OpBuilder builder(currentMap);
            mlir::IRMapping mapping;
            auto currentAttr = mlir::cast<tuples::ColumnDefAttr>(currentMap.getComputedCols()[i]);
            mlir::Value currentVal = computedValRange[i];
            auto newmap = builder.create<relalg::MapOp>(currentMap->getLoc(), tuples::TupleStreamType::get(builder.getContext()), tree, builder.getArrayAttr({currentAttr}));
            tree = newmap;
            newmap.getPredicate().push_back(new Block);
            newmap.getPredicate().addArgument(tuples::TupleType::get(builder.getContext()), currentMap->getLoc());
            builder.setInsertionPointToStart(&newmap.getPredicate().front());
            auto ret1 = builder.create<tuples::ReturnOp>(currentMap->getLoc());
            mapping.map(currentMap.getLambdaArgument(), newmap.getLambdaArgument());
            relalg::detail::inlineOpIntoBlock(currentVal.getDefiningOp(), currentVal.getDefiningOp()->getParentOp(), &newmap.getLambdaBlock(), mapping);
            builder.create<tuples::ReturnOp>(currentMap->getLoc(), mapping.lookup(currentVal));
            ret1->remove();
            ret1->dropAllReferences();
            ret1->destroy();
         }
      }
   }
   void runOnOperation() override {
      mlir::RewritePatternSet patterns(&getContext());
      patterns.insert<DecomposeInnerJoin>(&getContext());
      if (mlir::applyPatternsGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
         signalPassFailure();
      }
      getOperation().walk([&](relalg::SelectionOp op) {
         auto* terminator = op.getRegion().front().getTerminator();
         mlir::Value val = op.getRel();
         if(terminator->getNumOperands()>0){
            decomposeSelection(terminator->getOperand(0), val);
         }
         op.replaceAllUsesWith(val);
         op->erase();
      });
      getOperation().walk([&](relalg::MapOp op) {
         mlir::Value val = op.getRel();
         decomposeMap(op, val);
         op.replaceAllUsesWith(val);
         op->erase();
      });

      getOperation().walk([&](relalg::OuterJoinOp op) {
         auto* terminator = op.getRegion().front().getTerminator();
         if(terminator->getNumOperands()==0){
            return;
         }
         auto retval = terminator->getOperand(0);
         auto availableLeft = op.getChildren()[0].getAvailableColumns();
         auto availableRight = op.getChildren()[1].getAvailableColumns();
         auto mapped = analyze(&op.getPredicateBlock(), availableLeft, availableRight);
         auto val = decomposeOuterJoin(retval, availableLeft, availableRight, mapped);
         mlir::OpBuilder builder(terminator);
         builder.create<tuples::ReturnOp>(terminator->getLoc(), val ? mlir::ValueRange{val} : mlir::ValueRange{});
         terminator->erase();
      });

   }
};
} // end anonymous namespace


std::unique_ptr<mlir::Pass> relalg::createDecomposeLambdasPass() { return std::make_unique<DecomposeLambdas>(); }
