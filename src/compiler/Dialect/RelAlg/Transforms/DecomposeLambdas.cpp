#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/helper.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
      rewriter.replaceOp(op, sel.getResult());
      return mlir::success();
   }
};
class DecomposeLambdas : public mlir::PassWrapper<DecomposeLambdas, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-decompose-lambdas"; }
   bool deriveExtraConditions;

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DecomposeLambdas)
   DecomposeLambdas(bool deriveExtraConditions) : deriveExtraConditions(deriveExtraConditions) {}

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
   void getConditionValsFromSelection(mlir::Value v, std::vector<mlir::Value>& values) {
      if (auto andop = dyn_cast_or_null<db::AndOp>(v.getDefiningOp())) {
         for (auto operand : andop.getVals()) {
            getConditionValsFromSelection(operand, values);
         }
      } else {
         values.push_back(v);
      }
   }
   struct DerivedRestriction {
      enum DerivedRestrictionType {
         TRUE,
         FALSE,
         OR,
         COL_EQUALITY,
         COL_CONTAINS,
      };
      DerivedRestrictionType type;
      tuples::ColumnRefAttr col = {};
      std::string constVal = "";
      std::vector<DerivedRestriction> children = {};
      bool operator==(const DerivedRestriction& other) const {
         if (type != other.type) return false;
         switch (type) {
            case TRUE:
            case FALSE:
               return true;
            case OR:
               if (children.size() != other.children.size()) return false;
               for (size_t i = 0; i < children.size(); i++) {
                  if (!(children[i] == other.children[i])) return false;
               }
               return true;
            case COL_EQUALITY:
            case COL_CONTAINS:
               return col == other.col && constVal == other.constVal;
         }
         return false;
      }
      static DerivedRestriction buildOr(const std::vector<DerivedRestriction>& restrictions) {
         DerivedRestriction result;
         result.type = OR;
         auto insert = [&](const DerivedRestriction& r) {
            if (r.type == FALSE) return;
            for (auto& child : result.children) {
               if (child == r) return;
            }
            result.children.push_back(r);
         };
         for (auto child : restrictions) {
            if (child.type == OR) {
               for (auto& grandChild : child.children) {
                  insert(grandChild);
               }
            } else {
               insert(child);
            }
         }

         return result;
      }
      void dump() {
         switch (type) {
            case TRUE:
               llvm::dbgs() << "TRUE";
               break;
            case FALSE:
               llvm::dbgs() << "FALSE";
               break;
            case OR:
               llvm::dbgs() << "OR(";
               for (auto& child : children) {
                  child.dump();
                  llvm::dbgs() << ", ";
               }
               llvm::dbgs() << ")";
               break;
            case COL_EQUALITY:
               llvm::dbgs() << "COL_EQUALITY(";
               col.dump();
               llvm::dbgs() << ", ";
               llvm::printEscapedString(constVal, llvm::dbgs());
               llvm::dbgs() << ")";
               break;
            case COL_CONTAINS:
               llvm::dbgs() << "COL_CONTAINS(";
               col.dump();
               llvm::dbgs() << ", ";
               llvm::printEscapedString(constVal, llvm::dbgs());
               llvm::dbgs() << ")";
               break;
         }
      }
   };
   struct Requirement {
      enum RestrType {
         EQUAL,
         CONTAINS
      };
      RestrType type;
      std::string value;
   };
   DerivedRestriction deriveRestrictions(mlir::Value v, Requirement r) {
      auto definingOp = v.getDefiningOp();
      if (!definingOp) return DerivedRestriction{.type = DerivedRestriction::TRUE};
      auto opResult = mlir::cast<mlir::OpResult>(v);
      return llvm::TypeSwitch<mlir::Operation*, DerivedRestriction>(definingOp)
         .Case<tuples::GetColumnOp>([&](tuples::GetColumnOp getColOp) -> DerivedRestriction {
            if (r.type == Requirement::EQUAL) {
               return DerivedRestriction{.type = DerivedRestriction::COL_EQUALITY, .col = getColOp.getAttr(), .constVal = r.value};
            } else if (r.type == Requirement::CONTAINS) {
               return DerivedRestriction{.type = DerivedRestriction::COL_CONTAINS, .col = getColOp.getAttr(), .constVal = r.value};
            }
            return DerivedRestriction{.type = DerivedRestriction::TRUE};
         })
         .Case<db::ConstantOp>([&](db::ConstantOp constOp) -> DerivedRestriction {
            if (auto strAttr = mlir::dyn_cast_or_null<mlir::StringAttr>(constOp.getValue())) {
               if (strAttr.getValue() == r.value) {
                  return DerivedRestriction{.type = DerivedRestriction::TRUE};
               } else {
                  return DerivedRestriction{.type = DerivedRestriction::FALSE};
               }
            }
            return DerivedRestriction{.type = DerivedRestriction::TRUE};
         })
         .Case<db::NullOp>([&](db::NullOp nullOp) -> DerivedRestriction {
            //todo: maybe be more carefull if more restrictions are added
            return DerivedRestriction{.type = DerivedRestriction::FALSE};
         })
         .Case<db::AsNullableOp>([&](db::AsNullableOp asNullableOp) -> DerivedRestriction {
            return deriveRestrictions(asNullableOp.getVal(), r);
         })
         .Case<mlir::arith::SelectOp>([&](mlir::arith::SelectOp selectOp) -> DerivedRestriction {
            auto derivedTrue = deriveRestrictions(selectOp.getTrueValue(), r);
            auto derivedFalse = deriveRestrictions(selectOp.getFalseValue(), r);
            if (derivedTrue.type == DerivedRestriction::TRUE && derivedFalse.type == DerivedRestriction::TRUE) {
               return DerivedRestriction{.type = DerivedRestriction::TRUE};
            } else if (derivedTrue.type == DerivedRestriction::FALSE && derivedFalse.type == DerivedRestriction::FALSE) {
               return DerivedRestriction{.type = DerivedRestriction::FALSE};
            } else if (derivedTrue.type == DerivedRestriction::FALSE && derivedFalse.type == DerivedRestriction::TRUE) {
               return DerivedRestriction{.type = DerivedRestriction::TRUE};
            } else if (derivedTrue.type == DerivedRestriction::TRUE && derivedFalse.type == DerivedRestriction::FALSE) {
               return deriveRestrictions(selectOp.getCondition());
            }
            return DerivedRestriction::buildOr({derivedTrue, derivedFalse});
         })
         .Case<db::RuntimeCall>([&](db::RuntimeCall runtimeCall) -> DerivedRestriction {
            if (runtimeCall.getFn() == "Substring") {
               auto strArg = runtimeCall->getOperand(0);
               if (r.type == Requirement::CONTAINS || r.type == Requirement::EQUAL) {
                  return deriveRestrictions(strArg, r);
               }
            } else if (runtimeCall.getFn() == "Replace") {
               auto strArg = runtimeCall->getOperand(0);
               auto searchArg = runtimeCall->getOperand(1);
               if (auto constOp = mlir::dyn_cast_or_null<db::ConstantOp>(searchArg.getDefiningOp())) {
                  if (auto strAttr = mlir::dyn_cast_or_null<mlir::StringAttr>(constOp.getValue())) {
                     if (r.type == Requirement::CONTAINS || r.type == Requirement::EQUAL) {
                        return DerivedRestriction::buildOr({deriveRestrictions(strArg, Requirement(Requirement::CONTAINS, r.value)), deriveRestrictions(strArg, Requirement{.type = Requirement::CONTAINS, .value = strAttr.getValue().str()})});
                     }
                  }
               }
            }
            return DerivedRestriction{.type = DerivedRestriction::TRUE};
         })
         .Case<mlir::scf::IfOp>([&](mlir::scf::IfOp ifOp) -> DerivedRestriction {
            auto derivedThen = deriveRestrictions(ifOp.getThenRegion().front().getTerminator()->getOperand(opResult.getResultNumber()), r);
            auto derivedElse = deriveRestrictions(ifOp.getElseRegion().front().getTerminator()->getOperand(opResult.getResultNumber()), r);
            return DerivedRestriction::buildOr({derivedThen, derivedElse});
         })
         .Default([&](mlir::Operation* op) -> DerivedRestriction {
            return DerivedRestriction{.type = DerivedRestriction::TRUE};
         });
      return DerivedRestriction{.type = DerivedRestriction::TRUE};
   }
   mlir::Value buildRestriction(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value tuple, DerivedRestriction& restriction) {
      auto i1Type = builder.getIntegerType(1);
      switch (restriction.type) {
         case DerivedRestriction::TRUE:
            return builder.create<db::ConstantOp>(loc, i1Type, builder.getIntegerAttr(i1Type, 1));
         case DerivedRestriction::FALSE:
            return builder.create<db::ConstantOp>(loc, i1Type, builder.getIntegerAttr(i1Type, 0));
         case DerivedRestriction::COL_EQUALITY: {
            auto colVal = builder.create<tuples::GetColumnOp>(loc, restriction.col.getColumn().type, restriction.col, tuple);
            auto constVal = builder.create<db::ConstantOp>(loc, db::StringType::get(builder.getContext()), builder.getStringAttr(restriction.constVal));
            return builder.create<db::CmpOp>(loc, db::DBCmpPredicate::eq, colVal, constVal);
         }
         case DerivedRestriction::COL_CONTAINS: {
            auto colVal = builder.create<tuples::GetColumnOp>(loc, restriction.col.getColumn().type, restriction.col, tuple);
            auto constVal = builder.create<db::ConstantOp>(loc, db::StringType::get(builder.getContext()), builder.getStringAttr(restriction.constVal));
            return builder.create<db::RuntimeCall>(loc, i1Type, "Contains", mlir::ValueRange{colVal, constVal}).getRes();
         }
         case DerivedRestriction::OR: {
            std::vector<mlir::Value> vals;
            for (auto& child : restriction.children) {
               vals.push_back(buildRestriction(builder, loc, tuple, child));
            }
            return builder.create<db::OrOp>(loc, vals);
         }
      }
      assert(false);
      return mlir::Value();
   }
   DerivedRestriction deriveRestrictions(mlir::Value v) {
      if (auto cmpOp = dyn_cast_or_null<db::CmpOp>(v.getDefiningOp())) {
         if (auto rightValue = mlir::dyn_cast_or_null<db::ConstantOp>(cmpOp.getRight().getDefiningOp())) {
            return deriveRestrictions(cmpOp.getLeft(), Requirement{.type = Requirement::EQUAL, .value = mlir::cast<mlir::StringAttr>(rightValue.getValue()).str()});
         }
      }
      if (auto runtimeCall = dyn_cast_or_null<db::RuntimeCall>(v.getDefiningOp())) {
         if (runtimeCall.getFn() == "Contains") {
            auto rightValue = mlir::dyn_cast_or_null<db::ConstantOp>(runtimeCall.getOperand(1).getDefiningOp());
            if (rightValue) {
               return deriveRestrictions(runtimeCall.getOperand(0), Requirement{.type = Requirement::CONTAINS, .value = mlir::cast<mlir::StringAttr>(rightValue.getValue()).str()});
            }
         }
         if (runtimeCall.getFn() == "ConstLike") {
            auto rightValue = mlir::dyn_cast_or_null<db::ConstantOp>(runtimeCall.getOperand(1).getDefiningOp());
            if (rightValue) {
               auto pattern = mlir::cast<mlir::StringAttr>(rightValue.getValue()).str();
               //remove % from start and end, if present
               if (!pattern.empty() && pattern.front() == '%') {
                  pattern = pattern.substr(1);
               }
               if (!pattern.empty() && pattern.back() == '%') {
                  pattern = pattern.substr(0, pattern.size() - 1);
               }
               // now: if there are still % or _ in the pattern, we cannot derive a restriction
               if (pattern.find('%') != std::string::npos || pattern.find('_') != std::string::npos) {
                  return DerivedRestriction{.type = DerivedRestriction::TRUE};
               }
               return deriveRestrictions(runtimeCall.getOperand(0), Requirement{.type = Requirement::CONTAINS, .value = pattern});
            }
         }
      }
      if (auto orOp = dyn_cast_or_null<db::OrOp>(v.getDefiningOp())) {
         std::vector<DerivedRestriction> childRestrictions;
         for (auto operand : orOp.getVals()) {
            childRestrictions.push_back(deriveRestrictions(operand));
         }
         return DerivedRestriction::buildOr(childRestrictions);
      }
      return DerivedRestriction{.type = DerivedRestriction::TRUE};
   }
   void addRestricitonsToSelection(mlir::Value v, mlir::Value& tree) {
      auto currentSel = mlir::dyn_cast_or_null<relalg::SelectionOp>(v.getDefiningOp()->getParentOp());
      auto restriction = deriveRestrictions(v);
      if (restriction.type == DerivedRestriction::TRUE) return;
      mlir::OpBuilder builder(currentSel);
      auto newsel = builder.create<relalg::SelectionOp>(currentSel->getLoc(), tuples::TupleStreamType::get(builder.getContext()), tree);
      tree = newsel;
      newsel.initPredicate();
      auto terminator = newsel.getLambdaBlock().getTerminator();
      builder.setInsertionPointToStart(&newsel.getPredicate().front());
      auto val = buildRestriction(builder, currentSel->getLoc(), newsel.getPredicateArgument(), restriction);
      builder.create<tuples::ReturnOp>(currentSel->getLoc(), val);
      terminator->erase();
   }
   void decomposeSelection(mlir::Value v, mlir::Value& tree) {
      auto currentSel = mlir::dyn_cast_or_null<relalg::SelectionOp>(v.getDefiningOp()->getParentOp());
      using namespace mlir;
      if (deriveExtraConditions) {
         if (auto orOp = dyn_cast_or_null<db::OrOp>(v.getDefiningOp())) {
            //todo: fix potential dominator problem...
            deriveRestrictionsFromOr(orOp, tree);
         }
#if ENABLE_NECESSARY_PRECOND == 1
         addRestricitonsToSelection(v, tree);
#endif
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
         mlir::Value newAndOp = builder.create<db::AndOp>(andop->getLoc(), vals);
         andop.replaceAllUsesWith(newAndOp);
         andop->erase();
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
            terminator->erase();
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
            ret1->erase();
         }
      }
   }
   void runOnOperation() override {
      mlir::RewritePatternSet patterns(&getContext());
      patterns.insert<DecomposeInnerJoin>(&getContext());
      if (lingodb::compiler::applyPatternsGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
         signalPassFailure();
      }
      std::vector<mlir::Operation*> toErase;
      getOperation().walk([&](relalg::SelectionOp op) {
         auto* terminator = op.getRegion().front().getTerminator();
         mlir::Value val = op.getRel();
         if (terminator->getNumOperands() > 0) {
            std::vector<mlir::Value> conditionValues;
            getConditionValsFromSelection(terminator->getOperand(0), conditionValues);
            if (conditionValues.size() > 1) {
               // decomposition is only needed for multiple conditions
               for (auto condition : conditionValues) {
                  decomposeSelection(condition, val);
               }
               op.replaceAllUsesWith(val);
               toErase.push_back(op.getOperation());
            } else {
#if ENABLE_NECESSARY_PRECOND == 1
               if (deriveExtraConditions) {
                  addRestricitonsToSelection(conditionValues[0], val);
                  op.setOperand(val);
               }
#endif
            }

         } else {
            op.replaceAllUsesWith(val);
            toErase.push_back(op.getOperation());
         }
      });
      getOperation().walk([&](relalg::MapOp op) {
         mlir::Value val = op.getRel();
         if (op.getComputedCols().size() == 1) {
            // single column map does not need decomposition
            return;
         }
         if (auto returnOp = mlir::dyn_cast_or_null<tuples::ReturnOp>(op.getRegion().front().getTerminator())) {
            bool anyRelalgOp = false;
            for (auto v : returnOp.getResults()) {
               if (auto* defOp = v.getDefiningOp()) {
                  if (defOp->getDialect() == op.getContext()->getLoadedDialect<relalg::RelAlgDialect>()) {
                     anyRelalgOp = true;
                     break;
                  }
               }
            }
            if (!anyRelalgOp) return;
         }
         decomposeMap(op, val);
         op.replaceAllUsesWith(val);
         toErase.push_back(op.getOperation());
      });
      getOperation().walk([&](relalg::OuterJoinOp op) {
         auto* terminator = op.getRegion().front().getTerminator();
         if (terminator->getNumOperands() == 0) {
            return;
         }
         auto retval = terminator->getOperand(0);
         auto availableLeft = op.getChildren()[0].getAvailableColumns();
         auto availableRight = op.getChildren()[1].getAvailableColumns();
         auto mapped = analyze(&op.getPredicateBlock(), availableLeft, availableRight);
         auto val = decomposeOuterJoin(retval, availableLeft, availableRight, mapped);
         mlir::OpBuilder builder(terminator);
         builder.create<tuples::ReturnOp>(terminator->getLoc(), val ? mlir::ValueRange{val} : mlir::ValueRange{});
         toErase.push_back(terminator);
      });
      for (auto* op : toErase) {
         op->erase();
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> relalg::createDecomposeLambdasPass(bool deriveExtraConditions) { return std::make_unique<DecomposeLambdas>(deriveExtraConditions); }
