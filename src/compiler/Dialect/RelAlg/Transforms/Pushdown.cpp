#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/ColumnSet.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/utility/Setting.h"
#include "llvm/ADT/TypeSwitch.h"

#include "lingodb/compiler/Dialect/RelAlg/Transforms/ColumnCreatorAnalysis.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "json.h"
namespace {
lingodb::utility::GlobalSetting<bool> pushdownRestrictions("system.opt.pushdown_restrictions", true);
using namespace lingodb::compiler::dialect;
bool isNotNullCheckOnColumn(relalg::ColumnSet relevantColumns, relalg::SelectionOp selectionOp) {
   if (selectionOp.getPredicate().empty()) return false;
   if (selectionOp.getPredicate().front().empty()) return false;
   if (auto returnOp = mlir::dyn_cast_or_null<tuples::ReturnOp>(selectionOp.getPredicate().front().getTerminator())) {
      if (returnOp.getResults().size() != 1) return false;
      if (auto notOp = mlir::dyn_cast_or_null<db::NotOp>(returnOp.getResults()[0].getDefiningOp())) {
         if (auto isNullOp = mlir::dyn_cast_or_null<db::IsNullOp>(notOp.getVal().getDefiningOp())) {
            if (auto getColOp = mlir::dyn_cast_or_null<tuples::GetColumnOp>(isNullOp.getVal().getDefiningOp())) {
               return relevantColumns.contains(&getColOp.getAttr().getColumn());
            }
         }
      }
   }
   return false;
}
class OuterJoinToInnerJoin : public mlir::RewritePattern {
   public:
   OuterJoinToInnerJoin(mlir::MLIRContext* context)
      : RewritePattern(relalg::OuterJoinOp::getOperationName(), 1, context) {}
   mlir::LogicalResult match(mlir::Operation* op) const override {
      auto outerJoinOp = mlir::cast<relalg::OuterJoinOp>(op);
      mlir::Value currStream = outerJoinOp.asRelation();

      while (currStream) {
         auto users = currStream.getUsers();
         if (users.begin() == users.end()) break;
         auto second = users.begin();
         second++;
         if (second != users.end()) break;
         if (auto selectionOp = mlir::dyn_cast_or_null<relalg::SelectionOp>(*users.begin())) {
            if (isNotNullCheckOnColumn(outerJoinOp.getCreatedColumns(), selectionOp)) {
               return mlir::success();
            }
            currStream = selectionOp.asRelation();
         } else {
            currStream = mlir::Value();
         }
      }
      return mlir::failure();
   }
   void rewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto outerJoinOp = mlir::cast<relalg::OuterJoinOp>(op);
      auto newJoin = rewriter.create<relalg::InnerJoinOp>(op->getLoc(), outerJoinOp.getLeft(), outerJoinOp.getRight());
      rewriter.inlineRegionBefore(outerJoinOp.getPredicate(), newJoin.getPredicate(), newJoin.getPredicate().end());
      std::vector<mlir::Attribute> mapColumnDefs;
      auto* mapBlock = new mlir::Block;

      {
         std::vector<mlir::Value> returnValues;
         auto tuple = mapBlock->addArgument(tuples::TupleType::get(getContext()), op->getLoc());
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(mapBlock);
         for (auto m : outerJoinOp.getMapping()) {
            auto defAttr = mlir::dyn_cast_or_null<tuples::ColumnDefAttr>(m);
            auto refAttr = mlir::cast<tuples::ColumnRefAttr>(mlir::cast<mlir::ArrayAttr>(defAttr.getFromExisting())[0]);
            auto colVal = rewriter.create<tuples::GetColumnOp>(op->getLoc(), defAttr.getColumn().type, refAttr, tuple);
            if (colVal.getType() != defAttr.getColumn().type) {
               returnValues.push_back(rewriter.create<db::AsNullableOp>(op->getLoc(), defAttr.getColumn().type, colVal, mlir::Value()));
            } else {
               returnValues.push_back(colVal);
            }
            mapColumnDefs.push_back(defAttr);
         }
         rewriter.create<tuples::ReturnOp>(op->getLoc(), returnValues);
      }
      auto mapOp = rewriter.replaceOpWithNewOp<relalg::MapOp>(op, newJoin.asRelation(), rewriter.getArrayAttr(mapColumnDefs));
      mapOp.getPredicate().push_back(mapBlock);
   }
};
class SingleJoinToInnerJoin : public mlir::RewritePattern {
   public:
   SingleJoinToInnerJoin(mlir::MLIRContext* context)
      : RewritePattern(relalg::SingleJoinOp::getOperationName(), 1, context) {}
   mlir::LogicalResult match(mlir::Operation* op) const override {
      auto outerJoinOp = mlir::cast<relalg::SingleJoinOp>(op);
      mlir::Value currStream = outerJoinOp.asRelation();

      while (currStream) {
         auto users = currStream.getUsers();
         if (users.begin() == users.end()) break;
         auto second = users.begin();
         second++;
         if (second != users.end()) break;
         if (auto selectionOp = mlir::dyn_cast_or_null<relalg::SelectionOp>(*users.begin())) {
            if (isNotNullCheckOnColumn(outerJoinOp.getCreatedColumns(), selectionOp)) {
               return mlir::success();
            }
            currStream = selectionOp.asRelation();
         } else {
            currStream = mlir::Value();
         }
      }
      return mlir::failure();
   }
   void rewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto outerJoinOp = mlir::cast<relalg::SingleJoinOp>(op);
      auto newJoin = rewriter.create<relalg::InnerJoinOp>(op->getLoc(), outerJoinOp.getLeft(), outerJoinOp.getRight());
      rewriter.inlineRegionBefore(outerJoinOp.getPredicate(), newJoin.getPredicate(), newJoin.getPredicate().end());
      std::vector<mlir::Attribute> mapColumnDefs;
      auto* mapBlock = new mlir::Block;

      {
         std::vector<mlir::Value> returnValues;
         auto tuple = mapBlock->addArgument(tuples::TupleType::get(getContext()), op->getLoc());
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(mapBlock);
         for (auto m : outerJoinOp.getMapping()) {
            auto defAttr = mlir::dyn_cast_or_null<tuples::ColumnDefAttr>(m);
            auto refAttr = mlir::cast<tuples::ColumnRefAttr>(mlir::cast<mlir::ArrayAttr>(defAttr.getFromExisting())[0]);
            auto colVal = rewriter.create<tuples::GetColumnOp>(op->getLoc(), defAttr.getColumn().type, refAttr, tuple);
            if (colVal.getType() != defAttr.getColumn().type) {
               returnValues.push_back(rewriter.create<db::AsNullableOp>(op->getLoc(), defAttr.getColumn().type, colVal, mlir::Value()));
            } else {
               returnValues.push_back(colVal);
            }
            mapColumnDefs.push_back(defAttr);
         }
         rewriter.create<tuples::ReturnOp>(op->getLoc(), returnValues);
      }
      auto mapOp = rewriter.replaceOpWithNewOp<relalg::MapOp>(op, newJoin.asRelation(), rewriter.getArrayAttr(mapColumnDefs));
      mapOp.getPredicate().push_back(mapBlock);
   }
};
class Pushdown : public mlir::PassWrapper<Pushdown, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-pushdown"; }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Pushdown)
   private:
   llvm::DenseSet<mlir::Operation*> toErase;
   size_t countUses(Operator o) {
      size_t uses = 0;
      for (auto& u : o->getUses()) uses++; // NOLINT(clang-diagnostic-unused-variable)
      return uses;
   }
   std::string reverseCmpMode(std::string cmpMode) {
      if (cmpMode == "eq") return "eq";
      if (cmpMode == "neq") return "neq";
      if (cmpMode == "lt") return "gt";
      if (cmpMode == "gt") return "lt";
      if (cmpMode == "lte") return "gte";
      if (cmpMode == "gte") return "lte";
      if (cmpMode == "isa") return "isa";
      return "";
   }
   bool getColumnName(mlir::Value val, relalg::BaseTableOp baseTableOp, std::string& outColumnName, bool& nullable) {
      auto getColOp = mlir::dyn_cast_or_null<tuples::GetColumnOp>(val.getDefiningOp());
      if (auto castOp = mlir::dyn_cast_or_null<db::CastOp>(val.getDefiningOp())) {
         if (mlir::isa<db::StringType>(castOp.getType()) && mlir::isa<db::CharType>(castOp.getVal().getType())) {
            getColOp = mlir::dyn_cast_or_null<tuples::GetColumnOp>(castOp.getVal().getDefiningOp());
         }
      }
      if (!getColOp) return false;
      auto& column = getColOp.getAttr().getColumn();
      if (mlir::isa<db::NullableType>(column.type)) {
         nullable = true;
      } else {
         nullable = false;
      }
      for (auto x : baseTableOp.getColumns()) {
         if (&mlir::cast<tuples::ColumnDefAttr>(x.getValue()).getColumn() == &column) {
            outColumnName = x.getName().str();
            return true;
         }
      }
      return false;
   }
   bool getConstant(mlir::Value val, nlohmann::json& outConst) {
      auto constOp = mlir::dyn_cast_or_null<db::ConstantOp>(val.getDefiningOp());
      if (!constOp) return false;
      if (auto strAttr = mlir::dyn_cast_or_null<mlir::StringAttr>(constOp.getValue())) {
         outConst = strAttr.getValue();
         return true;
      } else if (auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(constOp.getValue())) {
         outConst = intAttr.getInt();
         return true;
      }
      return false;
   }
   bool appendRestrictions(relalg::BaseTableOp baseTableOp, nlohmann::json::array_t restrictions) {
      nlohmann::json::array_t existingRestrictions;
      if (baseTableOp->hasAttr("restriction")) {
         auto existingRestrictionsStr = cast<mlir::StringAttr>(baseTableOp->getAttr("restriction")).getValue();
         existingRestrictions = nlohmann::json::parse(existingRestrictionsStr.str()).get<nlohmann::json::array_t>();
      }
      for (auto& r : restrictions) {
         existingRestrictions.push_back(r);
      }
      auto restrictionsStr = nlohmann::json(existingRestrictions).dump();
      baseTableOp->setAttr("restriction", mlir::StringAttr::get(baseTableOp.getContext(), restrictionsStr));
      return true;
   }
   bool tryPushdownIntoBasetable(relalg::SelectionOp selectionOp, relalg::BaseTableOp baseTableOp) {
      if (selectionOp.getPredicate().empty()) return false;
      auto returnOp = mlir::dyn_cast_or_null<tuples::ReturnOp>(selectionOp.getPredicate().front().getTerminator());
      if (!returnOp) return false;
      if (returnOp.getResults().size() != 1) return false;
      if (auto notOp = mlir::dyn_cast_or_null<db::NotOp>(returnOp.getResults()[0].getDefiningOp())) {
         if (auto isNullOp = mlir::dyn_cast_or_null<db::IsNullOp>(notOp.getVal().getDefiningOp())) {
            std::string columnName;
            bool colNullable;
            if (getColumnName(isNullOp.getVal(), baseTableOp, columnName, colNullable)) {
               assert(!columnName.empty() && "must be column");
               appendRestrictions(baseTableOp, nlohmann::json::array_t{{
                                                  {"column", columnName},
                                                  {"cmp", "isnotnull"},
                                                  {"value", 0},
                                               }});
               return true;
            }
         }
      }
      if (auto condOp = mlir::dyn_cast_or_null<db::CmpOp>(returnOp.getResults()[0].getDefiningOp())) {
         if (getBaseType(condOp.getLeft().getType()) != getBaseType(condOp.getRight().getType())) return false;
         auto* left = condOp.getLeft().getDefiningOp();
         auto* right = condOp.getRight().getDefiningOp();
         if (!left || !right) return false;
         nlohmann::json constValue;
         std::string columnName;
         bool colNullable;
         std::string cmpMode = stringifyDBCmpPredicate(condOp.getPredicate()).str();
         if (getColumnName(condOp.getLeft(), baseTableOp, columnName, colNullable) &&
             getConstant(condOp.getRight(), constValue)) {
            // left is column, right is constant
         } else if (getColumnName(condOp.getRight(), baseTableOp, columnName, colNullable) &&
                    getConstant(condOp.getLeft(), constValue)) {
            // right is column, left is constant
            // reverse cmp mode
            cmpMode = reverseCmpMode(cmpMode);
         } else {
            return false;
         }
         assert(!columnName.empty() && "one side must be column");
         if (colNullable) {
            appendRestrictions(baseTableOp, nlohmann::json::array_t{{
                                               {"column", columnName},
                                               {"cmp", "isnotnull"},
                                               {"value", 0},
                                            }});
         }
         appendRestrictions(baseTableOp, nlohmann::json::array_t{{
                                            {"column", columnName},
                                            {"cmp", cmpMode},
                                            {"value", constValue},
                                         }});
         return true;
      }
      if (auto betweenOp = mlir::dyn_cast_or_null<db::BetweenOp>(returnOp.getResults()[0].getDefiningOp())) {
         std::string columnName;
         nlohmann::json lowerConst;
         nlohmann::json upperConst;
         bool colNullable;
         if (betweenOp.getVal().getType() != betweenOp.getLower().getType()) return false;
         if (betweenOp.getVal().getType() != betweenOp.getUpper().getType()) return false;
         if (getColumnName(betweenOp.getVal(), baseTableOp, columnName, colNullable) &&
             getConstant(betweenOp.getLower(), lowerConst) &&
             getConstant(betweenOp.getUpper(), upperConst)) {
            if (colNullable) {
               appendRestrictions(baseTableOp, nlohmann::json::array_t{{
                                                  {"column", columnName},
                                                  {"cmp", "isnotnull"},
                                                  {"value", 0},
                                               }});
            }
            appendRestrictions(baseTableOp, nlohmann::json::array_t{{
                                                                       {"column", columnName},
                                                                       {"cmp", betweenOp.getLowerInclusive() ? "gte" : "gt"},
                                                                       {"value", lowerConst},
                                                                    },
                                                                    {
                                                                       {"column", columnName},
                                                                       {"cmp", betweenOp.getUpperInclusive() ? "lte" : "lt"},
                                                                       {"value", upperConst},
                                                                    }});
            return true;
         }
      }
      return false;
   }
   Operator pushdown(Operator topush, Operator curr, relalg::ColumnCreatorAnalysis& columnCreatorAnalysis) {
      if (countUses(curr) > 1) {
         topush.setChildren({curr});
         return topush;
      }
      UnaryOperator topushUnary = mlir::dyn_cast_or_null<UnaryOperator>(topush.getOperation());
      relalg::ColumnSet usedAttributes = topush.getUsedColumns();
      auto res = ::llvm::TypeSwitch<mlir::Operation*, Operator>(curr.getOperation())
                    .Case<UnaryOperator>([&](UnaryOperator unaryOperator) {
                       Operator asOp = mlir::dyn_cast_or_null<Operator>(unaryOperator.getOperation());
                       auto child = mlir::dyn_cast_or_null<Operator>(unaryOperator.child());
                       bool allColumnsAvailable = true;
                       for (const auto* c : usedAttributes) {
                          allColumnsAvailable &= columnCreatorAnalysis.getCreator(c).canColumnReach(Operator{}, child, c);
                       }
                       if (topushUnary.reorderable(unaryOperator) && allColumnsAvailable) {
                          topush->moveBefore(asOp.getOperation());
                          asOp.setChildren({pushdown(topush, child, columnCreatorAnalysis)});
                          return asOp;
                       }
                       topush.setChildren({asOp});
                       return topush;
                    })
                    .Case<BinaryOperator>([&](BinaryOperator binop) {
                       Operator asOp = mlir::dyn_cast_or_null<Operator>(binop.getOperation());
                       auto left = mlir::dyn_cast_or_null<Operator>(binop.leftChild());
                       auto right = mlir::dyn_cast_or_null<Operator>(binop.rightChild());
                       bool pushableLeft = true;
                       bool pushableRight = true;

                       for (auto* c : usedAttributes) {
                          pushableLeft &= columnCreatorAnalysis.getCreator(c).canColumnReach(Operator{}, left, c);
                          pushableRight &= columnCreatorAnalysis.getCreator(c).canColumnReach(Operator{}, right, c);
                       }

                       pushableLeft &= topushUnary.lPushable(binop);
                       pushableRight &= topushUnary.rPushable(binop);
                       if (!pushableLeft && !pushableRight) {
                          topush.setChildren({asOp});
                          return topush;
                       } else if (pushableLeft) {
                          topush->moveBefore(asOp.getOperation());
                          left = pushdown(topush, left, columnCreatorAnalysis);
                       } else if (pushableRight) {
                          topush->moveBefore(asOp.getOperation());
                          right = pushdown(topush, right, columnCreatorAnalysis);
                       }
                       asOp.setChildren({left, right});
                       return asOp;
                    })
                    .Case<relalg::NestedOp>([&](relalg::NestedOp nestedOp) -> Operator {
                       auto returnOp = mlir::cast<tuples::ReturnOp>(nestedOp.getNestedFn().front().getTerminator());
                       mlir::Value returnedStream = returnOp.getResults()[0];
                       bool canPushThrough = true;
                       while (auto* definingOp = returnedStream.getDefiningOp()) {
                          if (definingOp->hasOneUse() && definingOp->getNumOperands() > 0) {
                             bool isFirstTupleStream = mlir::isa<tuples::TupleStreamType>(definingOp->getOperand(0).getType());
                             bool noOtherTupleStream = llvm::none_of(definingOp->getOperands().drop_front(), [](mlir::Value v) { return mlir::isa<tuples::TupleStreamType>(v.getType()); });
                             if (isFirstTupleStream && noOtherTupleStream) {
                                returnedStream = definingOp->getOperand(0);
                             } else {
                                canPushThrough = false;
                             }
                             if (auto subop = mlir::dyn_cast_or_null<subop::SubOperator>(definingOp)) {
                                auto writtenMembers = subop.getWrittenMembers();
                                if (!writtenMembers.empty()) {
                                   canPushThrough = false;
                                }
                             }
                          } else {
                             canPushThrough = false;
                          }
                       }
                       if (canPushThrough) {
                          for (size_t i = 0; i < nestedOp.getNumOperands(); i++) {
                             if (returnedStream == nestedOp.getNestedFn().getArgument(i)) {
                                bool allColumnsAvailable = true;
                                auto candidate = mlir::cast<Operator>(nestedOp->getOperand(i).getDefiningOp());
                                for (const auto* c : usedAttributes) {
                                   allColumnsAvailable &= columnCreatorAnalysis.getCreator(c).canColumnReach(Operator{}, candidate, c);
                                }
                                if (allColumnsAvailable) {
                                   topush->moveBefore(candidate.getOperation());
                                   Operator pushedDown = pushdown(topush, candidate, columnCreatorAnalysis);
                                   nestedOp.setOperand(i, pushedDown.asRelation());
                                   return nestedOp;
                                }
                             }
                          }
                       }
                       topush.setChildren({nestedOp});
                       return topush;
                    })
                    .Case<relalg::BaseTableOp>([&](relalg::BaseTableOp baseTableOp) -> Operator {
                       if (!pushdownRestrictions.getValue()) {
                          topush.setChildren({baseTableOp});
                          return topush;
                       }
                       auto successful = tryPushdownIntoBasetable(mlir::cast<relalg::SelectionOp>(topush.getOperation()), baseTableOp);
                       if (successful) {
                          topush->dropAllReferences();
                          topush->remove();
                          toErase.insert(topush.getOperation());
                          return baseTableOp;
                       } else {
                          topush.setChildren({baseTableOp});
                          return topush;
                       }
                    })
                    .Default([&](mlir::Operation* others) {
                       topush.setChildren({mlir::cast<Operator>(others)});
                       return topush;
                    });
      return res;
   }

   void runOnOperation() override {
      relalg::ColumnCreatorAnalysis columnCreatorAnalysis(getOperation());
      using namespace mlir;
      getOperation()->walk([&](relalg::SelectionOp sel) {
         SmallPtrSet<mlir::Operation*, 4> users;
         for (auto* u : sel->getUsers()) {
            users.insert(u);
         }
         Operator pushedDown = pushdown(sel, sel.getChildren()[0], columnCreatorAnalysis);
         if (sel.getOperation() != pushedDown.getOperation()) {
            sel.getResult().replaceUsesWithIf(pushedDown->getResult(0), [&](mlir::OpOperand& operand) {
               return users.contains(operand.getOwner());
            });
         }
      });
      for (auto* op : toErase) {
         op->erase();
      }
      mlir::RewritePatternSet patterns(&getContext());
      patterns.insert<OuterJoinToInnerJoin>(&getContext());
      patterns.insert<SingleJoinToInnerJoin>(&getContext());
      if (mlir::applyPatternsGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
         signalPassFailure();
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> relalg::createPushdownPass() { return std::make_unique<Pushdown>(); }
