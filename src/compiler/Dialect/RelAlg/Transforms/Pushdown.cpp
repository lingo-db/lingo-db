#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/ColumnSet.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/helper.h"
#include "lingodb/utility/Setting.h"

#include "llvm/ADT/TypeSwitch.h"

#include "lingodb/compiler/Dialect/RelAlg/Transforms/ColumnCreatorAnalysis.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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
   lingodb::runtime::FilterOp convertCmpMode(std::string cmpMode) {
      if (cmpMode == "eq") return lingodb::runtime::FilterOp::EQ;
      if (cmpMode == "neq") return lingodb::runtime::FilterOp::NEQ;
      if (cmpMode == "lt") return lingodb::runtime::FilterOp::LT;
      if (cmpMode == "gt") return lingodb::runtime::FilterOp::GT;
      if (cmpMode == "lte") return lingodb::runtime::FilterOp::LTE;
      if (cmpMode == "gte") return lingodb::runtime::FilterOp::GTE;
      std::cerr << "Unsupprted cmpMode" << cmpMode << std::endl;
      if (cmpMode == "isa") return lingodb::runtime::FilterOp::IN;
      return lingodb::runtime::FilterOp::NOTNULL;
   }
   lingodb::runtime::FilterOp reverseCmpMode(lingodb::runtime::FilterOp cmpMode) {
      switch (cmpMode) {
         case lingodb::runtime::FilterOp::EQ:
         case lingodb::runtime::FilterOp::NEQ:
            return cmpMode;
         case lingodb::runtime::FilterOp::LT:
            return lingodb::runtime::FilterOp::GT;
         case lingodb::runtime::FilterOp::GT:
            return lingodb::runtime::FilterOp::LT;
         case lingodb::runtime::FilterOp::LTE:
            return lingodb::runtime::FilterOp::GTE;
         case lingodb::runtime::FilterOp::GTE:
            return lingodb::runtime::FilterOp::LTE;
         default: {
            std::cerr << "Unsupported cmp mode for reversal\n";
            return lingodb::runtime::FilterOp::IN;
         }
      }
   }
   bool getColumnName(mlir::Value val, relalg::BaseTableOp baseTableOp, std::string& outColumnName, bool& nullable) {
      auto getColOp = mlir::dyn_cast_or_null<tuples::GetColumnOp>(val.getDefiningOp());
      if (auto castOp = mlir::dyn_cast_or_null<db::CastOp>(val.getDefiningOp())) {
         if (mlir::isa<db::StringType>(getBaseType(castOp.getType())) && mlir::isa<db::CharType>(getBaseType(castOp.getVal().getType()))) {
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
   bool getConstant(mlir::Value val, std::variant<std::string, int64_t, double>& outConst) {
      auto constOp = mlir::dyn_cast_or_null<db::ConstantOp>(val.getDefiningOp());
      if (!constOp) return false;
      if (auto strAttr = mlir::dyn_cast_or_null<mlir::StringAttr>(constOp.getValue())) {
         outConst = strAttr.getValue().str();
         return true;
      } else if (auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(constOp.getValue())) {
         outConst = intAttr.getInt();
         return true;
      }
      return false;
   }
   bool appendRestrictions(relalg::BaseTableOp baseTableOp, std::vector<lingodb::runtime::FilterDescription> restrictions) {
      std::vector<lingodb::runtime::FilterDescription> existingRestrictions = baseTableOp.getDatasource().filterDescription;

      for (auto& r : restrictions) {
         existingRestrictions.push_back(r);
      }
      baseTableOp.getProperties<>().datasource.filterDescription = existingRestrictions;
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
               lingodb::runtime::FilterDescription desc{.columnName = columnName, .columnId = 0, .op = lingodb::runtime::FilterOp::NOTNULL, .value = 0, .values = {}};
               appendRestrictions(baseTableOp, {desc});
               return true;
            }
         }
      }
      if (auto condOp = mlir::dyn_cast_or_null<db::CmpOp>(returnOp.getResults()[0].getDefiningOp())) {
         if (getBaseType(condOp.getLeft().getType()) != getBaseType(condOp.getRight().getType())) return false;
         auto* left = condOp.getLeft().getDefiningOp();
         auto* right = condOp.getRight().getDefiningOp();
         if (!left || !right) return false;
         std::variant<std::string, int64_t, double> constValue{};
         std::string columnName;
         bool colNullable;
         lingodb::runtime::FilterOp cmpMode = convertCmpMode(stringifyDBCmpPredicate(condOp.getPredicate()).str());
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
            lingodb::runtime::FilterDescription desc{.columnName = columnName, .columnId = 0, .op = lingodb::runtime::FilterOp::NOTNULL, .value = 0, .values = {}};
            appendRestrictions(baseTableOp, {desc});
         }
         lingodb::runtime::FilterDescription desc{.columnName = columnName, .columnId = 0, .op = cmpMode, .value = constValue, .values = {}};
         appendRestrictions(baseTableOp, {desc});
         return true;
      }
      if (auto betweenOp = mlir::dyn_cast_or_null<db::BetweenOp>(returnOp.getResults()[0].getDefiningOp())) {
         std::string columnName;
         std::variant<std::string, int64_t, double> lowerConst;
         std::variant<std::string, int64_t, double> upperConst;
         bool colNullable;
         if (getBaseType(betweenOp.getVal().getType()) != betweenOp.getLower().getType()) return false;
         if (getBaseType(betweenOp.getVal().getType()) != betweenOp.getUpper().getType()) return false;
         if (getColumnName(betweenOp.getVal(), baseTableOp, columnName, colNullable) &&
             getConstant(betweenOp.getLower(), lowerConst) &&
             getConstant(betweenOp.getUpper(), upperConst)) {
            if (colNullable) {
               lingodb::runtime::FilterDescription desc{.columnName = columnName, .columnId = 0, .op = lingodb::runtime::FilterOp::NOTNULL, .value = 0, .values = {}};
               appendRestrictions(baseTableOp, {desc});
            }

            lingodb::runtime::FilterDescription desc{.columnName = columnName, .columnId = 0, .op = betweenOp.getLowerInclusive() ? lingodb::runtime::FilterOp::GTE : lingodb::runtime::FilterOp::GT, .value = lowerConst, .values = {}};
            lingodb::runtime::FilterDescription desc2{.columnName = columnName, .columnId = 0, .op = betweenOp.getUpperInclusive() ? lingodb::runtime::FilterOp::LTE : lingodb::runtime::FilterOp::LT, .value = upperConst, .values = {}};
            appendRestrictions(baseTableOp, {desc, desc2});
            return true;
         }
      }
      if (auto inOp = mlir::dyn_cast_or_null<db::OneOfOp>(returnOp.getResults()[0].getDefiningOp())) {
         std::string columnName;
         bool colNullable;
         std::variant<std::vector<std::string>, std::vector<int64_t>, std::vector<double>> vals;
         if (!getColumnName(inOp.getVal(), baseTableOp, columnName, colNullable)) return false;
         for (auto val : inOp.getVals()) {
            std::variant<std::string, int64_t, double> constVal;
            if (!getConstant(val, constVal)) return false;
            std::visit([&](auto const& val) {
               using T = std::decay_t<decltype(val)>;
               if constexpr (std::is_same_v<T, std::string>) {
                  if (!std::holds_alternative<std::vector<std::string>>(vals)) vals = std::vector<std::string>{};
                  std::get<0>(vals).push_back(val);
               }
               if constexpr (std::is_same_v<T, int64_t>) {
                  if (!std::holds_alternative<std::vector<int64_t>>(vals)) vals = std::vector<int64_t>{};
                  std::get<1>(vals).push_back(val);
               }
               if constexpr (std::is_same_v<T, double>) {
                  if (!std::holds_alternative<std::vector<double>>(vals)) vals = std::vector<double>{};
                  std::get<2>(vals).push_back(val);
               }
            },
                       constVal);
         }
         if (colNullable) {
            lingodb::runtime::FilterDescription desc{.columnName = columnName, .columnId = 0, .op = lingodb::runtime::FilterOp::NOTNULL, .value = 0, .values = {}};
            appendRestrictions(baseTableOp, {desc});
         }
         lingodb::runtime::FilterDescription desc{.columnName = columnName, .columnId = 0, .op = lingodb::runtime::FilterOp::IN, .value = 0, .values = vals};
         appendRestrictions(baseTableOp, {desc});
         return true;
      }

      return false;
   }
   Operator pushdown(Operator topush, Operator curr, relalg::ColumnCreatorAnalysis& columnCreatorAnalysis, bool ignoreMultUse = false) {
      if (countUses(curr) > 1 && !ignoreMultUse) {
         topush.setChildren({curr});
         return topush;
      }
      UnaryOperator topushUnary = mlir::dyn_cast_or_null<UnaryOperator>(topush.getOperation());
      relalg::ColumnSet usedAttributes = topush.getUsedColumns();
      if (mlir::isa<relalg::SelectionOp>(topush.getOperation()) && mlir::isa<relalg::SelectionOp>(curr.getOperation())) {
         if (mlir::OperationEquivalence::isRegionEquivalentTo(&topush->getRegion(0), &curr->getRegion(0), mlir::OperationEquivalence::IgnoreLocations)) {
            toErase.insert(curr);
            return pushdown(topush, curr.getChildren()[0], columnCreatorAnalysis);
         }
      }
      auto res = ::llvm::TypeSwitch<mlir::Operation*, Operator>(curr.getOperation())

                    .Case<relalg::RenamingOp>([&](relalg::RenamingOp renamingOp) {
                       Operator asOp = mlir::dyn_cast_or_null<Operator>(renamingOp.getOperation());
                       auto child = mlir::dyn_cast_or_null<Operator>(renamingOp.child());
                       bool allColumnsAvailable = true;

                       std::unordered_map<const tuples::Column*, const tuples::Column*> colMapping;
                       for (auto mappingAttr : renamingOp.getColumnsAttr()) {
                          auto columnDefAttr = mlir::dyn_cast_or_null<tuples::ColumnDefAttr>(mappingAttr);
                          auto columnRefAttr = mlir::cast<tuples::ColumnRefAttr>(mlir::cast<mlir::ArrayAttr>(columnDefAttr.getFromExisting())[0]);
                          colMapping[&columnDefAttr.getColumn()] = &columnRefAttr.getColumn();
                       }
                       for (const auto* c : usedAttributes) {
                          c = colMapping.contains(c) ? colMapping[c] : c;
                          allColumnsAvailable &= columnCreatorAnalysis.getCreator(c).canColumnReach(Operator{}, child, c);
                       }

                       if (!allColumnsAvailable) {
                          topush.setChildren({asOp});
                          return topush;
                       }
                       auto& colManager = renamingOp.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
                       topush->walk([&](tuples::GetColumnOp getColOp) {
                          auto* col = &getColOp.getAttr().getColumn();
                          if (colMapping.contains(col)) {
                             getColOp.setAttrAttr(colManager.createRef(colMapping[col]));
                          }
                       });
                       topush->moveBefore(asOp.getOperation());
                       asOp.setChildren({pushdown(topush, child, columnCreatorAnalysis)});
                       return asOp;
                    })
                    .Case<relalg::WindowOp>([&](relalg::WindowOp windowOp) {
                       Operator asOp = mlir::dyn_cast_or_null<Operator>(windowOp.getOperation());
                       auto child = mlir::dyn_cast_or_null<Operator>(windowOp.child());
                       bool allColumnsAvailable = true;
                       for (const auto* c : usedAttributes) {
                          allColumnsAvailable &= columnCreatorAnalysis.getCreator(c).canColumnReach(Operator{}, child, c);
                       }
                       auto windowOrderCols = relalg::ColumnSet::fromArrayAttr(windowOp.getOrderBy());
                       auto computedCols = relalg::ColumnSet::fromArrayAttr(windowOp.getComputedCols());
                       if (allColumnsAvailable && !windowOrderCols.intersects(usedAttributes) && !computedCols.intersects(usedAttributes)) {
                          topush->moveBefore(asOp.getOperation());
                          asOp.setChildren({pushdown(topush, child, columnCreatorAnalysis)});
                          return asOp;
                       }
                       topush.setChildren({asOp});
                       return topush;
                    })
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
                    .Case<relalg::UnionOp>([&](relalg::UnionOp unionOp) {
                       Operator asOp = mlir::dyn_cast_or_null<Operator>(unionOp.getOperation());
                       llvm::SmallVector<Operator, 4> newChildren;
                       size_t i = 0;
                       for (auto childVal : unionOp.getOperands()) {
                          Operator topushnow;
                          if (i == unionOp->getNumOperands() - 1) {
                             topushnow = topush;
                             topushnow->moveBefore(unionOp.getOperation());
                          } else {
                             topushnow = topush.clone();
                             mlir::OpBuilder builder(unionOp.getContext());
                             builder.setInsertionPoint(unionOp.getOperation());
                             builder.insert(topushnow.getOperation());
                          }
                          auto child = mlir::dyn_cast_or_null<Operator>(childVal.getDefiningOp());
                          std::unordered_map<const tuples::Column*, const tuples::Column*> colMapping;
                          for (auto mappingAttr : unionOp.getMapping()) {
                             auto columnDefAttr = mlir::dyn_cast_or_null<tuples::ColumnDefAttr>(mappingAttr);
                             auto columnRefAttr = mlir::cast<tuples::ColumnRefAttr>(mlir::cast<mlir::ArrayAttr>(columnDefAttr.getFromExisting())[i]);
                             colMapping[&columnDefAttr.getColumn()] = &columnRefAttr.getColumn();
                          }
                          auto& colManager = unionOp.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
                          topushnow->walk([&](tuples::GetColumnOp getColOp) {
                             auto* col = &getColOp.getAttr().getColumn();
                             if (colMapping.contains(col)) {
                                getColOp.setAttrAttr(colManager.createRef(colMapping[col]));
                             }
                          });
                          newChildren.push_back(pushdown(topushnow, child, columnCreatorAnalysis));
                          i++;
                       }
                       asOp.setChildren(newChildren);
                       return asOp;
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
      getOperation()->walk([&](relalg::SelectionOp sel) {
         if (auto op = mlir::cast<Operator>(sel.getRel().getDefiningOp())) {
            if (countUses(op) > 1) {
               std::vector<std::vector<relalg::SelectionOp>> collectedSelectionOps;
               //for each use: count selection chain
               for (auto* user : op->getUsers()) {
                  std::vector<relalg::SelectionOp> selectionOps;
                  while (user) {
                     if (auto selOp = mlir::dyn_cast<relalg::SelectionOp>(user)) {
                        selectionOps.push_back(selOp);
                     } else {
                        break;
                     }
                     if (countUses(mlir::cast<Operator>(user)) != 1) break;
                     user = *user->getUsers().begin();
                  }
                  collectedSelectionOps.push_back(selectionOps);
               }
               std::vector<relalg::SelectionOp> sharedSelections;
               for (auto currSelOp : collectedSelectionOps[0]) {
                  std::vector<relalg::SelectionOp> sameOps;
                  bool allMatch = true;
                  for (size_t i = 1; i < collectedSelectionOps.size(); i++) {
                     bool anyMatch = false;
                     for (auto otherSelOp : collectedSelectionOps[i]) {
                        if (mlir::OperationEquivalence::isRegionEquivalentTo(&currSelOp->getRegion(0), &otherSelOp->getRegion(0), mlir::OperationEquivalence::IgnoreLocations)) {
                           anyMatch = true;
                           sameOps.push_back(otherSelOp);
                           break;
                        }
                     }
                     if (!anyMatch) {
                        allMatch = false;
                        break;
                     }
                  }
                  if (allMatch) {
                     SmallPtrSet<mlir::Operation*, 4> users;
                     for (auto* u : currSelOp->getUsers()) {
                        users.insert(u);
                     }
                     auto childBefore = currSelOp.getChildren()[0];
                     auto relBefore = currSelOp.getRel();
                     Operator pushedDown = pushdown(currSelOp, op, columnCreatorAnalysis, true);
                     if (currSelOp.getOperation() != pushedDown.getOperation()) {
                        //"remove currSelOp from previous "chain"
                        currSelOp.getResult().replaceUsesWithIf(relBefore, [&](mlir::OpOperand& operand) {
                           return users.contains(operand.getOwner());
                        });
                        for (auto o : sameOps) {
                           o.replaceAllUsesWith(o.getRel());
                           toErase.insert(o.getOperation());
                        }
                     } else {
                        currSelOp.setChildren({childBefore});
                     }
                  }
               }
            }
         }
      });

      for (auto* op : toErase) {
         op->erase();
      }
      mlir::RewritePatternSet patterns(&getContext());
      patterns.insert<OuterJoinToInnerJoin>(&getContext());
      patterns.insert<SingleJoinToInnerJoin>(&getContext());
      if (lingodb::compiler::applyPatternsGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
         signalPassFailure();
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> relalg::createPushdownPass() { return std::make_unique<Pushdown>(); }
