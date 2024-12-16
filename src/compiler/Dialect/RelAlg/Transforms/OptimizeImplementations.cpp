#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "lingodb/compiler/Dialect/RelAlg/Transforms/queryopt/QueryGraph.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/mlir-support/eval.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/TypeSwitch.h"

#include <stack>
#include <unordered_set>

namespace {
using namespace lingodb::compiler::dialect;

class HashJoinUtils {
   public:
   static bool isAndedResult(mlir::Operation* op, bool first = true) {
      if (mlir::isa<tuples::ReturnOp>(op)) {
         return true;
      }
      if (mlir::isa<db::AndOp>(op) || first) {
         for (auto* user : op->getUsers()) {
            if (!isAndedResult(user, false)) return false;
         }
         return true;
      } else {
         return false;
      }
   }
   struct MapBlockInfo {
      std::vector<mlir::Value> results;
      mlir::Block* block;
      std::vector<mlir::Attribute> createdColumns;
   };
   static std::pair<std::vector<mlir::Attribute>, std::vector<mlir::Attribute>> extractKeys(mlir::Block* block, relalg::ColumnSet keyAttributes, relalg::ColumnSet otherAttributes, MapBlockInfo& mapBlockInfo) {
      std::vector<mlir::Attribute> toHash;
      std::vector<mlir::Attribute> nullsEqual;
      llvm::DenseMap<mlir::Value, relalg::ColumnSet> required;
      mlir::IRMapping mapping;
      mapping.map(block->getArgument(0), mapBlockInfo.block->getArgument(0));
      size_t i = 0;
      block->walk([&](mlir::Operation* op) {
         if (auto getAttr = mlir::dyn_cast_or_null<tuples::GetColumnOp>(op)) {
            required.insert({getAttr.getResult(), relalg::ColumnSet::from(getAttr.getAttr())});
         } else if (auto cmpOp = mlir::dyn_cast_or_null<relalg::CmpOpInterface>(op)) {
            if (cmpOp.isEqualityPred(true) && isAndedResult(op)) {
               auto leftAttributes = required[cmpOp.getLeft()];
               auto rightAttributes = required[cmpOp.getRight()];
               mlir::Value keyVal;
               if (leftAttributes.isSubsetOf(keyAttributes) && rightAttributes.isSubsetOf(otherAttributes)) {
                  keyVal = cmpOp.getLeft();
               } else if (rightAttributes.isSubsetOf(keyAttributes) && leftAttributes.isSubsetOf(otherAttributes)) {
                  keyVal = cmpOp.getRight();
               }
               if (keyVal) {
                  if (auto getColOp = mlir::dyn_cast_or_null<tuples::GetColumnOp>(keyVal.getDefiningOp())) {
                     toHash.push_back(getColOp.getAttr());
                  } else {
                     //todo: remove nasty hack:
                     mlir::OpBuilder builder(cmpOp->getContext());
                     builder.setInsertionPointToEnd(mapBlockInfo.block);
                     auto helperOp = builder.create<mlir::arith::ConstantOp>(cmpOp.getLoc(), builder.getIndexAttr(0));

                     relalg::detail::inlineOpIntoBlock(keyVal.getDefiningOp(), keyVal.getDefiningOp()->getParentOp(), mapBlockInfo.block, mapping, helperOp);
                     helperOp->remove();
                     helperOp->destroy();

                     auto& colManager = builder.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
                     auto def = colManager.createDef(colManager.getUniqueScope("join"), "key" + std::to_string(i++));
                     def.getColumn().type = keyVal.getType();
                     auto ref = colManager.createRef(&def.getColumn());
                     mapBlockInfo.createdColumns.push_back(def);
                     mapBlockInfo.results.push_back(mapping.lookupOrNull(keyVal));
                     toHash.push_back(ref);
                     {
                        mlir::OpBuilder builder2(cmpOp->getContext());
                        builder2.setInsertionPointToStart(block);
                        keyVal.replaceAllUsesWith(builder2.create<tuples::GetColumnOp>(builder2.getUnknownLoc(), keyVal.getType(), ref, block->getArgument(0)));
                     }
                  }
                  mlir::OpBuilder builder2(cmpOp->getContext());
                  nullsEqual.push_back(builder2.getI8IntegerAttr(!cmpOp.isEqualityPred(false)));
                  builder2.setInsertionPoint(cmpOp);
                  mlir::Value constTrue = builder2.create<mlir::arith::ConstantIntOp>(builder2.getUnknownLoc(), 1, 1);
                  if (mlir::isa<db::NullableType>(cmpOp->getResult(0).getType())) {
                     constTrue = builder2.create<db::AsNullableOp>(builder2.getUnknownLoc(), cmpOp->getResult(0).getType(), constTrue);
                  }
                  cmpOp->replaceAllUsesWith(mlir::ValueRange{constTrue});
               }
            }
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
      return {toHash, nullsEqual};
   }
};
class OptimizeImplementations : public mlir::PassWrapper<OptimizeImplementations, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-optimize-implementations"; }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizeImplementations)
   bool hashImplPossible(mlir::Block* block, relalg::ColumnSet availableLeft, relalg::ColumnSet availableRight) { //todo: does not work always
      llvm::DenseMap<mlir::Value, relalg::ColumnSet> required;
      relalg::ColumnSet leftKeys, rightKeys;
      std::vector<mlir::Type> types;
      bool res = false;
      block->walk([&](mlir::Operation* op) {
         if (auto getAttr = mlir::dyn_cast_or_null<tuples::GetColumnOp>(op)) {
            required.insert({getAttr.getResult(), relalg::ColumnSet::from(getAttr.getAttr())});
         } else if (auto cmpOp = mlir::dyn_cast_or_null<relalg::CmpOpInterface>(op)) {
            if (cmpOp.isEqualityPred(true) && HashJoinUtils::isAndedResult(op)) {
               auto leftAttributes = required[cmpOp.getLeft()];
               auto rightAttributes = required[cmpOp.getRight()];
               if (leftAttributes.isSubsetOf(availableLeft) && rightAttributes.isSubsetOf(availableRight)) {
                  res = true;
               } else if (leftAttributes.isSubsetOf(availableRight) && rightAttributes.isSubsetOf(availableLeft)) {
                  res = true;
               }
            }
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
      return res;
   }

   // Verify that the join predicate in block takes all and only the primary key columns from baseTableOp into consideration
   bool containsExactlyPrimaryKey(mlir::MLIRContext* ctxt, mlir::Operation* baseTableOp, mlir::Block* block, std::string& indexName) {
      llvm::DenseMap<mlir::Value, relalg::ColumnSet> columns;
      auto baseTable = mlir::cast<relalg::BaseTableOp>(baseTableOp);
      auto& colManager = ctxt->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

      // Initialize map to verify presence of all primary key attributes
      std::unordered_map<std::string, bool> primaryKeyFound;
      for (auto primaryKeyAttribute : baseTable.getMeta().getMeta()->getPrimaryKey()) {
         primaryKeyFound[primaryKeyAttribute] = false;
      }

      // Verify all cmp operations
      bool res = true;
      block->walk([&](mlir::Operation* op) {
         if (auto getAttr = mlir::dyn_cast_or_null<tuples::GetColumnOp>(op)) {
            columns.insert({getAttr.getResult(), relalg::ColumnSet::from(getAttr.getAttr())});
         } else if (auto cmpOp = mlir::dyn_cast_or_null<relalg::CmpOpInterface>(op)) {
            relalg::ColumnSet relevantColumns = columns[cmpOp.getLeft()];
            relevantColumns.insert(columns[cmpOp.getRight()]);
            for (auto* relevantColumn : relevantColumns) {
               std::string tableName = colManager.getName(relevantColumn).first;
               std::string columnName = colManager.getName(relevantColumn).second;

               // Only take columns contained in baseTableOp into consideration
               if (baseTable.getCreatedColumns().contains(relevantColumn)) {
                  // Check that no non-primary key attribute was used
                  if (!primaryKeyFound.contains(columnName)) res = false;
                  // Mark primary key attribute as used
                  else
                     primaryKeyFound[columnName] = true;
               }
            }
         }
      });
      // Check if all primary key attributes were found
      for (auto primaryKeyAttribute : primaryKeyFound) {
         res &= primaryKeyAttribute.second;
      }
      indexName = "pk_hash";
      return res;
   }

   bool isBaseRelationWithSelects(Operator op, std::stack<mlir::Operation*>& path) {
      // Saves operations until base relation is reached on stack for easy access
      return ::llvm::TypeSwitch<mlir::Operation*, bool>(op.getOperation())
         .Case<relalg::BaseTableOp>([&](relalg::BaseTableOp baseTableOp) {
            path.push(baseTableOp.getOperation());
            return true;
         })
         .Case<relalg::SelectionOp>([&](relalg::SelectionOp selectionOp) {
            path.push(selectionOp.getOperation());
            for (auto& child : selectionOp.getChildren()) {
               if (!isBaseRelationWithSelects(mlir::cast<Operator>(child.getOperation()), path)) return false;
            }
            return true;
         })
         .Default([&](auto&) {
            return false;
         });
   }

   void prepareForHash(PredicateOperator predicateOperator) {
      auto binOp = mlir::cast<BinaryOperator>(predicateOperator.getOperation());
      auto left = mlir::cast<Operator>(binOp.leftChild());
      auto right = mlir::cast<Operator>(binOp.rightChild());
      //left
      {
         mlir::OpBuilder builder(&this->getContext());
         HashJoinUtils::MapBlockInfo mapBlockInfo;
         mapBlockInfo.block = new mlir::Block;
         mapBlockInfo.block->addArgument(tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
         auto [keys, nullsEqual] = HashJoinUtils::extractKeys(&predicateOperator.getPredicateBlock(), left.getAvailableColumns(), right.getAvailableColumns(), mapBlockInfo);
         if (!mapBlockInfo.createdColumns.empty()) {
            builder.setInsertionPoint(predicateOperator);
            auto mapOp = builder.create<relalg::MapOp>(builder.getUnknownLoc(), tuples::TupleStreamType::get(builder.getContext()), left.asRelation(), builder.getArrayAttr(mapBlockInfo.createdColumns));
            mapOp.getPredicate().push_back(mapBlockInfo.block);
            mlir::OpBuilder builder2(builder.getContext());
            builder2.setInsertionPointToEnd(mapBlockInfo.block);
            builder2.create<tuples::ReturnOp>(builder2.getUnknownLoc(), mapBlockInfo.results);
            left = mapOp;
         }
         predicateOperator->setAttr("leftHash", builder.getArrayAttr(keys));
         predicateOperator->setAttr("nullsEqual", builder.getArrayAttr(nullsEqual));
      }

      //right
      {
         mlir::OpBuilder builder(&this->getContext());
         HashJoinUtils::MapBlockInfo mapBlockInfo;
         mapBlockInfo.block = new mlir::Block;
         mapBlockInfo.block->addArgument(tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
         auto [keys, nullEquals] = HashJoinUtils::extractKeys(&predicateOperator.getPredicateBlock(), right.getAvailableColumns(), left.getAvailableColumns(), mapBlockInfo);
         if (!mapBlockInfo.createdColumns.empty()) {
            builder.setInsertionPoint(predicateOperator);
            auto mapOp = builder.create<relalg::MapOp>(builder.getUnknownLoc(), tuples::TupleStreamType::get(builder.getContext()), right.asRelation(), builder.getArrayAttr(mapBlockInfo.createdColumns));
            mapOp.getPredicate().push_back(mapBlockInfo.block);
            mlir::OpBuilder builder2(builder.getContext());
            builder2.setInsertionPointToEnd(mapBlockInfo.block);
            builder2.create<tuples::ReturnOp>(builder2.getUnknownLoc(), mapBlockInfo.results);
            right = mapOp;
         }
         predicateOperator->setAttr("rightHash", builder.getArrayAttr(keys));
      }
      mlir::cast<Operator>(predicateOperator.getOperation()).setChildren({left, right});
   }
   static mlir::Value mapColsToNullable(mlir::Value stream, mlir::OpBuilder& rewriter, mlir::Location loc, mlir::ArrayAttr mapping, size_t exisingOffset = 0, relalg::ColumnSet excluded = {}) {
      auto& colManager = rewriter.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      std::vector<mlir::Attribute> defAttrs;
      auto* mapBlock = new mlir::Block;
      auto tupleArg = mapBlock->addArgument(tuples::TupleType::get(rewriter.getContext()), loc);
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(mapBlock);
         std::vector<mlir::Value> res;
         for (mlir::Attribute attr : mapping) {
            auto relationDefAttr = mlir::dyn_cast_or_null<tuples::ColumnDefAttr>(attr);
            auto* defAttr = &relationDefAttr.getColumn();
            auto fromExisting = mlir::cast<tuples::ColumnRefAttr>(mlir::cast<mlir::ArrayAttr>(relationDefAttr.getFromExisting())[exisingOffset]);
            if (excluded.contains(&fromExisting.getColumn())) continue;
            mlir::Value value = rewriter.create<tuples::GetColumnOp>(loc, rewriter.getI64Type(), fromExisting, tupleArg);
            if (fromExisting.getColumn().type != defAttr->type) {
               mlir::Value tmp = rewriter.create<db::AsNullableOp>(loc, defAttr->type, value);
               value = tmp;
            }
            res.push_back(value);
            defAttrs.push_back(colManager.createDef(defAttr));
         }
         rewriter.create<tuples::ReturnOp>(loc, res);
      }
      auto mapOp = rewriter.create<relalg::MapOp>(loc, tuples::TupleStreamType::get(rewriter.getContext()), stream, rewriter.getArrayAttr(defAttrs));
      mapOp.getPredicate().push_back(mapBlock);
      return mapOp.getResult();
   }

   size_t estimatedEvaluationCost(mlir::Value v) {
      if (auto* definingOp = v.getDefiningOp()) {
         return llvm::TypeSwitch<mlir::Operation*, size_t>(definingOp)
            .Case([&](db::ConstantOp) {
               return 0;
            })
            .Case([&](db::OrOp orOp) {
               size_t res = orOp.getVals().size();
               for (auto val : orOp.getVals()) {
                  res += estimatedEvaluationCost(val);
               }
               return res;
            })
            .Case([&](tuples::GetColumnOp& getColumnOp) {
               if (mlir::isa<db::StringType>(getBaseType(getColumnOp.getType()))) {
                  return 4;
               } else if (mlir::isa<db::DecimalType>(getBaseType(getColumnOp.getType()))) {
                  return 2;
               } else {
                  return 1;
               }
            })
            .Case([&](relalg::CmpOpInterface cmpOp) {
               auto t = cmpOp.getLeft().getType();
               auto childCost = estimatedEvaluationCost(cmpOp.getLeft()) + estimatedEvaluationCost(cmpOp.getRight());
               if (mlir::isa<db::StringType>(getBaseType(t))) {
                  return 10 + childCost;
               } else {
                  return 1 + childCost;
               }
            })
            .Case([&](db::BetweenOp cmpOp) {
               auto t = cmpOp.getLower().getType();
               auto childCost = estimatedEvaluationCost(cmpOp.getVal()) + estimatedEvaluationCost(cmpOp.getLower()) + estimatedEvaluationCost(cmpOp.getUpper());
               if (mlir::isa<db::StringType>(getBaseType(t))) {
                  return 20 + childCost;
               } else {
                  return 2 + childCost;
               }
            })
            .Default([&](mlir::Operation* op) {
               return 1000;
            });
      } else {
         return 10000;
      }
   }
   static relalg::ColumnSet getRequired(Operator op) {
      auto available = op.getAvailableColumns();

      relalg::ColumnSet required;
      for (auto* user : op->getUsers()) {
         if (auto consumingOp = mlir::dyn_cast_or_null<Operator>(user)) {
            required.insert(getRequired(consumingOp));
            required.insert(consumingOp.getUsedColumns());
         }
         if (auto materializeOp = mlir::dyn_cast_or_null<relalg::MaterializeOp>(user)) {
            required.insert(relalg::ColumnSet::fromArrayAttr(materializeOp.getCols()));
         }
      }
      return available.intersect(required);
   }

   void runOnOperation() override {
      std::vector<mlir::Operation*> toErase;
      getOperation().walk([&](Operator op) {
         ::llvm::TypeSwitch<mlir::Operation*, void>(op.getOperation())
            .Case<relalg::SelectionOp>([&](relalg::SelectionOp selectionOp) {
               std::unordered_set<mlir::Operation*> users(selectionOp->getUsers().begin(), selectionOp->getUsers().end());
               if (users.empty()) return;
               bool reorder = users.size() > 1 || !mlir::isa<relalg::SelectionOp>(*users.begin());
               if (!reorder) return;
               std::vector<relalg::SelectionOp> selections;
               selections.push_back(selectionOp);
               relalg::SelectionOp currentSelection = selectionOp;
               while (currentSelection) {
                  Operator child = currentSelection.getChildren()[0];
                  if (std::vector<mlir::Operation*>(child->getUsers().begin(), child->getUsers().end()).size() > 1) {
                     child = {};
                  }
                  currentSelection = mlir::dyn_cast_or_null<relalg::SelectionOp>(child.getOperation());
                  if (currentSelection) {
                     selections.push_back(currentSelection);
                  }
               }
               auto firstStream = selections[selections.size() - 1].getRel();
               relalg::BaseTableOp baseTableOp = mlir::dyn_cast_or_null<relalg::BaseTableOp>(selections[selections.size() - 1].getRel().getDefiningOp());
               if (baseTableOp) {
                  std::unordered_map<const tuples::Column*, std::string> mapping;
                  for (auto c : baseTableOp.getColumns()) {
                     mapping[&mlir::cast<tuples::ColumnDefAttr>(c.getValue()).getColumn()] = c.getName().str();
                  }
                  auto meta = baseTableOp.getMeta().getMeta();
                  auto sample = meta->getSample();
                  if (sample) {
                     for (auto selOp : selections) {
                        auto v = mlir::cast<tuples::ReturnOp>(selOp.getPredicateBlock().getTerminator()).getResults()[0];
                        auto expr = relalg::buildEvalExpr(v, mapping);
                        auto optionalCount = lingodb::compiler::support::eval::countResults(sample, std::move(expr));
                        if (optionalCount) {
                           auto count = optionalCount.value();
                           if (count == 0) count = 1;
                           double selectivity = static_cast<double>(count) / static_cast<double>(sample->num_rows());
                           selOp->setAttr("selectivity", mlir::FloatAttr::get(mlir::Float64Type::get(&getContext()), selectivity));
                        }
                     }
                  }
               }
               for (auto selOp : selections) {
                  auto v = mlir::cast<tuples::ReturnOp>(selOp.getPredicateBlock().getTerminator()).getResults()[0];
                  double evaluationCost = estimatedEvaluationCost(v);
                  selOp->setAttr("evaluationCost", mlir::FloatAttr::get(mlir::Float64Type::get(&getContext()), evaluationCost));
               }
               if (selections.size() > 1) {
                  std::vector<relalg::SelectionOp> finalOrder;
                  std::vector<std::pair<double, relalg::SelectionOp>> toProcess;
                  for (auto sel : selections) {
                     double selectivity = sel->hasAttr("selectivity") ? sel->getAttrOfType<mlir::FloatAttr>("selectivity").getValueAsDouble() : 1;
                     toProcess.push_back({selectivity, sel});
                  }
                  std::sort(toProcess.begin(), toProcess.end(), [](auto a, auto b) { return a.first < b.first; });
                  for (size_t i = 0; i < toProcess.size() - 1; i++) {
                     auto& first = toProcess[i];
                     auto& second = toProcess[i + 1];
                     auto s1 = first.first;
                     auto s2 = second.first;
                     auto c1 = first.second->getAttrOfType<mlir::FloatAttr>("evaluationCost").getValueAsDouble();
                     auto c2 = second.second->getAttrOfType<mlir::FloatAttr>("evaluationCost").getValueAsDouble();
                     if (c1 + s1 * c2 > c2 + s2 * c1) {
                        std::swap(first, second);
                     }
                  }
                  mlir::Value stream = firstStream;
                  for (auto sel : toProcess) {
                     sel.second->moveAfter(stream.getDefiningOp());
                     sel.second.setOperand(stream);
                     stream = sel.second;
                  }
                  selectionOp.getResult().replaceUsesWithIf(stream, [&](auto& use) {
                     return users.contains(use.getOwner());
                  });
               }
            })
            .Case<relalg::LimitOp>([&](relalg::LimitOp limitOp) {
               if (auto sortOp = mlir::dyn_cast_or_null<relalg::SortOp>(limitOp.getRel().getDefiningOp())) {
                  mlir::OpBuilder builder(limitOp);
                  toErase.push_back(limitOp);
                  toErase.push_back(sortOp);

                  limitOp.replaceAllUsesWith(builder.create<relalg::TopKOp>(limitOp.getLoc(), limitOp.getMaxRows(), sortOp.getRel(), sortOp.getSortspecs()).asRelation());
               }
            })
            .Case<relalg::InnerJoinOp, relalg::CollectionJoinOp, relalg::FullOuterJoinOp>([&](PredicateOperator predicateOperator) {
               auto binOp = mlir::cast<BinaryOperator>(predicateOperator.getOperation());
               auto left = mlir::cast<Operator>(binOp.leftChild());
               auto right = mlir::cast<Operator>(binOp.rightChild());
               if (hashImplPossible(&predicateOperator.getPredicateBlock(), left.getAvailableColumns(), right.getAvailableColumns())) {
                  // Determine if index nested loop is possible and is beneficial
                  std::stack<mlir::Operation*> leftPath, rightPath;
                  std::string leftIndexName, rightIndexName;
                  bool leftCanUsePrimaryKeyIndex = isBaseRelationWithSelects(left, leftPath) && containsExactlyPrimaryKey(binOp.getContext(), leftPath.top(), &predicateOperator.getPredicateBlock(), leftIndexName);
                  bool rightCanUsePrimaryKeyIndex = isBaseRelationWithSelects(right, rightPath) && containsExactlyPrimaryKey(binOp.getContext(), rightPath.top(), &predicateOperator.getPredicateBlock(), rightIndexName);
                  bool isInnerJoin = mlir::isa<relalg::InnerJoinOp>(predicateOperator);
                  bool reversed = false;

                  prepareForHash(predicateOperator);

                  // Select possible build side to the left
                  if (isInnerJoin && (leftCanUsePrimaryKeyIndex || rightCanUsePrimaryKeyIndex)) {
                     if (leftCanUsePrimaryKeyIndex && rightCanUsePrimaryKeyIndex) {
                        // Compute heuristic of which base table the index is more beneficial
                        // Used heuristic: prefer bigger ratio of |buildSide| / |probeSide|
                        auto leftBaseTable = mlir::cast<relalg::BaseTableOp>(leftPath.top());
                        auto rightBaseTable = mlir::cast<relalg::BaseTableOp>(rightPath.top());
                        int numBaseRowsLeft = leftBaseTable.getMeta().getMeta()->getNumRows() + 1;
                        int numBaseRowsRight = rightBaseTable.getMeta().getMeta()->getNumRows() + 1;
                        int numNonBaseRowsLeft = left->hasAttr("rows") ? mlir::dyn_cast_or_null<mlir::FloatAttr>(left->getAttr("rows")).getValueAsDouble() + 1 : 1;
                        int numNonBaseRowsRight = right->hasAttr("rows") ? mlir::dyn_cast_or_null<mlir::FloatAttr>(right->getAttr("rows")).getValueAsDouble() + 1 : 1;
                        // Exchange left and right side if deemed beneficial by heuristic
                        if (numNonBaseRowsRight / numBaseRowsLeft < numNonBaseRowsLeft / numBaseRowsRight) {
                           reversed = true;
                           std::swap(left, right);
                           std::swap(leftPath, rightPath);
                           std::swap(leftIndexName, rightIndexName);
                           mlir::Attribute tmp = predicateOperator->getAttr("rightHash");
                           predicateOperator->setAttr("rightHash", predicateOperator->getAttr("leftHash"));
                           predicateOperator->setAttr("leftHash", tmp);
                        }
                     } else if (!leftCanUsePrimaryKeyIndex) {
                        // Exchange left and right side
                        reversed = true;
                        std::swap(left, right);
                        std::swap(leftPath, rightPath);
                        std::swap(leftIndexName, rightIndexName);
                        leftCanUsePrimaryKeyIndex = true;
                        mlir::Attribute tmp = predicateOperator->getAttr("rightHash");
                        predicateOperator->setAttr("rightHash", predicateOperator->getAttr("leftHash"));
                        predicateOperator->setAttr("leftHash", tmp);
                     }
                  }

                  // Compute correct number of rows for index nested loop join
                  double numRowsLeft = 0, numRowsRight = std::numeric_limits<double>::max(); // default: disable inlj
                  if (auto leftCardinalityAttr = mlir::dyn_cast_or_null<mlir::FloatAttr>(left->getAttr("rows"))) {
                     numRowsLeft = leftCardinalityAttr.getValueAsDouble();
                  }
                  if (auto rightCardinalityAttr = mlir::dyn_cast_or_null<mlir::FloatAttr>(right->getAttr("rows"))) {
                     numRowsRight = rightCardinalityAttr.getValueAsDouble();
                  }
                  if (isInnerJoin && leftCanUsePrimaryKeyIndex && right->hasAttr("rows") && 20 * numRowsRight < numRowsLeft) {
                     // base relations do not need to be moved
                     auto leftBaseTable = mlir::cast<relalg::BaseTableOp>(leftPath.top());
                     leftPath.pop();

                     // update binOp
                     binOp->setOperands(mlir::ValueRange{leftBaseTable, binOp->getOperand(!reversed)});
                     mlir::Operation* lastMoved = binOp.getOperation();
                     mlir::Operation* firstMoved = nullptr;

                     mlir::OpBuilder builder(binOp);

                     // Move selections on left side after join
                     while (!leftPath.empty()) {
                        if (!firstMoved) firstMoved = leftPath.top();
                        leftPath.top()->moveAfter(lastMoved);
                        leftPath.top()->setOperands(mlir::ValueRange{lastMoved->getResult(0)});
                        lastMoved = leftPath.top();
                        leftPath.pop();
                     }

                     // If selections were moved, replace usages of join with last moved selection
                     if (firstMoved) {
                        binOp->replaceAllUsesWith(mlir::ValueRange{lastMoved->getResults()});
                        firstMoved->setOperands(binOp->getResults());
                     }
                     leftBaseTable->setAttr("virtual", mlir::UnitAttr::get(&getContext()));

                     // Add name of table to leftHash annotation
                     std::vector<mlir::Attribute> leftHash;
                     leftHash.push_back(leftBaseTable.getTableIdentifierAttr());
                     for (auto attr : mlir::dyn_cast_or_null<mlir::ArrayAttr>(op->getAttr("leftHash"))) {
                        leftHash.push_back(attr);
                     }
                     op->setAttr("leftHash", mlir::ArrayAttr::get(&getContext(), leftHash));

                     op->setAttr("impl", mlir::StringAttr::get(op.getContext(), "indexNestedLoop"));
                     op->setAttr("useIndexNestedLoop", mlir::UnitAttr::get(op.getContext()));
                     op->setAttr("index", mlir::StringAttr::get(op.getContext(), leftIndexName));
                  } else {
                     op->setAttr("impl", mlir::StringAttr::get(op.getContext(), "hash"));
                     op->setAttr("useHashJoin", mlir::UnitAttr::get(op.getContext()));
                     prepareForHash(predicateOperator);
                  }
               }
            })
            .Case<relalg::SemiJoinOp, relalg::AntiSemiJoinOp, relalg::OuterJoinOp, relalg::MarkJoinOp>([&](PredicateOperator predicateOperator) {
               auto binOp = mlir::cast<BinaryOperator>(predicateOperator.getOperation());
               auto left = mlir::cast<Operator>(binOp.leftChild());
               auto right = mlir::cast<Operator>(binOp.rightChild());
               if (left->hasAttr("rows") && right->hasAttr("rows")) {
                  double rowsLeft = 0;
                  double rowsRight = 0;
                  if (auto lDAttr = mlir::dyn_cast_or_null<mlir::FloatAttr>(left->getAttr("rows"))) {
                     rowsLeft = lDAttr.getValueAsDouble();
                  } else if (auto lIAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(left->getAttr("rows"))) {
                     rowsLeft = lIAttr.getInt();
                  }
                  if (auto rDAttr = mlir::dyn_cast_or_null<mlir::FloatAttr>(right->getAttr("rows"))) {
                     rowsRight = rDAttr.getValueAsDouble();
                  } else if (auto rIAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(right->getAttr("rows"))) {
                     rowsRight = rIAttr.getInt();
                  }
                  if (rowsLeft < rowsRight) {
                     op->setAttr("reverseSides", mlir::UnitAttr::get(op.getContext()));
                  }
               }
               if (hashImplPossible(&predicateOperator.getPredicateBlock(), left.getAvailableColumns(), right.getAvailableColumns())) {
                  op->setAttr("useHashJoin", mlir::UnitAttr::get(op.getContext()));
                  prepareForHash(predicateOperator);
                  if (left->hasAttr("rows") && right->hasAttr("rows")) {
                     double rowsLeft = 0;
                     double rowsRight = 0;
                     if (auto lDAttr = mlir::dyn_cast_or_null<mlir::FloatAttr>(left->getAttr("rows"))) {
                        rowsLeft = lDAttr.getValueAsDouble();
                     } else if (auto lIAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(left->getAttr("rows"))) {
                        rowsLeft = lIAttr.getInt();
                     }
                     if (auto rDAttr = mlir::dyn_cast_or_null<mlir::FloatAttr>(right->getAttr("rows"))) {
                        rowsRight = rDAttr.getValueAsDouble();
                     } else if (auto rIAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(right->getAttr("rows"))) {
                        rowsRight = rIAttr.getInt();
                     }
                     if (rowsLeft < rowsRight) {
                        op->setAttr("impl", mlir::StringAttr::get(op.getContext(), "markhash"));
                     } else {
                        op->setAttr("impl", mlir::StringAttr::get(op.getContext(), "hash"));
                     }
                  } else {
                     op->setAttr("impl", mlir::StringAttr::get(op.getContext(), "hash"));
                  }
               }
            })
            .Case<relalg::SingleJoinOp>([&](relalg::SingleJoinOp op) {
               if (auto returnOp = mlir::dyn_cast_or_null<tuples::ReturnOp>(op.getPredicateBlock().getTerminator())) {
                  if (returnOp.getResults().empty()) {
                     op->setAttr("impl", mlir::StringAttr::get(op.getContext(), "constant"));
                     op->setAttr("constantJoin", mlir::UnitAttr::get(op.getContext()));
                  }
               }
               auto left = mlir::cast<Operator>(op.leftChild());
               auto right = mlir::cast<Operator>(op.rightChild());
               if (hashImplPossible(&op.getPredicateBlock(), left.getAvailableColumns(), right.getAvailableColumns())) {
                  prepareForHash(op);
                  op->setAttr("useHashJoin", mlir::UnitAttr::get(op.getContext()));
                  op->setAttr("impl", mlir::StringAttr::get(op.getContext(), "hash"));
               }
            })

            .Case<relalg::OuterJoinOp>([&](relalg::OuterJoinOp op) {})
            .Case<relalg::FullOuterJoinOp>([&](relalg::FullOuterJoinOp op) {})

            .Default([&](auto x) {
            });
      });
      for (auto* op : toErase) {
         op->erase();
      }
      toErase.clear();
      getOperation().walk([&](relalg::InnerJoinOp op) {
         auto usedColumns = op.getUsedColumns();
         if (!op->hasAttr("leftHash") || !op->hasAttr("rightHash")) return;
         auto leftKeys = mlir::cast<mlir::ArrayAttr>(op->getAttr("leftHash"));
         auto rightKeys = mlir::cast<mlir::ArrayAttr>(op->getAttr("rightHash"));
         auto leftKeySet = relalg::ColumnSet::fromArrayAttr(leftKeys);
         auto rightKeySet = relalg::ColumnSet::fromArrayAttr(rightKeys);
         relalg::ColumnSet reallyRequiredColumns;
         reallyRequiredColumns.insert(rightKeySet);
         auto currentChild = op.getRight();
         std::vector<mlir::Operation*> otherOps;
         bool needsSplit = false;

         while (true) {
            if (auto aggrOp = mlir::dyn_cast_or_null<relalg::AggregationOp>(currentChild.getDefiningOp())) {
               if (relalg::ColumnSet::fromArrayAttr(aggrOp.getComputedCols()).intersects(reallyRequiredColumns)) {
                  return;
               } else {
                  needsSplit |= relalg::ColumnSet::fromArrayAttr(aggrOp.getComputedCols()).intersects(usedColumns);
                  auto groupByKeySet = relalg::ColumnSet::fromArrayAttr(aggrOp.getGroupByCols());
                  if (groupByKeySet.size() != leftKeySet.size()) return;
                  groupByKeySet.remove(leftKeySet);
                  groupByKeySet.remove(rightKeySet);
                  if (!groupByKeySet.empty()) return;

                  auto fds = mlir::cast<Operator>(op.getLeft().getDefiningOp()).getFDs();
                  if (!fds.isDuplicateFreeKey(leftKeySet)) return;
                  std::vector<mlir::Operation*> users(op->getUsers().begin(), op->getUsers().end());
                  if (users.size() != 1) {
                     return;
                  }
                  auto required = getRequired(op);
                  mlir::Operation* moveBefore = users[0];
                  for (auto* o : otherOps) {
                     o->moveBefore(moveBefore);
                     moveBefore = o;
                  }
                  auto topRightVal = op.getRight();
                  op.getResult().replaceAllUsesWith(topRightVal);
                  op.setOperand(1, aggrOp.getRel());
                  aggrOp->moveAfter(op);
                  aggrOp->setOperand(0, op.getResult());
                  if (needsSplit) {
                     mlir::OpBuilder builder(&getContext());
                     builder.setInsertionPointAfter(topRightVal.getDefiningOp());
                     auto sel = builder.create<relalg::SelectionOp>(op->getLoc(), topRightVal);
                     topRightVal.replaceAllUsesExcept(sel.asRelation(), sel);
                     sel.getPredicate().takeBody(op.getPredicate());
                     mlir::Block* newJoinPred = new mlir::Block;
                     newJoinPred->addArgument(tuples::TupleType::get(builder.getContext()), op.getLoc());
                     builder.setInsertionPointToStart(newJoinPred);
                     mlir::Value trueVal = builder.create<mlir::arith::ConstantIntOp>(op.getLoc(), 1, 1);
                     builder.create<tuples::ReturnOp>(op.getLoc(), trueVal);
                     op.getPredicate().push_back(newJoinPred);
                     required.insert(sel.getUsedColumns());
                  }
                  auto available = mlir::cast<Operator>(topRightVal.getDefiningOp()).getAvailableColumns();
                  required.remove(available);
                  //llvm::dbgs() << "still required:";
                  //required.dump(&getContext());
                  if (!required.empty()) {
                     mlir::OpBuilder builder(&getContext());
                     auto previousTerminator = mlir::cast<tuples::ReturnOp>(aggrOp.getAggrFunc().front().getTerminator());
                     std::vector<mlir::Value> aggregates(previousTerminator.getResults().begin(), previousTerminator.getResults().end());
                     builder.setInsertionPointToEnd(&aggrOp.getAggrFunc().front());
                     std::vector<mlir::Attribute> computedCols(aggrOp.getComputedCols().begin(), aggrOp.getComputedCols().end());
                     auto& colManager = getContext().getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
                     std::vector<mlir::Attribute> renaming;
                     for (auto* column : required) {
                        auto* newCol = colManager.get(colManager.getUniqueScope("moved_aggr"), colManager.getName(column).second).get();
                        newCol->type = column->type;
                        renaming.push_back(colManager.createDef(column, builder.getArrayAttr({colManager.createRef(newCol)})));
                        computedCols.push_back(colManager.createDef(newCol));
                        aggregates.push_back(builder.create<relalg::AggrFuncOp>(aggrOp->getLoc(), column->type, relalg::AggrFunc::any, aggrOp.getAggrFunc().getArgument(0), colManager.createRef(column)));
                     }
                     builder.create<tuples::ReturnOp>(aggrOp.getLoc(), aggregates);
                     previousTerminator->erase();
                     builder.setInsertionPointAfter(aggrOp);
                     auto renamed = builder.create<relalg::RenamingOp>(aggrOp.getLoc(), aggrOp.getResult(), builder.getArrayAttr(renaming));
                     aggrOp.getResult().replaceAllUsesExcept(renamed.getResult(), renamed);
                     aggrOp.setComputedColsAttr(builder.getArrayAttr(computedCols));
                  }
                  return;
               }
            } else if (auto selOp = mlir::dyn_cast_or_null<relalg::SelectionOp>(currentChild.getDefiningOp())) {
               otherOps.push_back(selOp.getOperation());
               currentChild = selOp.getRel();
            } else if (auto mapOp = mlir::dyn_cast_or_null<relalg::MapOp>(currentChild.getDefiningOp())) {
               if (relalg::ColumnSet::fromArrayAttr(mapOp.getComputedCols()).intersects(reallyRequiredColumns)) {
                  return;
               } else {
                  needsSplit |= relalg::ColumnSet::fromArrayAttr(mapOp.getComputedCols()).intersects(usedColumns);
                  otherOps.push_back(mapOp.getOperation());
                  currentChild = mapOp.getRel();
               }
            } else {
               return;
            }
         }
      });
      bool enableGroupJoins = false;
      //todo: reactivate groupjoins
      if (enableGroupJoins) {
         getOperation().walk([&](relalg::AggregationOp op) {
            auto* potentialJoin = op.getRel().getDefiningOp();
            relalg::MapOp mapOp = mlir::dyn_cast_or_null<relalg::MapOp>(potentialJoin);
            relalg::ColumnSet usedColumns = op.getUsedColumns();
            if (mapOp) {
               usedColumns.insert(mapOp.getUsedColumns());
               potentialJoin = mapOp.getRel().getDefiningOp();
            }

            auto isInnerJoin = mlir::isa<relalg::InnerJoinOp>(potentialJoin);
            auto isOuterJoin = mlir::isa<relalg::OuterJoinOp>(potentialJoin);
            if (!isInnerJoin && !isOuterJoin) return;
            PredicateOperator join = mlir::cast<PredicateOperator>(potentialJoin);
            Operator joinOperator = mlir::cast<Operator>(potentialJoin);
            usedColumns.insert(joinOperator.getUsedColumns());
            if (!join->hasAttr("useHashJoin") || !join->hasAttr("leftHash") || !join->hasAttr("rightHash")) return;
            auto leftKeys = mlir::cast<mlir::ArrayAttr>(join->getAttr("leftHash"));
            auto rightKeys = mlir::cast<mlir::ArrayAttr>(join->getAttr("rightHash"));
            for (auto p : llvm::zip(leftKeys, rightKeys)) {
               auto t1 = mlir::cast<tuples::ColumnRefAttr>(std::get<0>(p)).getColumn().type;
               auto t2 = mlir::cast<tuples::ColumnRefAttr>(std::get<1>(p)).getColumn().type;
               if (t1 != t2) return;
            }
            auto leftKeySet = relalg::ColumnSet::fromArrayAttr(leftKeys);
            auto rightKeySet = relalg::ColumnSet::fromArrayAttr(rightKeys);
            auto groupByKeySet = relalg::ColumnSet::fromArrayAttr(op.getGroupByCols());
            if (groupByKeySet.size() != leftKeySet.size()) return;
            groupByKeySet.remove(leftKeySet);
            groupByKeySet.remove(rightKeySet);
            if (!groupByKeySet.empty()) return;

            auto leftChild = joinOperator.getChildren()[0];
            auto rightChild = joinOperator.getChildren()[1];
            //auto leftUsedColumns = usedColumns.intersect(leftChild.getAvailableColumns());
            auto fds = leftChild.getFDs();
            if (!fds.isDuplicateFreeKey(leftKeySet)) return;
            bool containsProjection = false;
            bool containsCountRows = false;
            op.walk([&](relalg::ProjectionOp) { containsProjection = true; });
            op.walk([&](relalg::CountRowsOp) { containsCountRows = true; });
            if (containsProjection || (isOuterJoin && containsCountRows)) return;
            mlir::OpBuilder builder(op);
            mlir::ArrayAttr mappedCols = mapOp ? mapOp.getComputedCols() : builder.getArrayAttr({});
            mlir::Value left = leftChild.asRelation();
            mlir::Value right = rightChild.asRelation();
            if (isOuterJoin) {
               auto outerJoin = mlir::cast<relalg::OuterJoinOp>(potentialJoin);
               right = mapColsToNullable(right, builder, op.getLoc(), outerJoin.getMapping());
            }
            llvm::dbgs() << "introducing groupjoin\n";
            auto groupJoinOp = builder.create<relalg::GroupJoinOp>(op.getLoc(), left, right, isOuterJoin ? relalg::GroupJoinBehavior::outer : relalg::GroupJoinBehavior::inner, leftKeys, rightKeys, mappedCols, op.getComputedCols());
            if (mapOp) {
               mlir::IRMapping mapping;
               mapOp.getPredicate().cloneInto(&groupJoinOp.getMapFunc(), mapping);
            } else {
               auto* b = new mlir::Block;
               mlir::OpBuilder mB(&getContext());
               groupJoinOp.getMapFunc().push_back(b);
               mB.setInsertionPointToStart(b);
               mB.create<tuples::ReturnOp>(op.getLoc());
            }
            {
               mlir::IRMapping mapping;
               op.getAggrFunc().cloneInto(&groupJoinOp.getAggrFunc(), mapping);
            }
            {
               mlir::IRMapping mapping;
               join.getPredicateRegion().cloneInto(&groupJoinOp.getPredicate(), mapping);
            }
            op->replaceAllUsesWith(groupJoinOp);
            toErase.push_back(op.getOperation());
            if (mapOp) {
               toErase.push_back(mapOp);
            }
            toErase.push_back(join.getOperation());
         });
      }
      for (auto* op : toErase) {
         op->erase();
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> relalg::createOptimizeImplementationsPass() { return std::make_unique<OptimizeImplementations>(); }
