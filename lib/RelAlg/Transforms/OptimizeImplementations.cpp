#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace {
class HashJoinUtils {
   public:
   static bool isAndedResult(mlir::Operation* op, bool first = true) {
      if (mlir::isa<mlir::tuples::ReturnOp>(op)) {
         return true;
      }
      if (mlir::isa<mlir::db::AndOp>(op) || first) {
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
   static std::pair<std::vector<mlir::Attribute>, std::vector<mlir::Attribute>> extractKeys(mlir::Block* block, mlir::relalg::ColumnSet keyAttributes, mlir::relalg::ColumnSet otherAttributes, MapBlockInfo& mapBlockInfo) {
      std::vector<mlir::Attribute> toHash;
      std::vector<mlir::Attribute> nullsEqual;
      llvm::DenseMap<mlir::Value, mlir::relalg::ColumnSet> required;
      mlir::BlockAndValueMapping mapping;
      mapping.map(block->getArgument(0), mapBlockInfo.block->getArgument(0));
      size_t i = 0;
      block->walk([&](mlir::Operation* op) {
         if (auto getAttr = mlir::dyn_cast_or_null<mlir::tuples::GetColumnOp>(op)) {
            required.insert({getAttr.getResult(), mlir::relalg::ColumnSet::from(getAttr.getAttr())});
         } else if (auto cmpOp = mlir::dyn_cast_or_null<mlir::relalg::CmpOpInterface>(op)) {
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
                  if (auto getColOp = mlir::dyn_cast_or_null<mlir::tuples::GetColumnOp>(keyVal.getDefiningOp())) {
                     toHash.push_back(getColOp.getAttr());
                  } else {
                     //todo: remove nasty hack:
                     mlir::OpBuilder builder(cmpOp->getContext());
                     builder.setInsertionPointToEnd(mapBlockInfo.block);
                     auto helperOp = builder.create<mlir::arith::ConstantOp>(cmpOp.getLoc(), builder.getIndexAttr(0));

                     mlir::relalg::detail::inlineOpIntoBlock(keyVal.getDefiningOp(), keyVal.getDefiningOp()->getParentOp(), mapBlockInfo.block, mapping, helperOp);
                     helperOp->remove();
                     helperOp->destroy();

                     auto& colManager = builder.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
                     auto def = colManager.createDef(colManager.getUniqueScope("join"), "key" + std::to_string(i++));
                     def.getColumn().type = keyVal.getType();
                     auto ref = colManager.createRef(&def.getColumn());
                     mapBlockInfo.createdColumns.push_back(def);
                     mapBlockInfo.results.push_back(mapping.lookupOrNull(keyVal));
                     toHash.push_back(ref);
                     {
                        mlir::OpBuilder builder2(cmpOp->getContext());
                        builder2.setInsertionPointToStart(block);
                        keyVal.replaceAllUsesWith(builder2.create<mlir::tuples::GetColumnOp>(builder2.getUnknownLoc(), keyVal.getType(), ref, block->getArgument(0)));
                     }
                  }
                  mlir::OpBuilder builder2(cmpOp->getContext());
                  nullsEqual.push_back(builder2.getI8IntegerAttr(!cmpOp.isEqualityPred(false)));
                  builder2.setInsertionPoint(cmpOp);
                  mlir::Value constTrue = builder2.create<mlir::arith::ConstantIntOp>(builder2.getUnknownLoc(), 1, 1);
                  if (cmpOp->getResult(0).getType().isa<mlir::db::NullableType>()) {
                     constTrue = builder2.create<mlir::db::AsNullableOp>(builder2.getUnknownLoc(), cmpOp->getResult(0).getType(), constTrue);
                  }
                  cmpOp->replaceAllUsesWith(mlir::ValueRange{constTrue});
               }
            }
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
      return {toHash, nullsEqual};
   }
};
class OptimizeImplementations : public mlir::PassWrapper<OptimizeImplementations, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-optimize-implementations"; }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizeImplementations)
   bool hashImplPossible(mlir::Block* block, mlir::relalg::ColumnSet availableLeft, mlir::relalg::ColumnSet availableRight) { //todo: does not work always
      llvm::DenseMap<mlir::Value, mlir::relalg::ColumnSet> required;
      mlir::relalg::ColumnSet leftKeys, rightKeys;
      std::vector<mlir::Type> types;
      bool res = false;
      block->walk([&](mlir::Operation* op) {
         if (auto getAttr = mlir::dyn_cast_or_null<mlir::tuples::GetColumnOp>(op)) {
            required.insert({getAttr.getResult(), mlir::relalg::ColumnSet::from(getAttr.getAttr())});
         } else if (auto cmpOp = mlir::dyn_cast_or_null<mlir::relalg::CmpOpInterface>(op)) {
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
      return res;
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
         mapBlockInfo.block->addArgument(mlir::tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
         auto [keys, nullsEqual] = HashJoinUtils::extractKeys(&predicateOperator.getPredicateBlock(), left.getAvailableColumns(), right.getAvailableColumns(), mapBlockInfo);
         if (!mapBlockInfo.createdColumns.empty()) {
            builder.setInsertionPoint(predicateOperator);
            auto mapOp = builder.create<mlir::relalg::MapOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), left.asRelation(), builder.getArrayAttr(mapBlockInfo.createdColumns));
            mapOp.getPredicate().push_back(mapBlockInfo.block);
            mlir::OpBuilder builder2(builder.getContext());
            builder2.setInsertionPointToEnd(mapBlockInfo.block);
            builder2.create<mlir::tuples::ReturnOp>(builder2.getUnknownLoc(), mapBlockInfo.results);
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
         mapBlockInfo.block->addArgument(mlir::tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
         auto [keys, nullEquals] = HashJoinUtils::extractKeys(&predicateOperator.getPredicateBlock(), right.getAvailableColumns(), left.getAvailableColumns(), mapBlockInfo);
         if (!mapBlockInfo.createdColumns.empty()) {
            builder.setInsertionPoint(predicateOperator);
            auto mapOp = builder.create<mlir::relalg::MapOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), right.asRelation(), builder.getArrayAttr(mapBlockInfo.createdColumns));
            mapOp.getPredicate().push_back(mapBlockInfo.block);
            mlir::OpBuilder builder2(builder.getContext());
            builder2.setInsertionPointToEnd(mapBlockInfo.block);
            builder2.create<mlir::tuples::ReturnOp>(builder2.getUnknownLoc(), mapBlockInfo.results);
            right = mapOp;
         }
         predicateOperator->setAttr("rightHash", builder.getArrayAttr(keys));
      }
      mlir::cast<Operator>(predicateOperator.getOperation()).setChildren({left, right});
   }
   static mlir::Value mapColsToNullable(mlir::Value stream, mlir::OpBuilder& rewriter, mlir::Location loc, mlir::ArrayAttr mapping, size_t exisingOffset = 0, mlir::relalg::ColumnSet excluded = {}) {
      auto& colManager = rewriter.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
      std::vector<mlir::Attribute> defAttrs;
      auto* mapBlock = new mlir::Block;
      auto tupleArg = mapBlock->addArgument(mlir::tuples::TupleType::get(rewriter.getContext()), loc);
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(mapBlock);
         std::vector<mlir::Value> res;
         for (mlir::Attribute attr : mapping) {
            auto relationDefAttr = attr.dyn_cast_or_null<mlir::tuples::ColumnDefAttr>();
            auto* defAttr = &relationDefAttr.getColumn();
            auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>()[exisingOffset].cast<mlir::tuples::ColumnRefAttr>();
            if (excluded.contains(&fromExisting.getColumn())) continue;
            mlir::Value value = rewriter.create<mlir::tuples::GetColumnOp>(loc, rewriter.getI64Type(), fromExisting, tupleArg);
            if (fromExisting.getColumn().type != defAttr->type) {
               mlir::Value tmp = rewriter.create<mlir::db::AsNullableOp>(loc, defAttr->type, value);
               value = tmp;
            }
            res.push_back(value);
            defAttrs.push_back(colManager.createDef(defAttr));
         }
         rewriter.create<mlir::tuples::ReturnOp>(loc, res);
      }
      auto mapOp = rewriter.create<mlir::relalg::MapOp>(loc, mlir::tuples::TupleStreamType::get(rewriter.getContext()), stream, rewriter.getArrayAttr(defAttrs));
      mapOp.getPredicate().push_back(mapBlock);
      return mapOp.getResult();
   }
   void runOnOperation() override {
      std::vector<mlir::Operation*> toErase;
      getOperation().walk([&](Operator op) {
         ::llvm::TypeSwitch<mlir::Operation*, void>(op.getOperation())
            .Case<mlir::relalg::LimitOp>([&](mlir::relalg::LimitOp limitOp) {
               if (auto sortOp = mlir::dyn_cast_or_null<mlir::relalg::SortOp>(limitOp.getRel().getDefiningOp())) {
                  mlir::OpBuilder builder(limitOp);
                  toErase.push_back(limitOp);
                  toErase.push_back(sortOp);

                  limitOp.replaceAllUsesWith(builder.create<mlir::relalg::TopKOp>(limitOp.getLoc(), limitOp.getMaxRows(), sortOp.getRel(), sortOp.getSortspecs()).asRelation());
               }
            })
            .Case<mlir::relalg::InnerJoinOp, mlir::relalg::CollectionJoinOp, mlir::relalg::FullOuterJoinOp>([&](PredicateOperator predicateOperator) {
               auto binOp = mlir::cast<BinaryOperator>(predicateOperator.getOperation());
               auto left = mlir::cast<Operator>(binOp.leftChild());
               auto right = mlir::cast<Operator>(binOp.rightChild());
               if (hashImplPossible(&predicateOperator.getPredicateBlock(), left.getAvailableColumns(), right.getAvailableColumns())) {
                  op->setAttr("impl", mlir::StringAttr::get(op.getContext(), "hash"));
                  op->setAttr("useHashJoin", mlir::UnitAttr::get(op.getContext()));
                  prepareForHash(predicateOperator);
               }
            })
            .Case<mlir::relalg::SemiJoinOp, mlir::relalg::AntiSemiJoinOp, mlir::relalg::OuterJoinOp, mlir::relalg::MarkJoinOp>([&](PredicateOperator predicateOperator) {
               auto binOp = mlir::cast<BinaryOperator>(predicateOperator.getOperation());
               auto left = mlir::cast<Operator>(binOp.leftChild());
               auto right = mlir::cast<Operator>(binOp.rightChild());
               if (left->hasAttr("rows") && right->hasAttr("rows")) {
                  double rowsLeft = 0;
                  double rowsRight = 0;
                  if (auto lDAttr = left->getAttr("rows").dyn_cast_or_null<mlir::FloatAttr>()) {
                     rowsLeft = lDAttr.getValueAsDouble();
                  } else if (auto lIAttr = left->getAttr("rows").dyn_cast_or_null<mlir::IntegerAttr>()) {
                     rowsLeft = lIAttr.getInt();
                  }
                  if (auto rDAttr = right->getAttr("rows").dyn_cast_or_null<mlir::FloatAttr>()) {
                     rowsRight = rDAttr.getValueAsDouble();
                  } else if (auto rIAttr = right->getAttr("rows").dyn_cast_or_null<mlir::IntegerAttr>()) {
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
                     if (auto lDAttr = left->getAttr("rows").dyn_cast_or_null<mlir::FloatAttr>()) {
                        rowsLeft = lDAttr.getValueAsDouble();
                     } else if (auto lIAttr = left->getAttr("rows").dyn_cast_or_null<mlir::IntegerAttr>()) {
                        rowsLeft = lIAttr.getInt();
                     }
                     if (auto rDAttr = right->getAttr("rows").dyn_cast_or_null<mlir::FloatAttr>()) {
                        rowsRight = rDAttr.getValueAsDouble();
                     } else if (auto rIAttr = right->getAttr("rows").dyn_cast_or_null<mlir::IntegerAttr>()) {
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
            .Case<mlir::relalg::SingleJoinOp>([&](mlir::relalg::SingleJoinOp op) {
               if (auto returnOp = mlir::dyn_cast_or_null<mlir::tuples::ReturnOp>(op.getPredicateBlock().getTerminator())) {
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

            .Case<mlir::relalg::OuterJoinOp>([&](mlir::relalg::OuterJoinOp op) {})
            .Case<mlir::relalg::FullOuterJoinOp>([&](mlir::relalg::FullOuterJoinOp op) {})

            .Default([&](auto x) {
            });
      });
      for (auto* op : toErase) {
         op->erase();
      }
      toErase.clear();
      getOperation().walk([&](mlir::relalg::AggregationOp op) {
         auto* potentialJoin = op.getRel().getDefiningOp();
         mlir::relalg::MapOp mapOp = mlir::dyn_cast_or_null<mlir::relalg::MapOp>(potentialJoin);
         mlir::relalg::ColumnSet usedColumns = op.getUsedColumns();
         if (mapOp) {
            usedColumns.insert(mapOp.getUsedColumns());
            potentialJoin = mapOp.getRel().getDefiningOp();
         }

         auto isInnerJoin = mlir::isa<mlir::relalg::InnerJoinOp>(potentialJoin);
         auto isOuterJoin = mlir::isa<mlir::relalg::OuterJoinOp>(potentialJoin);
         if (!isInnerJoin && !isOuterJoin) return;
         PredicateOperator join = mlir::cast<PredicateOperator>(potentialJoin);
         Operator joinOperator = mlir::cast<Operator>(potentialJoin);
         usedColumns.insert(joinOperator.getUsedColumns());
         if (!join->hasAttr("leftHash") || !join->hasAttr("rightHash")) return;
         auto leftKeys = join->getAttr("leftHash").cast<mlir::ArrayAttr>();
         auto rightKeys = join->getAttr("rightHash").cast<mlir::ArrayAttr>();
         auto leftKeySet = mlir::relalg::ColumnSet::fromArrayAttr(leftKeys);
         auto rightKeySet = mlir::relalg::ColumnSet::fromArrayAttr(rightKeys);
         auto groupByKeySet = mlir::relalg::ColumnSet::fromArrayAttr(op.getGroupByCols());
         if (groupByKeySet.size() != leftKeySet.size()) return;
         groupByKeySet.remove(leftKeySet);
         groupByKeySet.remove(rightKeySet);
         if (!groupByKeySet.empty()) return;

         auto leftChild = joinOperator.getChildren()[0];
         auto rightChild = joinOperator.getChildren()[1];
         auto leftUsedColumns = usedColumns.intersect(leftChild.getAvailableColumns());
         auto fds = leftChild.getFDs();
         if (!leftUsedColumns.isSubsetOf(fds.expand(leftKeySet))) return;
         bool containsProjection = false;
         bool containsCountRows = false;
         op.walk([&](mlir::relalg::ProjectionOp) { containsProjection = true; });
         op.walk([&](mlir::relalg::CountRowsOp) { containsCountRows = true; });
         if (containsProjection || (isOuterJoin && containsCountRows)) return;
         mlir::OpBuilder builder(op);
         mlir::ArrayAttr mappedCols = mapOp ? mapOp.getComputedCols() : builder.getArrayAttr({});
         mlir::Value left = leftChild.asRelation();
         mlir::Value right = rightChild.asRelation();
         if (isOuterJoin) {
            auto outerJoin = mlir::cast<mlir::relalg::OuterJoinOp>(potentialJoin);
            right = mapColsToNullable(right, builder, op.getLoc(), outerJoin.getMapping());
         }
         auto groupJoinOp = builder.create<mlir::relalg::GroupJoinOp>(op.getLoc(), left, right, isOuterJoin ? mlir::relalg::GroupJoinBehavior::outer : mlir::relalg::GroupJoinBehavior::inner, leftKeys, rightKeys, mappedCols, op.getComputedCols());
         if (mapOp) {
            mlir::BlockAndValueMapping mapping;
            mapOp.getPredicate().cloneInto(&groupJoinOp.getMapFunc(), mapping);
         } else {
            auto* b = new mlir::Block;
            mlir::OpBuilder mB(&getContext());
            groupJoinOp.getMapFunc().push_back(b);
            mB.setInsertionPointToStart(b);
            mB.create<mlir::tuples::ReturnOp>(op.getLoc());
         }
         {
            mlir::BlockAndValueMapping mapping;
            op.getAggrFunc().cloneInto(&groupJoinOp.getAggrFunc(), mapping);
         }
         {
            mlir::BlockAndValueMapping mapping;
            join.getPredicateRegion().cloneInto(&groupJoinOp.getPredicate(), mapping);
         }
         op->replaceAllUsesWith(groupJoinOp);
         toErase.push_back(op.getOperation());
         if (mapOp) {
            toErase.push_back(mapOp);
         }
         toErase.push_back(join.getOperation());
         llvm::dbgs() << "introducing groupjoin\n";
      });
      for (auto* op : toErase) {
         op->erase();
      }
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createOptimizeImplementationsPass() { return std::make_unique<OptimizeImplementations>(); }
} // end namespace relalg
} // end namespace mlir