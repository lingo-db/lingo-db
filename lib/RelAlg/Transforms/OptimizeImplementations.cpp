#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Conversion/RelAlgToDB/HashJoinTranslator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/Passes.h"
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
   static std::vector<mlir::Attribute> extractKeys(mlir::Block* block, mlir::relalg::ColumnSet keyAttributes, mlir::relalg::ColumnSet otherAttributes, MapBlockInfo& mapBlockInfo) {
      std::vector<mlir::Attribute> toHash;
      llvm::DenseMap<mlir::Value, mlir::relalg::ColumnSet> required;
      mlir::BlockAndValueMapping mapping;
      mapping.map(block->getArgument(0), mapBlockInfo.block->getArgument(0));
      size_t i = 0;
      block->walk([&](mlir::Operation* op) {
         if (auto getAttr = mlir::dyn_cast_or_null<mlir::tuples::GetColumnOp>(op)) {
            required.insert({getAttr.getResult(), mlir::relalg::ColumnSet::from(getAttr.attr())});
         } else if (auto cmpOp = mlir::dyn_cast_or_null<mlir::relalg::CmpOpInterface>(op)) {
            if (cmpOp.isEqualityPred() && isAndedResult(op)) {
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
                     toHash.push_back(getColOp.attr());
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
      return toHash;
   }
};
class OptimizeImplementations : public mlir::PassWrapper<OptimizeImplementations, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-optimize-implementations"; }

   public:
   bool hashImplPossible(mlir::Block* block, mlir::relalg::ColumnSet availableLeft, mlir::relalg::ColumnSet availableRight) { //todo: does not work always
      llvm::DenseMap<mlir::Value, mlir::relalg::ColumnSet> required;
      mlir::relalg::ColumnSet leftKeys, rightKeys;
      std::vector<mlir::Type> types;
      bool res = false;
      block->walk([&](mlir::Operation* op) {
         if (auto getAttr = mlir::dyn_cast_or_null<mlir::tuples::GetColumnOp>(op)) {
            required.insert({getAttr.getResult(), mlir::relalg::ColumnSet::from(getAttr.attr())});
         } else if (auto cmpOp = mlir::dyn_cast_or_null<mlir::relalg::CmpOpInterface>(op)) {
            if (cmpOp.isEqualityPred() && HashJoinUtils::isAndedResult(op)) {
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
         auto keys = HashJoinUtils::extractKeys(&predicateOperator.getPredicateBlock(), left.getAvailableColumns(), right.getAvailableColumns(), mapBlockInfo);
         if (!mapBlockInfo.createdColumns.empty()) {
            builder.setInsertionPoint(predicateOperator);
            auto mapOp = builder.create<mlir::relalg::MapOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), left.asRelation(), builder.getArrayAttr(mapBlockInfo.createdColumns));
            mapOp.predicate().push_back(mapBlockInfo.block);
            mlir::OpBuilder builder2(builder.getContext());
            builder2.setInsertionPointToEnd(mapBlockInfo.block);
            builder2.create<mlir::tuples::ReturnOp>(builder2.getUnknownLoc(), mapBlockInfo.results);
            left = mapOp;
         }
         predicateOperator->setAttr("leftHash", builder.getArrayAttr(keys));
      }
      //right
      {
         mlir::OpBuilder builder(&this->getContext());
         HashJoinUtils::MapBlockInfo mapBlockInfo;
         mapBlockInfo.block = new mlir::Block;
         mapBlockInfo.block->addArgument(mlir::tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
         auto keys = HashJoinUtils::extractKeys(&predicateOperator.getPredicateBlock(), right.getAvailableColumns(), left.getAvailableColumns(), mapBlockInfo);
         if (!mapBlockInfo.createdColumns.empty()) {
            builder.setInsertionPoint(predicateOperator);
            auto mapOp = builder.create<mlir::relalg::MapOp>(builder.getUnknownLoc(), mlir::tuples::TupleStreamType::get(builder.getContext()), right.asRelation(), builder.getArrayAttr(mapBlockInfo.createdColumns));
            mapOp.predicate().push_back(mapBlockInfo.block);
            mlir::OpBuilder builder2(builder.getContext());
            builder2.setInsertionPointToEnd(mapBlockInfo.block);
            builder2.create<mlir::tuples::ReturnOp>(builder2.getUnknownLoc(), mapBlockInfo.results);
            right = mapOp;
         }
         predicateOperator->setAttr("rightHash", builder.getArrayAttr(keys));
      }
      mlir::cast<Operator>(predicateOperator.getOperation()).setChildren({left, right});
   }
   void runOnOperation() override {
      getOperation().walk([&](Operator op) {
         ::llvm::TypeSwitch<mlir::Operation*, void>(op.getOperation())
            .Case<mlir::relalg::InnerJoinOp,mlir::relalg::CollectionJoinOp>([&](PredicateOperator predicateOperator) {
               auto binOp = mlir::cast<BinaryOperator>(predicateOperator.getOperation());
               auto left = mlir::cast<Operator>(binOp.leftChild());
               auto right = mlir::cast<Operator>(binOp.rightChild());
               if (hashImplPossible(&predicateOperator.getPredicateBlock(), left.getAvailableColumns(), right.getAvailableColumns())) {
                  op->setAttr("impl", mlir::StringAttr::get(op.getContext(), "hash"));
                  op->setAttr("useHashJoin",mlir::UnitAttr::get(op.getContext()));
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
                  op->setAttr("useHashJoin",mlir::UnitAttr::get(op.getContext()));
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
                  if (returnOp.results().empty()) {
                     op->setAttr("impl", mlir::StringAttr::get(op.getContext(), "constant"));
                  }
               }
               auto left = mlir::cast<Operator>(op.leftChild());
               auto right = mlir::cast<Operator>(op.rightChild());
               if (hashImplPossible(&op.getPredicateBlock(), left.getAvailableColumns(), right.getAvailableColumns())) {
                  prepareForHash(op);
                  op->setAttr("useHashJoin",mlir::UnitAttr::get(op.getContext()));
                  op->setAttr("impl", mlir::StringAttr::get(op.getContext(), "hash"));
               }
            })

            .Case<mlir::relalg::OuterJoinOp>([&](mlir::relalg::OuterJoinOp op) {})
            .Case<mlir::relalg::FullOuterJoinOp>([&](mlir::relalg::FullOuterJoinOp op) {})

            .Default([&](auto x) {
            });
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createOptimizeImplementationsPass() { return std::make_unique<OptimizeImplementations>(); }
} // end namespace relalg
} // end namespace mlir