#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/OpImplementation.h"
#include <functional>
#include <unordered_set>
using namespace mlir::relalg;
using operator_list = llvm::SmallVector<Operator, 4>;
static operator_list getChildOperators(mlir::Operation* parent) {
   operator_list children;
   for (auto operand : parent->getOperands()) {
      if (auto childOperator = mlir::dyn_cast_or_null<Operator>(operand.getDefiningOp())) {
         children.push_back(childOperator);
      }
   }
   return children;
}

static ColumnSet collectColumns(operator_list operators, std::function<ColumnSet(Operator)> fn) {
   ColumnSet collected;
   for (auto op : operators) {
      auto res = fn(op);
      collected.insert(res);
   }
   return collected;
}
ColumnSet mlir::relalg::detail::getUsedColumns(mlir::Operation* op) {
   ColumnSet creations;
   op->walk([&](GetColumnOp attrOp) {
      creations.insert(&attrOp.attr().getColumn());
   });
   return creations;
}
ColumnSet mlir::relalg::detail::getAvailableColumns(mlir::Operation* op) {
   Operator asOperator = mlir::dyn_cast_or_null<Operator>(op);
   auto collected = collectColumns(getChildOperators(op), [](Operator op) { return op.getAvailableColumns(); });
   auto selfCreated = asOperator.getCreatedColumns();
   collected.insert(selfCreated);
   return collected;
}
FunctionalDependencies mlir::relalg::detail::getFDs(mlir::Operation* op) {
   FunctionalDependencies dependencies;
   for (auto child : getChildOperators(op)) {
      dependencies.insert(child.getFDs());
   }
   return dependencies;
}
ColumnSet mlir::relalg::detail::getFreeColumns(mlir::Operation* op) {
   auto available = collectColumns(getChildOperators(op), [](Operator op) { return op.getAvailableColumns(); });
   auto collectedFree = collectColumns(getChildOperators(op), [](Operator op) { return op.getFreeColumns(); });
   auto used = mlir::cast<Operator>(op).getUsedColumns();
   collectedFree.insert(used);
   collectedFree.remove(available);
   return collectedFree;
}

bool mlir::relalg::detail::isDependentJoin(mlir::Operation* op) {
   if (auto join = mlir::dyn_cast_or_null<BinaryOperator>(op)) {
      if (isJoin(op)) {
         auto left = mlir::dyn_cast_or_null<Operator>(join.leftChild());
         auto right = mlir::dyn_cast_or_null<Operator>(join.rightChild());
         auto availableLeft = left.getAvailableColumns();
         auto availableRight = right.getAvailableColumns();
         return left.getFreeColumns().intersects(availableRight) || right.getFreeColumns().intersects(availableLeft);
      }
   }
   return false;
}
mlir::relalg::detail::BinaryOperatorType mlir::relalg::detail::getBinaryOperatorType(Operation* op) {
   return ::llvm::TypeSwitch<mlir::Operation*, BinaryOperatorType>(op)
      .Case<mlir::relalg::UnionOp>([&](mlir::Operation* op) { return BinaryOperatorType::Union; })
      .Case<mlir::relalg::IntersectOp>([&](mlir::Operation* op) { return BinaryOperatorType::Intersection; })
      .Case<mlir::relalg::ExceptOp>([&](mlir::Operation* op) { return BinaryOperatorType::Except; })
      .Case<mlir::relalg::CrossProductOp>([&](mlir::Operation* op) { return BinaryOperatorType::CP; })
      .Case<mlir::relalg::InnerJoinOp>([&](mlir::Operation* op) { return BinaryOperatorType::InnerJoin; })
      .Case<mlir::relalg::SemiJoinOp>([&](mlir::Operation* op) { return BinaryOperatorType::SemiJoin; })
      .Case<mlir::relalg::AntiSemiJoinOp>([&](mlir::Operation* op) { return BinaryOperatorType::AntiSemiJoin; })
      .Case<mlir::relalg::SingleJoinOp>([&](mlir::Operation* op) { return BinaryOperatorType::OuterJoin; })
      .Case<mlir::relalg::MarkJoinOp>([&](mlir::Operation* op) { return BinaryOperatorType::MarkJoin; })
      .Case<mlir::relalg::CollectionJoinOp>([&](mlir::Operation* op) { return BinaryOperatorType::CollectionJoin; })
      .Case<mlir::relalg::OuterJoinOp>([&](mlir::relalg::OuterJoinOp op) { return BinaryOperatorType::OuterJoin; })
      .Case<mlir::relalg::FullOuterJoinOp>([&](mlir::relalg::FullOuterJoinOp op) { return BinaryOperatorType::FullOuterJoin; })
      .Default([&](auto x) {
         return BinaryOperatorType::None;
      });
}
mlir::relalg::detail::UnaryOperatorType mlir::relalg::detail::getUnaryOperatorType(Operation* op) {
   return ::llvm::TypeSwitch<mlir::Operation*, UnaryOperatorType>(op)
      .Case<mlir::relalg::SelectionOp>([&](mlir::Operation* op) { return UnaryOperatorType::Selection; })
      .Case<mlir::relalg::MapOp>([&](mlir::Operation* op) { return UnaryOperatorType::Map; })
      .Case<mlir::relalg::ProjectionOp>([&](mlir::relalg::ProjectionOp op) { return op.set_semantic() == mlir::relalg::SetSemantic::distinct ? UnaryOperatorType::DistinctProjection : UnaryOperatorType::Projection; })
      .Case<mlir::relalg::AggregationOp>([&](mlir::Operation* op) { return UnaryOperatorType::Aggregation; })
      .Default([&](auto x) {
         return UnaryOperatorType::None;
      });
}
ColumnSet MapOp::getCreatedColumns() {
   return ColumnSet::fromArrayAttr(computed_cols());
}
ColumnSet AggregationOp::getCreatedColumns() {
   return ColumnSet::fromArrayAttr(computed_cols());
}
ColumnSet AggregationOp::getUsedColumns() {
   auto used = mlir::relalg::detail::getUsedColumns(getOperation());
   used.insert(ColumnSet::fromArrayAttr(group_by_cols()));
   getOperation()->walk([&](mlir::relalg::AggrFuncOp aggrFn) {
      used.insert(&aggrFn.attr().getColumn());
   });
   return used;
}
ColumnSet SortOp::getUsedColumns() {
   ColumnSet used;
   for (Attribute a : sortspecs()) {
      used.insert(&a.dyn_cast_or_null<mlir::relalg::SortSpecificationAttr>().getAttr().getColumn());
   }
   return used;
}

ColumnSet ConstRelationOp::getCreatedColumns() {
   return ColumnSet::fromArrayAttr(columns());
}
ColumnSet AntiSemiJoinOp::getAvailableColumns() {
   return mlir::relalg::detail::getAvailableColumns(leftChild());
}
ColumnSet SemiJoinOp::getAvailableColumns() {
   return mlir::relalg::detail::getAvailableColumns(leftChild());
}
ColumnSet MarkJoinOp::getAvailableColumns() {
   auto available = mlir::relalg::detail::getAvailableColumns(leftChild());
   available.insert(&markattr().getColumn());
   return available;
}
ColumnSet RenamingOp::getCreatedColumns() {
   ColumnSet created;

   for (Attribute attr : columns()) {
      auto relationDefAttr = attr.dyn_cast_or_null<ColumnDefAttr>();
      created.insert(&relationDefAttr.getColumn());
   }
   return created;
}
ColumnSet RenamingOp::getUsedColumns() {
   ColumnSet used;

   for (Attribute attr : columns()) {
      auto relationDefAttr = attr.dyn_cast_or_null<ColumnDefAttr>();
      auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<ArrayAttr>();
      used.insert(ColumnSet::fromArrayAttr(fromExisting));
   }
   return used;
}
ColumnSet RenamingOp::getAvailableColumns() {
   auto availablePreviously = collectColumns(getChildOperators(*this), [](Operator op) { return op.getAvailableColumns(); });
   availablePreviously.remove(getUsedColumns());
   auto created = getCreatedColumns();
   availablePreviously.insert(created);
   return availablePreviously;
}
ColumnSet mlir::relalg::detail::getSetOpCreatedColumns(mlir::Operation* op) {
   ColumnSet created;
   for (mlir::Attribute attr : op->getAttr("mapping").cast<mlir::ArrayAttr>()) {
      auto relationDefAttr = attr.dyn_cast_or_null<ColumnDefAttr>();
      created.insert(&relationDefAttr.getColumn());
   }
   return created;
}
ColumnSet mlir::relalg::detail::getSetOpUsedColumns(mlir::Operation* op) {
   ColumnSet used;
   for (Attribute attr : op->getAttr("mapping").cast<mlir::ArrayAttr>()) {
      auto relationDefAttr = attr.dyn_cast_or_null<ColumnDefAttr>();
      auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<ArrayAttr>();
      used.insert(ColumnSet::fromArrayAttr(fromExisting));
   }
   return used;
}
ColumnSet OuterJoinOp::getCreatedColumns() {
   ColumnSet created;

   for (Attribute attr : mapping()) {
      auto relationDefAttr = attr.dyn_cast_or_null<ColumnDefAttr>();
      created.insert(&relationDefAttr.getColumn());
   }
   return created;
}
ColumnSet OuterJoinOp::getUsedColumns() {
   return mlir::relalg::detail::getUsedColumns(getOperation());
}
ColumnSet OuterJoinOp::getAvailableColumns() {
   ColumnSet renamed;

   for (Attribute attr : mapping()) {
      auto relationDefAttr = attr.dyn_cast_or_null<ColumnDefAttr>();
      auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<ArrayAttr>();
      renamed.insert(ColumnSet::fromArrayAttr(fromExisting));
   }
   auto availablePreviously = collectColumns(getChildOperators(*this), [](Operator op) { return op.getAvailableColumns(); });
   availablePreviously.remove(renamed);
   auto created = getCreatedColumns();
   availablePreviously.insert(created);
   return availablePreviously;
}
ColumnSet SingleJoinOp::getCreatedColumns() {
   ColumnSet created;

   for (Attribute attr : mapping()) {
      auto relationDefAttr = attr.dyn_cast_or_null<ColumnDefAttr>();
      created.insert(&relationDefAttr.getColumn());
   }
   return created;
}
ColumnSet SingleJoinOp::getUsedColumns() {
   return mlir::relalg::detail::getUsedColumns(getOperation());
}
ColumnSet SingleJoinOp::getAvailableColumns() {
   ColumnSet renamed;

   for (Attribute attr : mapping()) {
      auto relationDefAttr = attr.dyn_cast_or_null<ColumnDefAttr>();
      auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<ArrayAttr>();
      renamed.insert(ColumnSet::fromArrayAttr(fromExisting));
   }
   auto availablePreviously = collectColumns(getChildOperators(*this), [](Operator op) { return op.getAvailableColumns(); });
   availablePreviously.remove(renamed);
   auto created = getCreatedColumns();
   availablePreviously.insert(created);
   return availablePreviously;
}
ColumnSet CollectionJoinOp::getCreatedColumns() {
   ColumnSet created;
   created.insert(&collAttr().getColumn());
   return created;
}
ColumnSet CollectionJoinOp::getUsedColumns() {
   return mlir::relalg::detail::getUsedColumns(getOperation());
}
ColumnSet CollectionJoinOp::getAvailableColumns() {
   auto availablePreviously = collectColumns(getChildOperators(*this), [](Operator op) { return op.getAvailableColumns(); });
   auto created = getCreatedColumns();
   availablePreviously.insert(created);
   return availablePreviously;
}
ColumnSet MarkJoinOp::getCreatedColumns() {
   ColumnSet created;
   created.insert(&markattr().getColumn());
   return created;
}

ColumnSet BaseTableOp::getCreatedColumns() {
   ColumnSet creations;
   for (auto mapping : columns()) {
      auto attr = mapping.getValue();
      auto relationDefAttr = attr.dyn_cast_or_null<ColumnDefAttr>();
      creations.insert(&relationDefAttr.getColumn());
   }
   return creations;
}
mlir::relalg::FunctionalDependencies BaseTableOp::getFDs() {
   FunctionalDependencies dependencies;
   auto metaData = meta().getMeta();
   if (!metaData->isPresent()) return dependencies;
   if (metaData->getPrimaryKey().empty()) return dependencies;
   auto right = getAvailableColumns();
   std::unordered_set<std::string> pks(metaData->getPrimaryKey().begin(), metaData->getPrimaryKey().end());
   ColumnSet pk;
   for (auto mapping : columns()) {
      auto attr = mapping.getValue();
      auto relationDefAttr = attr.dyn_cast_or_null<ColumnDefAttr>();
      if (pks.contains(mapping.getName().str())) {
         pk.insert(&relationDefAttr.getColumn());
      }
   }
   right.remove(pk);
   dependencies.insert(pk, right);
   return dependencies;
}
ColumnSet mlir::relalg::AggregationOp::getAvailableColumns() {
   ColumnSet available = getCreatedColumns();
   available.insert(ColumnSet::fromArrayAttr(group_by_cols()));
   return available;
}
ColumnSet mlir::relalg::ProjectionOp::getAvailableColumns() {
   return ColumnSet::fromArrayAttr(cols());
}
ColumnSet mlir::relalg::ProjectionOp::getUsedColumns() {
   return set_semantic() == SetSemantic::distinct ? ColumnSet::fromArrayAttr(cols()) : ColumnSet();
}
bool mlir::relalg::detail::isJoin(Operation* op) {
   auto opType = getBinaryOperatorType(op);
   return BinaryOperatorType::InnerJoin <= opType && opType <= BinaryOperatorType::CollectionJoin;
}

void mlir::relalg::detail::addPredicate(mlir::Operation* op, std::function<mlir::Value(mlir::Value, mlir::OpBuilder&)> predicateProducer) {
   auto lambdaOperator = mlir::dyn_cast_or_null<PredicateOperator>(op);
   auto* terminator = lambdaOperator.getPredicateBlock().getTerminator();
   mlir::OpBuilder builder(terminator);
   auto additionalPred = predicateProducer(lambdaOperator.getPredicateArgument(), builder);
   if (terminator->getNumOperands() > 0) {
      mlir::Value oldValue = terminator->getOperand(0);
      bool nullable = oldValue.getType().isa<mlir::db::NullableType>() || additionalPred.getType().isa<mlir::db::NullableType>();
      mlir::Type restype = builder.getI1Type();
      if (nullable) {
         restype = mlir::db::NullableType::get(builder.getContext(), restype);
      }
      mlir::Value anded = builder.create<mlir::db::AndOp>(op->getLoc(), restype, mlir::ValueRange({oldValue, additionalPred}));
      builder.create<mlir::relalg::ReturnOp>(op->getLoc(), anded);
   } else {
      builder.create<mlir::relalg::ReturnOp>(op->getLoc(), additionalPred);
   }
   terminator->erase();
}
void mlir::relalg::detail::initPredicate(mlir::Operation* op) {
   auto* context = op->getContext();
   mlir::Type tupleType = mlir::relalg::TupleType::get(context);
   auto* block = new mlir::Block;
   op->getRegion(0).push_back(block);
   block->addArgument(tupleType, op->getLoc());
   mlir::OpBuilder builder(context);
   builder.setInsertionPointToStart(block);
   builder.create<mlir::relalg::ReturnOp>(op->getLoc());
}

static void addRequirements(mlir::Operation* op, mlir::Operation* includeChildren, mlir::Operation* excludeChildren, llvm::SmallVector<mlir::Operation*, 8>& extracted, llvm::SmallPtrSet<mlir::Operation*, 8>& alreadyPresent, mlir::BlockAndValueMapping& mapping) {
   if (!op)
      return;
   if (alreadyPresent.contains(op))
      return;
   if (!includeChildren->isAncestor(op))
      return;
   for (auto operand : op->getOperands()) {
      if (!mapping.contains(operand)) {
         addRequirements(operand.getDefiningOp(), includeChildren, excludeChildren, extracted, alreadyPresent, mapping);
      }
   }
   op->walk([&](mlir::Operation* op2) {
      for (auto operand : op2->getOperands()) {
         if (!mapping.contains(operand)) {
            auto* definingOp = operand.getDefiningOp();
            if (definingOp && !op->isAncestor(definingOp)) {
               addRequirements(definingOp, includeChildren, excludeChildren, extracted, alreadyPresent, mapping);
            }
         }
      }
   });
   alreadyPresent.insert(op);
   if (!excludeChildren->isAncestor(op)) {
      extracted.push_back(op);
   }
}
void mlir::relalg::detail::inlineOpIntoBlock(mlir::Operation* vop, mlir::Operation* includeChildren, mlir::Operation* excludeChildren, mlir::Block* newBlock, mlir::BlockAndValueMapping& mapping, mlir::Operation* first) {
   llvm::SmallVector<mlir::Operation*, 8> extracted;
   llvm::SmallPtrSet<mlir::Operation*, 8> alreadyPresent;
   addRequirements(vop, includeChildren, excludeChildren, extracted, alreadyPresent, mapping);
   mlir::OpBuilder builder(vop->getContext());
   builder.setInsertionPointToStart(newBlock);
   first = first ? first : (newBlock->empty() ? nullptr : &newBlock->front());
   for (auto* op : extracted) {
      auto* cloneOp = builder.clone(*op, mapping);
      if (first) {
         cloneOp->moveBefore(first);
      } else {
         cloneOp->moveBefore(newBlock, newBlock->begin());
         first = cloneOp;
      }
   }
}
void mlir::relalg::detail::moveSubTreeBefore(mlir::Operation* op, mlir::Operation* before) {
   auto tree = mlir::dyn_cast_or_null<Operator>(op);
   tree->moveBefore(before);
   for (auto child : tree.getChildren()) {
      moveSubTreeBefore(child, tree);
   }
}
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.cpp.inc"
