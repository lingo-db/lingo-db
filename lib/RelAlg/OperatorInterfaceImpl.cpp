#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/OpImplementation.h"
#include <functional>
using namespace mlir::relalg;
using operator_list = llvm::SmallVector<Operator, 4>;
Attributes mlir::relalg::detail::getCreatedAttributes(mlir::Operation* op) {
   Attributes creations;
   op->walk([&](AddAttrOp attrOp) {
      creations.insert(&attrOp.attr().getRelationalAttribute());
   });
   return creations;
}
static operator_list getChildOperators(mlir::Operation* parent) {
   operator_list children;
   for (auto operand : parent->getOperands()) {
      if (auto childOperator = mlir::dyn_cast_or_null<Operator>(operand.getDefiningOp())) {
         children.push_back(childOperator);
      }
   }
   return children;
}

static Attributes collectAttributes(operator_list operators, std::function<Attributes(Operator)> fn) {
   Attributes collected;
   for (auto op : operators) {
      auto res = fn(op);
      collected.insert(res);
   }
   return collected;
}
Attributes mlir::relalg::detail::getUsedAttributes(mlir::Operation* op) {
   Attributes creations;
   op->walk([&](GetAttrOp attrOp) {
      creations.insert(&attrOp.attr().getRelationalAttribute());
   });
   return creations;
}
Attributes mlir::relalg::detail::getAvailableAttributes(mlir::Operation* op) {
   Operator asOperator = mlir::dyn_cast_or_null<Operator>(op);
   auto collected = collectAttributes(getChildOperators(op), [](Operator op) { return op.getAvailableAttributes(); });
   auto selfCreated = asOperator.getCreatedAttributes();
   collected.insert(selfCreated);
   return collected;
}
Attributes mlir::relalg::detail::getFreeAttributes(mlir::Operation* op) {
   auto available = getAvailableAttributes(op);
   auto collectedFree = collectAttributes(getChildOperators(op), [](Operator op) { return op.getFreeAttributes(); });
   auto used = getUsedAttributes(op);
   collectedFree.insert(used);
   collectedFree.remove(available);
   return collectedFree;
}

bool mlir::relalg::detail::isDependentJoin(mlir::Operation* op) {
   if (auto join = mlir::dyn_cast_or_null<BinaryOperator>(op)) {
      if (isJoin(op)) {
         auto left = mlir::dyn_cast_or_null<Operator>(join.leftChild());
         auto right = mlir::dyn_cast_or_null<Operator>(join.rightChild());
         auto availableLeft = left.getAvailableAttributes();
         auto availableRight = right.getAvailableAttributes();
         return left.getFreeAttributes().intersects(availableRight) || right.getFreeAttributes().intersects(availableLeft);
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

Attributes AggregationOp::getUsedAttributes() {
   auto used = mlir::relalg::detail::getUsedAttributes(getOperation());
   used.insert(Attributes::fromArrayAttr(group_by_attrs()));
   getOperation()->walk([&](mlir::relalg::AggrFuncOp aggrFn) {
      used.insert(&aggrFn.attr().getRelationalAttribute());
   });
   return used;
}
Attributes SortOp::getUsedAttributes() {
   Attributes used;
   for (Attribute a : sortspecs()) {
      used.insert(&a.dyn_cast_or_null<SortSpecificationAttr>().getAttr().getRelationalAttribute());
   }
   return used;
}

Attributes ConstRelationOp::getCreatedAttributes() {
   return Attributes::fromArrayAttr(attributes());
}
Attributes AntiSemiJoinOp::getAvailableAttributes() {
   return mlir::relalg::detail::getAvailableAttributes(leftChild());
}
Attributes SemiJoinOp::getAvailableAttributes() {
   return mlir::relalg::detail::getAvailableAttributes(leftChild());
}
Attributes MarkJoinOp::getAvailableAttributes() {
   auto available = mlir::relalg::detail::getAvailableAttributes(leftChild());
   available.insert(&markattr().getRelationalAttribute());
   return available;
}
Attributes RenamingOp::getCreatedAttributes() {
   Attributes created;

   for (Attribute attr : attributes()) {
      auto relationDefAttr = attr.dyn_cast_or_null<RelationalAttributeDefAttr>();
      created.insert(&relationDefAttr.getRelationalAttribute());
   }
   return created;
}
Attributes MarkJoinOp::getCreatedAttributes() {
   Attributes created;
   created.insert(&markattr().getRelationalAttribute());
   return created;
}
Attributes RenamingOp::getUsedAttributes() {
   Attributes used;

   for (Attribute attr : attributes()) {
      auto relationDefAttr = attr.dyn_cast_or_null<RelationalAttributeDefAttr>();
      auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<ArrayAttr>();
      used.insert(Attributes::fromArrayAttr(fromExisting));
   }
   return used;
}
Attributes RenamingOp::getAvailableAttributes() {
   auto availablePreviously = collectAttributes(getChildOperators(*this), [](Operator op) { return op.getAvailableAttributes(); });
   availablePreviously.remove(getUsedAttributes());
   auto created = getCreatedAttributes();
   availablePreviously.insert(created);
   return availablePreviously;
}
Attributes BaseTableOp::getCreatedAttributes() {
   Attributes creations;
   for (auto mapping : columns()) {
      auto [_, attr] = mapping;
      auto relationDefAttr = attr.dyn_cast_or_null<RelationalAttributeDefAttr>();
      creations.insert(&relationDefAttr.getRelationalAttribute());
   }
   return creations;
}
Attributes mlir::relalg::AggregationOp::getAvailableAttributes() {
   Attributes available = getCreatedAttributes();
   available.insert(Attributes::fromArrayAttr(group_by_attrs()));
   return available;
}
Attributes mlir::relalg::ProjectionOp::getAvailableAttributes() {
   return Attributes::fromArrayAttr(attrs());
}

bool mlir::relalg::detail::isJoin(Operation* op) {
   auto opType = getBinaryOperatorType(op);
   return BinaryOperatorType::InnerJoin <= opType && opType <= BinaryOperatorType::MarkJoin;
}

void mlir::relalg::detail::addPredicate(mlir::Operation* op, std::function<mlir::Value(mlir::Value, mlir::OpBuilder&)> predicateProducer) {
   auto lambdaOperator = mlir::dyn_cast_or_null<PredicateOperator>(op);
   auto* terminator = lambdaOperator.getPredicateBlock().getTerminator();
   mlir::OpBuilder builder(terminator);
   auto additionalPred = predicateProducer(lambdaOperator.getPredicateArgument(), builder);
   if (terminator->getNumOperands() > 0) {
      mlir::Value oldValue = terminator->getOperand(0);
      bool nullable = oldValue.getType().dyn_cast_or_null<mlir::db::DBType>().isNullable() || additionalPred.getType().dyn_cast_or_null<mlir::db::DBType>().isNullable();
      mlir::Value anded = builder.create<mlir::db::AndOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext(), nullable), mlir::ValueRange({oldValue, additionalPred}));
      builder.create<mlir::relalg::ReturnOp>(builder.getUnknownLoc(), anded);
   } else {
      builder.create<mlir::relalg::ReturnOp>(builder.getUnknownLoc(), additionalPred);
   }
   terminator->remove();
   terminator->destroy();
}
void mlir::relalg::detail::initPredicate(mlir::Operation* op) {
   auto* context = op->getContext();
   mlir::Type tupleType = mlir::relalg::TupleType::get(context);
   auto* block = new mlir::Block;
   op->getRegion(0).push_back(block);
   block->addArgument(tupleType);
   mlir::OpBuilder builder(context);
   builder.setInsertionPointToStart(block);
   builder.create<mlir::relalg::ReturnOp>(builder.getUnknownLoc());
}

static void addRequirements(mlir::Operation* op, mlir::Operation* includeChildren, mlir::Operation* excludeChildren, llvm::SmallVector<mlir::Operation*, 8>& extracted, llvm::SmallPtrSet<mlir::Operation*, 8>& alreadyPresent) {
   if (!op)
      return;
   if (alreadyPresent.contains(op))
      return;
   if (!includeChildren->isAncestor(op))
      return;
   for (auto operand : op->getOperands()) {
      addRequirements(operand.getDefiningOp(), includeChildren, excludeChildren, extracted, alreadyPresent);
   }
   alreadyPresent.insert(op);
   if (!excludeChildren->isAncestor(op)) {
      extracted.push_back(op);
   }
}
void mlir::relalg::detail::inlineOpIntoBlock(mlir::Operation* vop, mlir::Operation* includeChildren, mlir::Operation* excludeChildren, mlir::Block* newBlock, mlir::BlockAndValueMapping& mapping) {
   llvm::SmallVector<mlir::Operation*, 8> extracted;
   llvm::SmallPtrSet<mlir::Operation*, 8> alreadyPresent;
   addRequirements(vop, includeChildren, excludeChildren, extracted, alreadyPresent);
   mlir::OpBuilder builder(vop->getContext());
   builder.setInsertionPointToStart(newBlock);
   mlir::Operation* first = &newBlock->front();
   for (auto* op : extracted) {
      auto* cloneOp = builder.clone(*op, mapping);
      cloneOp->moveBefore(first);
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
