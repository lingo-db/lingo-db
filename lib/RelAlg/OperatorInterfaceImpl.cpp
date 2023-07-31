#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"
#include <functional>
#include <unordered_set>
using namespace mlir::tuples;
using namespace mlir::relalg;
using operator_list = llvm::SmallVector<Operator, 4>;
namespace {
operator_list getChildOperators(mlir::Operation* parent) {
   operator_list children;
   for (auto operand : parent->getOperands()) {
      if (auto childOperator = mlir::dyn_cast_or_null<Operator>(operand.getDefiningOp())) {
         children.push_back(childOperator);
      }
   }
   return children;
}

ColumnSet collectColumns(operator_list operators, std::function<ColumnSet(Operator)> fn) {
   ColumnSet collected;
   for (auto op : operators) {
      auto res = fn(op);
      collected.insert(res);
   }
   return collected;
}
void addRequirements(mlir::Operation* op, mlir::Operation* includeChildren, mlir::Block* excludeChildren, llvm::SmallVector<mlir::Operation*, 8>& extracted, llvm::SmallPtrSet<mlir::Operation*, 8>& alreadyPresent, mlir::IRMapping& mapping) {
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
   if (!excludeChildren->findAncestorOpInBlock(*op)) {
      extracted.push_back(op);
   }
}
void replaceColumnUsesInLamda(mlir::MLIRContext* context, mlir::Block& block, const mlir::relalg::ColumnFoldInfo& columnInfo) {
   auto& colManager = context->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
   block.walk([&columnInfo, &colManager](mlir::tuples::GetColumnOp getColumnOp) {
      auto* currColumn = &getColumnOp.getAttr().getColumn();
      if (columnInfo.directMappings.contains(currColumn)) {
         getColumnOp.setAttrAttr(colManager.createRef(columnInfo.directMappings.at(currColumn)));
      }
   });
}
} // namespace

bool mlir::relalg::detail::canColumnReach(mlir::Operation* currentOp, mlir::Operation* sourceOp, mlir::Operation* targetOp, const mlir::tuples::Column* column) {
   if (currentOp == targetOp) {
      return true;
   }
   for (auto res : currentOp->getResults()) {
      if (res.getType().isa<mlir::tuples::TupleStreamType>()) {
         for (auto* user : res.getUsers()) {
            if (auto op = mlir::dyn_cast_or_null<Operator>(user)) {
               if (op.canColumnReach(mlir::cast<Operator>(currentOp), mlir::cast<Operator>(targetOp), column)) {
                  return true;
               }
            }
         }
      }
   }
   return false;
}
ColumnSet mlir::relalg::detail::getUsedColumns(mlir::Operation* op) {
   ColumnSet creations;
   op->walk([&](GetColumnOp attrOp) {
      creations.insert(&attrOp.getAttr().getColumn());
   });
   if (op->hasAttr("rightHash")) {
      creations.insert(ColumnSet::fromArrayAttr(op->getAttrOfType<mlir::ArrayAttr>("rightHash")));
   }
   if (op->hasAttr("leftHash")) {
      creations.insert(ColumnSet::fromArrayAttr(op->getAttrOfType<mlir::ArrayAttr>("leftHash")));
   }
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
      .Case<mlir::relalg::ProjectionOp>([&](mlir::relalg::ProjectionOp op) { return op.getSetSemantic() == mlir::relalg::SetSemantic::distinct ? UnaryOperatorType::DistinctProjection : UnaryOperatorType::Projection; })
      .Case<mlir::relalg::AggregationOp>([&](mlir::Operation* op) { return UnaryOperatorType::Aggregation; })
      .Default([&](auto x) {
         return UnaryOperatorType::None;
      });
}
ColumnSet MapOp::getCreatedColumns() {
   return ColumnSet::fromArrayAttr(getComputedCols());
}
ColumnSet AggregationOp::getCreatedColumns() {
   return ColumnSet::fromArrayAttr(getComputedCols());
}
ColumnSet AggregationOp::getUsedColumns() {
   auto used = mlir::relalg::detail::getUsedColumns(getOperation());
   used.insert(ColumnSet::fromArrayAttr(getGroupByCols()));
   getOperation()->walk([&](mlir::relalg::AggrFuncOp aggrFn) {
      used.insert(&aggrFn.getAttr().getColumn());
   });
   return used;
}
ColumnSet GroupJoinOp::getCreatedColumns() {
   return ColumnSet::fromArrayAttr(getComputedCols());
}
ColumnSet GroupJoinOp::getUsedColumns() {
   auto used = mlir::relalg::detail::getUsedColumns(getOperation());
   used.insert(ColumnSet::fromArrayAttr(getLeftCols()));
   used.insert(ColumnSet::fromArrayAttr(getRightCols()));
   getOperation()->walk([&](mlir::relalg::AggrFuncOp aggrFn) {
      used.insert(&aggrFn.getAttr().getColumn());
   });
   return used;
}
ColumnSet GroupJoinOp::getAvailableColumns() {
   ColumnSet available = getCreatedColumns();
   available.insert(ColumnSet::fromArrayAttr(getLeftCols()));
   available.insert(ColumnSet::fromArrayAttr(getRightCols()));
   return available;
}
bool GroupJoinOp::canColumnReach(Operator source, Operator target, const mlir::tuples::Column* col) {
   ColumnSet available = getAvailableColumns();
   if (available.contains(col)) {
      return mlir::relalg::detail::canColumnReach(this->getOperation(), source, target, col);
   }
   return false;
}
ColumnSet WindowOp::getCreatedColumns() {
   return ColumnSet::fromArrayAttr(getComputedCols());
}
ColumnSet WindowOp::getUsedColumns() {
   auto used = mlir::relalg::detail::getUsedColumns(getOperation());
   used.insert(ColumnSet::fromArrayAttr(getPartitionBy()));
   getOperation()->walk([&](mlir::relalg::AggrFuncOp aggrFn) {
      used.insert(&aggrFn.getAttr().getColumn());
   });
   for (Attribute a : getOrderBy()) {
      used.insert(&a.dyn_cast_or_null<mlir::relalg::SortSpecificationAttr>().getAttr().getColumn());
   }
   return used;
}
ColumnSet SortOp::getUsedColumns() {
   ColumnSet used;
   for (Attribute a : getSortspecs()) {
      used.insert(&a.dyn_cast_or_null<mlir::relalg::SortSpecificationAttr>().getAttr().getColumn());
   }
   return used;
}
ColumnSet TopKOp::getUsedColumns() {
   ColumnSet used;
   for (Attribute a : getSortspecs()) {
      used.insert(&a.dyn_cast_or_null<mlir::relalg::SortSpecificationAttr>().getAttr().getColumn());
   }
   return used;
}

ColumnSet ConstRelationOp::getCreatedColumns() {
   return ColumnSet::fromArrayAttr(getColumns());
}
ColumnSet AntiSemiJoinOp::getAvailableColumns() {
   return mlir::relalg::detail::getAvailableColumns(leftChild());
}
bool AntiSemiJoinOp::canColumnReach(Operator source, Operator target, const mlir::tuples::Column* col) {
   if (getRight().getDefiningOp() == source) {
      return false;
   }
   return mlir::relalg::detail::canColumnReach(this->getOperation(), source, target, col);
}
ColumnSet SemiJoinOp::getAvailableColumns() {
   return mlir::relalg::detail::getAvailableColumns(leftChild());
}
bool SemiJoinOp::canColumnReach(Operator source, Operator target, const mlir::tuples::Column* col) {
   if (getRight().getDefiningOp() == source) {
      return false;
   }
   return mlir::relalg::detail::canColumnReach(this->getOperation(), source, target, col);
}
ColumnSet MarkJoinOp::getAvailableColumns() {
   auto available = mlir::relalg::detail::getAvailableColumns(leftChild());
   available.insert(&getMarkattr().getColumn());
   return available;
}
bool MarkJoinOp::canColumnReach(Operator source, Operator target, const mlir::tuples::Column* col) {
   if (getRight().getDefiningOp() == source) {
      return false;
   }
   return mlir::relalg::detail::canColumnReach(this->getOperation(), source, target, col);
}
ColumnSet RenamingOp::getCreatedColumns() {
   ColumnSet created;

   for (Attribute attr : getColumns()) {
      auto relationDefAttr = attr.dyn_cast_or_null<ColumnDefAttr>();
      created.insert(&relationDefAttr.getColumn());
   }
   return created;
}
ColumnSet RenamingOp::getUsedColumns() {
   ColumnSet used;

   for (Attribute attr : getColumns()) {
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
bool mlir::relalg::RenamingOp::canColumnReach(Operator source, Operator target, const mlir::tuples::Column* col) {
   for (Attribute attr : getColumns()) {
      auto relationDefAttr = attr.dyn_cast_or_null<ColumnDefAttr>();
      auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<ArrayAttr>();
      if (&fromExisting[0].cast<mlir::tuples::ColumnRefAttr>().getColumn() == col) {
         return false;
      }
   }
   return mlir::relalg::detail::canColumnReach(this->getOperation(), source, target, col);
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

   for (Attribute attr : getMapping()) {
      auto relationDefAttr = attr.dyn_cast_or_null<ColumnDefAttr>();
      created.insert(&relationDefAttr.getColumn());
   }
   return created;
}
ColumnSet OuterJoinOp::getUsedColumns() {
   auto used = mlir::relalg::detail::getUsedColumns(getOperation());
   for (Attribute attr : this->getOperation()->getAttr("mapping").cast<mlir::ArrayAttr>()) {
      auto relationDefAttr = attr.dyn_cast_or_null<ColumnDefAttr>();
      auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<ArrayAttr>();
      used.insert(ColumnSet::fromArrayAttr(fromExisting));
   }
   return used;
}
ColumnSet OuterJoinOp::getAvailableColumns() {
   ColumnSet renamed;

   for (Attribute attr : getMapping()) {
      auto relationDefAttr = attr.dyn_cast_or_null<ColumnDefAttr>();
      auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<ArrayAttr>();
      renamed.insert(ColumnSet::fromArrayAttr(fromExisting));
   }
   auto availablePreviously = mlir::cast<Operator>(getLeft().getDefiningOp()).getAvailableColumns();
   auto created = getCreatedColumns();
   availablePreviously.insert(created);
   return availablePreviously;
}
bool OuterJoinOp::canColumnReach(Operator source, Operator target, const mlir::tuples::Column* col) {
   if (getRight().getDefiningOp() == source) {
      return false;
   }
   return mlir::relalg::detail::canColumnReach(this->getOperation(), source, target, col);
}
ColumnSet FullOuterJoinOp::getCreatedColumns() {
   ColumnSet created;

   for (Attribute attr : getMapping()) {
      auto relationDefAttr = attr.dyn_cast_or_null<ColumnDefAttr>();
      created.insert(&relationDefAttr.getColumn());
   }
   return created;
}
ColumnSet FullOuterJoinOp::getUsedColumns() {
   auto used = mlir::relalg::detail::getUsedColumns(getOperation());
   for (Attribute attr : this->getOperation()->getAttr("mapping").cast<mlir::ArrayAttr>()) {
      auto relationDefAttr = attr.dyn_cast_or_null<ColumnDefAttr>();
      auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<ArrayAttr>();
      used.insert(ColumnSet::fromArrayAttr(fromExisting));
   }
   return used;
}
ColumnSet FullOuterJoinOp::getAvailableColumns() {
   return getCreatedColumns();
}
bool FullOuterJoinOp::canColumnReach(Operator source, Operator target, const mlir::tuples::Column* col) {
   if (source == nullptr) {
      //source: full outer join
      return mlir::relalg::detail::canColumnReach(this->getOperation(), source, target, col);
   }
   return false;
}
ColumnSet SingleJoinOp::getCreatedColumns() {
   ColumnSet created;

   for (Attribute attr : getMapping()) {
      auto relationDefAttr = attr.dyn_cast_or_null<ColumnDefAttr>();
      created.insert(&relationDefAttr.getColumn());
   }
   return created;
}
ColumnSet SingleJoinOp::getUsedColumns() {
   auto used = mlir::relalg::detail::getUsedColumns(getOperation());
   for (Attribute attr : this->getOperation()->getAttr("mapping").cast<mlir::ArrayAttr>()) {
      auto relationDefAttr = attr.dyn_cast_or_null<ColumnDefAttr>();
      auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<ArrayAttr>();
      used.insert(ColumnSet::fromArrayAttr(fromExisting));
   }
   return used;
}
ColumnSet SingleJoinOp::getAvailableColumns() {
   ColumnSet renamed;

   for (Attribute attr : getMapping()) {
      auto relationDefAttr = attr.dyn_cast_or_null<ColumnDefAttr>();
      auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<ArrayAttr>();
      renamed.insert(ColumnSet::fromArrayAttr(fromExisting));
   }
   auto availablePreviously = mlir::cast<Operator>(getLeft().getDefiningOp()).getAvailableColumns();
   auto created = getCreatedColumns();
   availablePreviously.insert(created);
   return availablePreviously;
}
bool SingleJoinOp::canColumnReach(Operator source, Operator target, const mlir::tuples::Column* col) {
   if (getRight().getDefiningOp() == source) {
      return false;
   }
   return mlir::relalg::detail::canColumnReach(this->getOperation(), source, target, col);
}
ColumnSet CollectionJoinOp::getCreatedColumns() {
   ColumnSet created;
   created.insert(&getCollAttr().getColumn());
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
bool CollectionJoinOp::canColumnReach(Operator source, Operator target, const mlir::tuples::Column* col) {
   if (getRight().getDefiningOp() == source) {
      return false;
   }
   return mlir::relalg::detail::canColumnReach(this->getOperation(), source, target, col);
}
ColumnSet MarkJoinOp::getCreatedColumns() {
   ColumnSet created;
   created.insert(&getMarkattr().getColumn());
   return created;
}

ColumnSet BaseTableOp::getCreatedColumns() {
   ColumnSet creations;
   for (auto mapping : getColumns()) {
      auto attr = mapping.getValue();
      auto relationDefAttr = attr.dyn_cast_or_null<ColumnDefAttr>();
      creations.insert(&relationDefAttr.getColumn());
   }
   return creations;
}
mlir::relalg::FunctionalDependencies BaseTableOp::getFDs() {
   FunctionalDependencies dependencies;
   auto metaData = getMeta().getMeta();
   if (!metaData->isPresent()) return dependencies;
   if (metaData->getPrimaryKey().empty()) return dependencies;
   auto right = getAvailableColumns();
   std::unordered_set<std::string> pks(metaData->getPrimaryKey().begin(), metaData->getPrimaryKey().end());
   ColumnSet pk;
   for (auto mapping : getColumns()) {
      auto attr = mapping.getValue();
      auto relationDefAttr = attr.dyn_cast_or_null<ColumnDefAttr>();
      if (pks.contains(mapping.getName().str())) {
         pk.insert(&relationDefAttr.getColumn());
      }
   }
   right.remove(pk);
   dependencies.setKey(pk);
   dependencies.insert(pk, right);
   return dependencies;
}
mlir::relalg::FunctionalDependencies mlir::relalg::SelectionOp::getFDs() {
   FunctionalDependencies dependencies = getChildren()[0].getFDs();
   if (!getPredicateBlock().empty()) {
      if (auto returnOp = mlir::dyn_cast_or_null<mlir::tuples::ReturnOp>(getPredicateBlock().getTerminator())) {
         if (returnOp.getResults().size() == 1) {
            if (auto cmpOp = mlir::dyn_cast_or_null<mlir::relalg::CmpOpInterface>(returnOp.getResults()[0].getDefiningOp())) {
               if (cmpOp.isEqualityPred(false)) {
                  if (auto getColLeft = mlir::dyn_cast_or_null<mlir::tuples::GetColumnOp>(cmpOp.getLeft().getDefiningOp())) {
                     if (auto getColRight = mlir::dyn_cast_or_null<mlir::tuples::GetColumnOp>(cmpOp.getRight().getDefiningOp())) {
                        mlir::relalg::ColumnSet left;
                        left.insert(&getColLeft.getAttr().getColumn());
                        mlir::relalg::ColumnSet right;
                        right.insert(&getColRight.getAttr().getColumn());
                        dependencies.insert(left, right);
                        dependencies.insert(right, left);
                     }
                  }
               }
            }
         }
      }
   }
   return dependencies;
}
mlir::relalg::FunctionalDependencies mlir::relalg::AggregationOp::getFDs() {
   FunctionalDependencies dependencies = getChildren()[0].getFDs();
   if (!getGroupByCols().empty()) {
      dependencies.setKey(mlir::relalg::ColumnSet::fromArrayAttr(getGroupByCols()));
   }
   return dependencies;
}
mlir::relalg::FunctionalDependencies mlir::relalg::SemiJoinOp::getFDs() {
   return getChildren()[0].getFDs();
}
mlir::relalg::FunctionalDependencies mlir::relalg::AntiSemiJoinOp::getFDs() {
   return getChildren()[0].getFDs();
}
mlir::relalg::FunctionalDependencies mlir::relalg::InnerJoinOp::getFDs() {
   FunctionalDependencies leftFds = getChildren()[0].getFDs();
   FunctionalDependencies rightFds = getChildren()[1].getFDs();
   FunctionalDependencies newFds;
   if (!getPredicateBlock().empty()) {
      if (this->getOperation()->hasAttr("leftHash") && this->getOperation()->hasAttr("rightHash")) {
         auto left = this->getOperation()->getAttr("leftHash").cast<mlir::ArrayAttr>();
         auto right = this->getOperation()->getAttr("rightHash").cast<mlir::ArrayAttr>();
         if (left.size() == right.size()) { //todo remove! only hackish check to avoid bug related to hackish inl implementation
            for (auto z : llvm::zip(left, right)) {
               auto* leftColumn = &std::get<0>(z).cast<mlir::tuples::ColumnRefAttr>().getColumn();
               auto* rightColumn = &std::get<1>(z).cast<mlir::tuples::ColumnRefAttr>().getColumn();
               mlir::relalg::ColumnSet left;
               left.insert(leftColumn);
               mlir::relalg::ColumnSet right;
               right.insert(rightColumn);
               newFds.insert(left, right);
               newFds.insert(right, left);
            }
         }
      }
      if (auto returnOp = mlir::dyn_cast_or_null<mlir::tuples::ReturnOp>(getPredicateBlock().getTerminator())) {
         if (returnOp.getResults().size() == 1) {
            if (auto cmpOp = mlir::dyn_cast_or_null<mlir::relalg::CmpOpInterface>(returnOp.getResults()[0].getDefiningOp())) {
               if (cmpOp.isEqualityPred(false)) {
                  if (auto getColLeft = mlir::dyn_cast_or_null<mlir::tuples::GetColumnOp>(cmpOp.getLeft().getDefiningOp())) {
                     if (auto getColRight = mlir::dyn_cast_or_null<mlir::tuples::GetColumnOp>(cmpOp.getRight().getDefiningOp())) {
                        mlir::relalg::ColumnSet left;
                        left.insert(&getColLeft.getAttr().getColumn());
                        mlir::relalg::ColumnSet right;
                        right.insert(&getColRight.getAttr().getColumn());
                        newFds.insert(left, right);
                        newFds.insert(right, left);
                     }
                  }
               }
            }
         }
      }
   }
   newFds.insert(leftFds);
   newFds.insert(rightFds);
   if (leftFds.getKey() && rightFds.getKey()) {
      if (newFds.expand(leftFds.getKey().value()).intersect(rightFds.getKey().value()).size() == rightFds.getKey().value().size()) {
         newFds.setKey(leftFds.getKey().value());
      } else if (newFds.expand(rightFds.getKey().value()).intersect(leftFds.getKey().value()).size() == leftFds.getKey().value().size()) {
         newFds.setKey(rightFds.getKey().value());
      }
   }
   return newFds;
}
ColumnSet mlir::relalg::AggregationOp::getAvailableColumns() {
   ColumnSet available = getCreatedColumns();
   available.insert(ColumnSet::fromArrayAttr(getGroupByCols()));
   return available;
}

bool AggregationOp::canColumnReach(Operator source, Operator target, const mlir::tuples::Column* col) {
   ColumnSet available = getCreatedColumns();
   available.insert(ColumnSet::fromArrayAttr(getGroupByCols()));
   if (available.contains(col)) {
      return mlir::relalg::detail::canColumnReach(this->getOperation(), source, target, col);
   }
   return false;
}

ColumnSet mlir::relalg::ProjectionOp::getAvailableColumns() {
   return ColumnSet::fromArrayAttr(getCols());
}
ColumnSet mlir::relalg::ProjectionOp::getUsedColumns() {
   return getSetSemantic() == SetSemantic::distinct ? ColumnSet::fromArrayAttr(getCols()) : ColumnSet();
}
bool ProjectionOp::canColumnReach(Operator source, Operator target, const mlir::tuples::Column* col) {
   if (ColumnSet::fromArrayAttr(getCols()).contains(col)) {
      return mlir::relalg::detail::canColumnReach(this->getOperation(), source, target, col);
   }
   return false;
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
      builder.create<mlir::tuples::ReturnOp>(op->getLoc(), anded);
   } else {
      builder.create<mlir::tuples::ReturnOp>(op->getLoc(), additionalPred);
   }
   terminator->erase();
}
void mlir::relalg::detail::initPredicate(mlir::Operation* op) {
   auto* context = op->getContext();
   mlir::Type tupleType = mlir::tuples::TupleType::get(context);
   auto* block = new mlir::Block;
   op->getRegion(0).push_back(block);
   block->addArgument(tupleType, op->getLoc());
   mlir::OpBuilder builder(context);
   builder.setInsertionPointToStart(block);
   builder.create<mlir::tuples::ReturnOp>(op->getLoc());
}

void mlir::relalg::detail::inlineOpIntoBlock(mlir::Operation* vop, mlir::Operation* includeChildren, mlir::Block* newBlock, mlir::IRMapping& mapping, mlir::Operation* first) {
   llvm::SmallVector<mlir::Operation*, 8> extracted;
   llvm::SmallPtrSet<mlir::Operation*, 8> alreadyPresent;
   addRequirements(vop, includeChildren, newBlock, extracted, alreadyPresent, mapping);
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
   if(tree->isBeforeInBlock(before)){
      return;
   }
   tree->moveBefore(before);
   for (auto child : tree.getChildren()) {
      moveSubTreeBefore(child, tree);
   }
}

mlir::LogicalResult mlir::relalg::MapOp::foldColumns(mlir::relalg::ColumnFoldInfo& columnInfo) {
   replaceColumnUsesInLamda(getContext(), getPredicate().front(), columnInfo);
   auto returnOp = mlir::cast<mlir::tuples::ReturnOp>(getPredicate().front().getTerminator());
   for (auto z : llvm::zip(returnOp.getResults(), getComputedCols())) {
      if (auto getColumnOp = mlir::dyn_cast_or_null<mlir::tuples::GetColumnOp>(std::get<0>(z).getDefiningOp())) {
         auto* previousColumn = &getColumnOp.getAttr().getColumn();
         auto* currentColumn = &std::get<1>(z).cast<mlir::tuples::ColumnDefAttr>().getColumn();
         columnInfo.directMappings[currentColumn] = previousColumn;
      }
   }
   return success();
}
mlir::LogicalResult mlir::relalg::MapOp::eliminateDeadColumns(mlir::relalg::ColumnSet& usedColumns, mlir::Value& newStream) {
   auto returnOp = mlir::cast<mlir::tuples::ReturnOp>(getPredicate().front().getTerminator());
   std::vector<mlir::Value> results;
   std::vector<mlir::Attribute> resultingColumns;
   for (auto z : llvm::zip(returnOp.getResults(), getComputedCols())) {
      auto defAttr = std::get<1>(z).cast<mlir::tuples::ColumnDefAttr>();
      if (usedColumns.contains(&defAttr.getColumn())) {
         results.push_back(std::get<0>(z));
         resultingColumns.push_back(defAttr);
      }
   }
   if (results.size() == returnOp.getNumOperands()) {
      return mlir::failure();
   }
   if (results.size() == 0) {
      newStream = this->getRel();
      return mlir::success();
   }
   returnOp->setOperands(results);
   setComputedColsAttr(mlir::ArrayAttr::get(getContext(), resultingColumns));
   return mlir::success();
}
mlir::LogicalResult mlir::relalg::SelectionOp::foldColumns(mlir::relalg::ColumnFoldInfo& columnInfo) {
   replaceColumnUsesInLamda(getContext(), getPredicate().front(), columnInfo);
   return success();
}
mlir::LogicalResult mlir::relalg::CrossProductOp::foldColumns(mlir::relalg::ColumnFoldInfo& columnInfo) {
   return success();
}
mlir::LogicalResult mlir::relalg::InnerJoinOp::foldColumns(mlir::relalg::ColumnFoldInfo& columnInfo) {
   replaceColumnUsesInLamda(getContext(), getPredicate().front(), columnInfo);
   return success();
}
mlir::LogicalResult mlir::relalg::SemiJoinOp::foldColumns(mlir::relalg::ColumnFoldInfo& columnInfo) {
   replaceColumnUsesInLamda(getContext(), getPredicate().front(), columnInfo);
   return success();
}
mlir::LogicalResult mlir::relalg::AntiSemiJoinOp::foldColumns(mlir::relalg::ColumnFoldInfo& columnInfo) {
   replaceColumnUsesInLamda(getContext(), getPredicate().front(), columnInfo);
   return success();
}
ColumnSet mlir::relalg::NestedOp::getCreatedColumns() {
   return getAvailableColumns(); //todo: fix
}

ColumnSet mlir::relalg::NestedOp::getUsedColumns() {
   return mlir::relalg::ColumnSet::fromArrayAttr(getUsedCols());
}
ColumnSet mlir::relalg::NestedOp::getAvailableColumns() {
   return mlir::relalg::ColumnSet::fromArrayAttr(getAvailableCols());
}
bool mlir::relalg::NestedOp::canColumnReach(Operator source, Operator target, const mlir::tuples::Column* col) {
   ColumnSet available = getAvailableColumns();
   if (available.contains(col)) {
      return mlir::relalg::detail::canColumnReach(this->getOperation(), source, target, col);
   }
   return false;
}
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.cpp.inc"
