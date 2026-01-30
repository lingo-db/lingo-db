#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"

#include <functional>
#include <unordered_set>

using operator_list = llvm::SmallVector<Operator, 4>;
namespace {
using namespace lingodb::compiler::dialect;
using namespace relalg;
using namespace tuples;
operator_list getChildOperators(mlir::Operation* parent) {
   operator_list children;
   for (auto operand : parent->getOperands()) {
      if (auto childOperator = mlir::dyn_cast_or_null<Operator>(operand.getDefiningOp())) {
         children.push_back(childOperator);
      }
   }
   return children;
}

relalg::ColumnSet collectColumns(operator_list operators, std::function<relalg::ColumnSet(Operator)> fn) {
   relalg::ColumnSet collected;
   for (auto op : operators) {
      auto res = fn(op);
      collected.insert(res);
   }
   return collected;
}
relalg::ColumnSet collectColumnsWithCache(operator_list operators, relalg::AvailabilityCache& cache) {
   relalg::ColumnSet collected;
   for (auto op : operators) {
      auto res = cache.getAvailableColumnsFor(op);
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
void replaceColumnUsesInLamda(mlir::MLIRContext* context, mlir::Block& block, const relalg::ColumnFoldInfo& columnInfo) {
   auto& colManager = context->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   block.walk([&columnInfo, &colManager](tuples::GetColumnOp getColumnOp) {
      auto* currColumn = &getColumnOp.getAttr().getColumn();
      if (columnInfo.directMappings.contains(currColumn)) {
         getColumnOp.setAttrAttr(colManager.createRef(columnInfo.directMappings.at(currColumn)));
      }
   });
}
} // namespace

bool relalg::detail::canColumnReach(mlir::Operation* currentOp, mlir::Operation* sourceOp, mlir::Operation* targetOp, const tuples::Column* column) {
   if (currentOp == targetOp) {
      return true;
   }
   for (auto res : currentOp->getResults()) {
      if (mlir::isa<tuples::TupleStreamType>(res.getType())) {
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
ColumnSet relalg::detail::getUsedColumns(mlir::Operation* op) {
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
ColumnSet relalg::detail::getAvailableColumns(mlir::Operation* op, AvailabilityCache& cache) {
   Operator asOperator = mlir::dyn_cast_or_null<Operator>(op);
   auto collected = collectColumnsWithCache(getChildOperators(op), cache);
   auto selfCreated = asOperator.getCreatedColumns();
   collected.insert(selfCreated);
   return collected;
}
FunctionalDependencies relalg::detail::getFDs(mlir::Operation* op) {
   FunctionalDependencies dependencies;
   for (auto child : getChildOperators(op)) {
      dependencies.insert(child.getFDs());
   }
   return dependencies;
}
ColumnSet relalg::detail::getFreeColumns(mlir::Operation* op, AvailabilityCache& cache) {
   auto available = collectColumnsWithCache(getChildOperators(op), cache);
   auto collectedFree = collectColumns(getChildOperators(op), [](Operator op) { return op.getFreeColumns(); });
   auto used = mlir::cast<Operator>(op).getUsedColumns();
   collectedFree.insert(used);
   collectedFree.remove(available);
   return collectedFree;
}

bool relalg::detail::isDependentJoin(mlir::Operation* op, AvailabilityCache& cache) {
   if (auto join = mlir::dyn_cast_or_null<BinaryOperator>(op)) {
      if (isJoin(op)) {
         auto left = mlir::dyn_cast_or_null<Operator>(join.leftChild());
         auto right = mlir::dyn_cast_or_null<Operator>(join.rightChild());
         auto availableLeft = left.getAvailableColumns(cache);
         auto availableRight = right.getAvailableColumns(cache);
         return left.getFreeColumns().intersects(availableRight) || right.getFreeColumns().intersects(availableLeft);
      }
   }
   return false;
}
relalg::detail::BinaryOperatorType relalg::detail::getBinaryOperatorType(mlir::Operation* op) {
   return ::llvm::TypeSwitch<mlir::Operation*, BinaryOperatorType>(op)
      .Case<relalg::UnionOp>([&](mlir::Operation* op) { return BinaryOperatorType::Union; })
      .Case<relalg::IntersectOp>([&](mlir::Operation* op) { return BinaryOperatorType::Intersection; })
      .Case<relalg::ExceptOp>([&](mlir::Operation* op) { return BinaryOperatorType::Except; })
      .Case<relalg::CrossProductOp>([&](mlir::Operation* op) { return BinaryOperatorType::CP; })
      .Case<relalg::InnerJoinOp>([&](mlir::Operation* op) { return BinaryOperatorType::InnerJoin; })
      .Case<relalg::SemiJoinOp>([&](mlir::Operation* op) { return BinaryOperatorType::SemiJoin; })
      .Case<relalg::AntiSemiJoinOp>([&](mlir::Operation* op) { return BinaryOperatorType::AntiSemiJoin; })
      .Case<relalg::SingleJoinOp>([&](mlir::Operation* op) { return BinaryOperatorType::OuterJoin; })
      .Case<relalg::MarkJoinOp>([&](mlir::Operation* op) { return BinaryOperatorType::MarkJoin; })
      .Case<relalg::CollectionJoinOp>([&](mlir::Operation* op) { return BinaryOperatorType::CollectionJoin; })
      .Case<relalg::OuterJoinOp>([&](relalg::OuterJoinOp op) { return BinaryOperatorType::OuterJoin; })
      .Case<relalg::FullOuterJoinOp>([&](relalg::FullOuterJoinOp op) { return BinaryOperatorType::FullOuterJoin; })
      .Default([&](auto x) {
         return BinaryOperatorType::None;
      });
}
relalg::detail::UnaryOperatorType relalg::detail::getUnaryOperatorType(mlir::Operation* op) {
   return ::llvm::TypeSwitch<mlir::Operation*, UnaryOperatorType>(op)
      .Case<relalg::SelectionOp>([&](mlir::Operation* op) { return UnaryOperatorType::Selection; })
      .Case<relalg::MapOp>([&](mlir::Operation* op) { return UnaryOperatorType::Map; })
      .Case<relalg::ProjectionOp>([&](relalg::ProjectionOp op) { return op.getSetSemantic() == relalg::SetSemantic::distinct ? UnaryOperatorType::DistinctProjection : UnaryOperatorType::Projection; })
      .Case<relalg::AggregationOp>([&](mlir::Operation* op) { return UnaryOperatorType::Aggregation; })
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
   auto used = relalg::detail::getUsedColumns(getOperation());
   used.insert(ColumnSet::fromArrayAttr(getGroupByCols()));
   getOperation()->walk([&](relalg::AggrFuncOp aggrFn) {
      used.insert(&aggrFn.getAttr().getColumn());
   });
   return used;
}
ColumnSet GroupJoinOp::getCreatedColumns() {
   return ColumnSet::fromArrayAttr(getComputedCols());
}
ColumnSet GroupJoinOp::getUsedColumns() {
   auto used = relalg::detail::getUsedColumns(getOperation());
   used.insert(ColumnSet::fromArrayAttr(getLeftCols()));
   used.insert(ColumnSet::fromArrayAttr(getRightCols()));
   getOperation()->walk([&](relalg::AggrFuncOp aggrFn) {
      used.insert(&aggrFn.getAttr().getColumn());
   });
   return used;
}
ColumnSet GroupJoinOp::getAvailableColumns(AvailabilityCache&) {
   ColumnSet available = getCreatedColumns();
   available.insert(ColumnSet::fromArrayAttr(getLeftCols()));
   available.insert(ColumnSet::fromArrayAttr(getRightCols()));
   return available;
}
bool GroupJoinOp::canColumnReach(Operator source, Operator target, const tuples::Column* col) {
   AvailabilityCache cache;
   ColumnSet available = getAvailableColumns(cache);
   if (available.contains(col)) {
      return relalg::detail::canColumnReach(this->getOperation(), source, target, col);
   }
   return false;
}
ColumnSet WindowOp::getCreatedColumns() {
   return ColumnSet::fromArrayAttr(getComputedCols());
}
ColumnSet WindowOp::getUsedColumns() {
   auto used = relalg::detail::getUsedColumns(getOperation());
   used.insert(ColumnSet::fromArrayAttr(getPartitionBy()));
   getOperation()->walk([&](relalg::AggrFuncOp aggrFn) {
      used.insert(&aggrFn.getAttr().getColumn());
   });
   for (mlir::Attribute a : getOrderBy()) {
      used.insert(&mlir::dyn_cast_or_null<relalg::SortSpecificationAttr>(a).getAttr().getColumn());
   }
   return used;
}
ColumnSet SortOp::getUsedColumns() {
   ColumnSet used;
   for (mlir::Attribute a : getSortspecs()) {
      used.insert(&mlir::dyn_cast_or_null<relalg::SortSpecificationAttr>(a).getAttr().getColumn());
   }
   return used;
}
ColumnSet TopKOp::getUsedColumns() {
   ColumnSet used;
   for (mlir::Attribute a : getSortspecs()) {
      used.insert(&mlir::dyn_cast_or_null<relalg::SortSpecificationAttr>(a).getAttr().getColumn());
   }
   return used;
}

ColumnSet ConstRelationOp::getCreatedColumns() {
   return ColumnSet::fromArrayAttr(getColumns());
}
ColumnSet AntiSemiJoinOp::getAvailableColumns(AvailabilityCache& cache) {
   return cache.getAvailableColumnsFor(mlir::cast<Operator>(leftChild()));
}
bool AntiSemiJoinOp::canColumnReach(Operator source, Operator target, const tuples::Column* col) {
   if (getRight().getDefiningOp() == source) {
      return false;
   }
   return relalg::detail::canColumnReach(this->getOperation(), source, target, col);
}
ColumnSet SemiJoinOp::getAvailableColumns(AvailabilityCache& cache) {
   return cache.getAvailableColumnsFor(mlir::cast<Operator>(leftChild()));
}
bool SemiJoinOp::canColumnReach(Operator source, Operator target, const tuples::Column* col) {
   if (getRight().getDefiningOp() == source) {
      return false;
   }
   return relalg::detail::canColumnReach(this->getOperation(), source, target, col);
}
ColumnSet MarkJoinOp::getAvailableColumns(AvailabilityCache& cache) {
   auto available = cache.getAvailableColumnsFor(mlir::cast<Operator>(leftChild()));
   available.insert(&getMarkattr().getColumn());
   return available;
}
bool MarkJoinOp::canColumnReach(Operator source, Operator target, const tuples::Column* col) {
   if (getRight().getDefiningOp() == source) {
      return false;
   }
   return relalg::detail::canColumnReach(this->getOperation(), source, target, col);
}
ColumnSet RenamingOp::getCreatedColumns() {
   ColumnSet created;

   for (mlir::Attribute attr : getColumns()) {
      auto relationDefAttr = mlir::dyn_cast_or_null<ColumnDefAttr>(attr);
      created.insert(&relationDefAttr.getColumn());
   }
   return created;
}
ColumnSet RenamingOp::getUsedColumns() {
   ColumnSet used;

   for (mlir::Attribute attr : getColumns()) {
      auto relationDefAttr = mlir::dyn_cast_or_null<ColumnDefAttr>(attr);
      auto fromExisting = mlir::dyn_cast_or_null<mlir::ArrayAttr>(relationDefAttr.getFromExisting());
      used.insert(ColumnSet::fromArrayAttr(fromExisting));
   }
   return used;
}
ColumnSet RenamingOp::getAvailableColumns(AvailabilityCache& cache) {
   auto availablePreviously = collectColumnsWithCache(getChildOperators(*this), cache);
   availablePreviously.remove(getUsedColumns());
   auto created = getCreatedColumns();
   availablePreviously.insert(created);
   return availablePreviously;
}
bool relalg::RenamingOp::canColumnReach(Operator source, Operator target, const tuples::Column* col) {
   for (mlir::Attribute attr : getColumns()) {
      auto relationDefAttr = mlir::dyn_cast_or_null<ColumnDefAttr>(attr);
      auto fromExisting = mlir::dyn_cast_or_null<mlir::ArrayAttr>(relationDefAttr.getFromExisting());
      if (&mlir::cast<tuples::ColumnRefAttr>(fromExisting[0]).getColumn() == col) {
         return false;
      }
   }
   return relalg::detail::canColumnReach(this->getOperation(), source, target, col);
}
ColumnSet relalg::detail::getSetOpCreatedColumns(mlir::Operation* op) {
   ColumnSet created;
   for (mlir::Attribute attr : mlir::cast<mlir::ArrayAttr>(op->getAttr("mapping"))) {
      auto relationDefAttr = mlir::dyn_cast_or_null<ColumnDefAttr>(attr);
      created.insert(&relationDefAttr.getColumn());
   }
   return created;
}
ColumnSet relalg::detail::getSetOpUsedColumns(mlir::Operation* op) {
   ColumnSet used;
   for (mlir::Attribute attr : mlir::cast<mlir::ArrayAttr>(op->getAttr("mapping"))) {
      auto relationDefAttr = mlir::cast<ColumnDefAttr>(attr);
      auto fromExisting = mlir::cast<mlir::ArrayAttr>(relationDefAttr.getFromExisting());
      used.insert(ColumnSet::fromArrayAttr(fromExisting));
   }
   return used;
}
ColumnSet OuterJoinOp::getCreatedColumns() {
   ColumnSet created;

   for (mlir::Attribute attr : getMapping()) {
      auto relationDefAttr = mlir::dyn_cast_or_null<ColumnDefAttr>(attr);
      created.insert(&relationDefAttr.getColumn());
   }
   return created;
}
ColumnSet OuterJoinOp::getUsedColumns() {
   auto used = relalg::detail::getUsedColumns(getOperation());
   for (mlir::Attribute attr : mlir::cast<mlir::ArrayAttr>(this->getOperation()->getAttr("mapping"))) {
      auto relationDefAttr = mlir::dyn_cast_or_null<ColumnDefAttr>(attr);
      auto fromExisting = mlir::dyn_cast_or_null<mlir::ArrayAttr>(relationDefAttr.getFromExisting());
      used.insert(ColumnSet::fromArrayAttr(fromExisting));
   }
   return used;
}
ColumnSet OuterJoinOp::getAvailableColumns(AvailabilityCache& cache) {
   ColumnSet renamed;

   for (mlir::Attribute attr : getMapping()) {
      auto relationDefAttr = mlir::dyn_cast_or_null<ColumnDefAttr>(attr);
      auto fromExisting = mlir::cast<mlir::ArrayAttr>(relationDefAttr.getFromExisting());
      renamed.insert(ColumnSet::fromArrayAttr(fromExisting));
   }
   auto availablePreviously = cache.getAvailableColumnsFor(mlir::cast<Operator>(getLeft().getDefiningOp()));
   auto created = getCreatedColumns();
   availablePreviously.insert(created);
   return availablePreviously;
}
bool OuterJoinOp::canColumnReach(Operator source, Operator target, const tuples::Column* col) {
   if (getRight().getDefiningOp() == source) {
      return false;
   }
   return relalg::detail::canColumnReach(this->getOperation(), source, target, col);
}
ColumnSet FullOuterJoinOp::getCreatedColumns() {
   ColumnSet created;

   for (mlir::Attribute attr : getMapping()) {
      auto relationDefAttr = mlir::dyn_cast_or_null<ColumnDefAttr>(attr);
      created.insert(&relationDefAttr.getColumn());
   }
   return created;
}
ColumnSet FullOuterJoinOp::getUsedColumns() {
   auto used = relalg::detail::getUsedColumns(getOperation());
   for (mlir::Attribute attr : mlir::cast<mlir::ArrayAttr>(this->getOperation()->getAttr("mapping"))) {
      auto relationDefAttr = mlir::dyn_cast_or_null<ColumnDefAttr>(attr);
      auto fromExisting = mlir::cast<mlir::ArrayAttr>(relationDefAttr.getFromExisting());
      used.insert(ColumnSet::fromArrayAttr(fromExisting));
   }
   return used;
}
ColumnSet FullOuterJoinOp::getAvailableColumns(AvailabilityCache&) {
   return getCreatedColumns();
}
bool FullOuterJoinOp::canColumnReach(Operator source, Operator target, const tuples::Column* col) {
   if (source == nullptr) {
      //source: full outer join
      return relalg::detail::canColumnReach(this->getOperation(), source, target, col);
   }
   return false;
}
ColumnSet SingleJoinOp::getCreatedColumns() {
   ColumnSet created;

   for (mlir::Attribute attr : getMapping()) {
      auto relationDefAttr = mlir::dyn_cast_or_null<ColumnDefAttr>(attr);
      created.insert(&relationDefAttr.getColumn());
   }
   return created;
}
ColumnSet SingleJoinOp::getUsedColumns() {
   auto used = relalg::detail::getUsedColumns(getOperation());
   for (mlir::Attribute attr : mlir::cast<mlir::ArrayAttr>(this->getOperation()->getAttr("mapping"))) {
      auto relationDefAttr = mlir::dyn_cast_or_null<ColumnDefAttr>(attr);
      auto fromExisting = mlir::cast<mlir::ArrayAttr>(relationDefAttr.getFromExisting());
      used.insert(ColumnSet::fromArrayAttr(fromExisting));
   }
   return used;
}
ColumnSet SingleJoinOp::getAvailableColumns(AvailabilityCache& cache) {
   ColumnSet renamed;

   for (mlir::Attribute attr : getMapping()) {
      auto relationDefAttr = mlir::dyn_cast_or_null<ColumnDefAttr>(attr);
      auto fromExisting = mlir::cast<mlir::ArrayAttr>(relationDefAttr.getFromExisting());
      renamed.insert(ColumnSet::fromArrayAttr(fromExisting));
   }
   auto availablePreviously = cache.getAvailableColumnsFor(mlir::cast<Operator>(getLeft().getDefiningOp()));
   auto created = getCreatedColumns();
   availablePreviously.insert(created);
   return availablePreviously;
}
bool SingleJoinOp::canColumnReach(Operator source, Operator target, const tuples::Column* col) {
   if (getRight().getDefiningOp() == source) {
      return false;
   }
   return relalg::detail::canColumnReach(this->getOperation(), source, target, col);
}
ColumnSet CollectionJoinOp::getCreatedColumns() {
   ColumnSet created;
   created.insert(&getCollAttr().getColumn());
   return created;
}
ColumnSet CollectionJoinOp::getUsedColumns() {
   return relalg::detail::getUsedColumns(getOperation());
}
ColumnSet CollectionJoinOp::getAvailableColumns(AvailabilityCache& cache) {
   auto availablePreviously = collectColumnsWithCache(getChildOperators(*this), cache);
   auto created = getCreatedColumns();
   availablePreviously.insert(created);
   return availablePreviously;
}
bool CollectionJoinOp::canColumnReach(Operator source, Operator target, const tuples::Column* col) {
   if (getRight().getDefiningOp() == source) {
      return false;
   }
   return relalg::detail::canColumnReach(this->getOperation(), source, target, col);
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
      auto relationDefAttr = mlir::dyn_cast_or_null<ColumnDefAttr>(attr);
      creations.insert(&relationDefAttr.getColumn());
   }
   return creations;
}
relalg::FunctionalDependencies BaseTableOp::getFDs() {
   FunctionalDependencies dependencies;
   auto metaData = mlir::dyn_cast_or_null<relalg::TableMetaDataAttr>(getOperation()->getAttr("meta"));
   if (!metaData) return dependencies;
   if (metaData.getMeta()->getPrimaryKey().empty()) return dependencies;
   AvailabilityCache cache;
   auto right = getAvailableColumns(cache);
   auto primaryKey = metaData.getMeta()->getPrimaryKey();
   std::unordered_set<std::string> pks(primaryKey.begin(), primaryKey.end());
   ColumnSet pk;
   for (auto mapping : getColumns()) {
      auto attr = mapping.getValue();
      auto relationDefAttr = mlir::dyn_cast_or_null<ColumnDefAttr>(attr);
      if (pks.contains(mapping.getName().str())) {
         pk.insert(&relationDefAttr.getColumn());
      }
   }
   right.remove(pk);
   if (pk.size() == pks.size()) {
      dependencies.setKey(pk);
      dependencies.insert(pk, right);
   }
   return dependencies;
}
relalg::FunctionalDependencies relalg::SelectionOp::getFDs() {
   FunctionalDependencies dependencies = getChildren()[0].getFDs();
   if (!getPredicateBlock().empty()) {
      if (auto returnOp = mlir::dyn_cast_or_null<tuples::ReturnOp>(getPredicateBlock().getTerminator())) {
         if (returnOp.getResults().size() == 1) {
            if (auto cmpOp = mlir::dyn_cast_or_null<relalg::CmpOpInterface>(returnOp.getResults()[0].getDefiningOp())) {
               if (cmpOp.isEqualityPred(false)) {
                  if (auto getColLeft = mlir::dyn_cast_or_null<tuples::GetColumnOp>(cmpOp.getLeft().getDefiningOp())) {
                     if (auto getColRight = mlir::dyn_cast_or_null<tuples::GetColumnOp>(cmpOp.getRight().getDefiningOp())) {
                        relalg::ColumnSet left;
                        left.insert(&getColLeft.getAttr().getColumn());
                        relalg::ColumnSet right;
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
relalg::FunctionalDependencies relalg::AggregationOp::getFDs() {
   FunctionalDependencies dependencies = getChildren()[0].getFDs();
   if (!getGroupByCols().empty()) {
      dependencies.setKey(relalg::ColumnSet::fromArrayAttr(getGroupByCols()));
   }
   return dependencies;
}
relalg::FunctionalDependencies relalg::SemiJoinOp::getFDs() {
   return getChildren()[0].getFDs();
}
relalg::FunctionalDependencies relalg::AntiSemiJoinOp::getFDs() {
   return getChildren()[0].getFDs();
}
relalg::FunctionalDependencies relalg::InnerJoinOp::getFDs() {
   FunctionalDependencies leftFds = getChildren()[0].getFDs();
   FunctionalDependencies rightFds = getChildren()[1].getFDs();
   FunctionalDependencies newFds;
   if (!getPredicateBlock().empty()) {
      if (this->getOperation()->hasAttr("leftHash") && this->getOperation()->hasAttr("rightHash")) {
         auto left = mlir::cast<mlir::ArrayAttr>(this->getOperation()->getAttr("leftHash"));
         auto right = mlir::cast<mlir::ArrayAttr>(this->getOperation()->getAttr("rightHash"));
         if (left.size() == right.size()) { //todo remove! only hackish check to avoid bug related to hackish inl implementation
            for (auto z : llvm::zip(left, right)) {
               auto* leftColumn = &mlir::cast<tuples::ColumnRefAttr>(std::get<0>(z)).getColumn();
               auto* rightColumn = &mlir::cast<tuples::ColumnRefAttr>(std::get<1>(z)).getColumn();
               relalg::ColumnSet left;
               left.insert(leftColumn);
               relalg::ColumnSet right;
               right.insert(rightColumn);
               newFds.insert(left, right);
               newFds.insert(right, left);
            }
         }
      }
      if (auto returnOp = mlir::dyn_cast_or_null<tuples::ReturnOp>(getPredicateBlock().getTerminator())) {
         if (returnOp.getResults().size() == 1) {
            if (auto cmpOp = mlir::dyn_cast_or_null<relalg::CmpOpInterface>(returnOp.getResults()[0].getDefiningOp())) {
               if (cmpOp.isEqualityPred(false)) {
                  if (auto getColLeft = mlir::dyn_cast_or_null<tuples::GetColumnOp>(cmpOp.getLeft().getDefiningOp())) {
                     if (auto getColRight = mlir::dyn_cast_or_null<tuples::GetColumnOp>(cmpOp.getRight().getDefiningOp())) {
                        relalg::ColumnSet left;
                        left.insert(&getColLeft.getAttr().getColumn());
                        relalg::ColumnSet right;
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
ColumnSet relalg::AggregationOp::getAvailableColumns(AvailabilityCache&) {
   ColumnSet available = getCreatedColumns();
   available.insert(ColumnSet::fromArrayAttr(getGroupByCols()));
   return available;
}

bool AggregationOp::canColumnReach(Operator source, Operator target, const tuples::Column* col) {
   ColumnSet available = getCreatedColumns();
   available.insert(ColumnSet::fromArrayAttr(getGroupByCols()));
   if (available.contains(col)) {
      return relalg::detail::canColumnReach(this->getOperation(), source, target, col);
   }
   return false;
}

ColumnSet relalg::ProjectionOp::getAvailableColumns(AvailabilityCache&) {
   return ColumnSet::fromArrayAttr(getCols());
}
ColumnSet relalg::ProjectionOp::getUsedColumns() {
   return getSetSemantic() == SetSemantic::distinct ? ColumnSet::fromArrayAttr(getCols()) : ColumnSet();
}
bool ProjectionOp::canColumnReach(Operator source, Operator target, const tuples::Column* col) {
   if (ColumnSet::fromArrayAttr(getCols()).contains(col)) {
      return relalg::detail::canColumnReach(this->getOperation(), source, target, col);
   }
   return false;
}
bool relalg::detail::isJoin(mlir::Operation* op) {
   auto opType = getBinaryOperatorType(op);
   return BinaryOperatorType::InnerJoin <= opType && opType <= BinaryOperatorType::CollectionJoin;
}

void relalg::detail::addPredicate(mlir::Operation* op, std::function<mlir::Value(mlir::Value, mlir::OpBuilder&)> predicateProducer) {
   auto lambdaOperator = mlir::dyn_cast_or_null<PredicateOperator>(op);
   auto* terminator = lambdaOperator.getPredicateBlock().getTerminator();
   mlir::OpBuilder builder(terminator);
   auto additionalPred = predicateProducer(lambdaOperator.getPredicateArgument(), builder);
   if (terminator->getNumOperands() > 0) {
      mlir::Value oldValue = terminator->getOperand(0);
      bool nullable = mlir::isa<db::NullableType>(oldValue.getType()) || mlir::isa<db::NullableType>(additionalPred.getType());
      mlir::Type restype = builder.getI1Type();
      if (nullable) {
         restype = db::NullableType::get(builder.getContext(), restype);
      }
      mlir::Value anded = builder.create<db::AndOp>(op->getLoc(), restype, mlir::ValueRange({oldValue, additionalPred}));
      builder.create<tuples::ReturnOp>(op->getLoc(), anded);
   } else {
      builder.create<tuples::ReturnOp>(op->getLoc(), additionalPred);
   }
   terminator->erase();
}
void relalg::detail::initPredicate(mlir::Operation* op) {
   auto* context = op->getContext();
   mlir::Type tupleType = tuples::TupleType::get(context);
   auto* block = new mlir::Block;
   op->getRegion(0).push_back(block);
   block->addArgument(tupleType, op->getLoc());
   mlir::OpBuilder builder(context);
   builder.setInsertionPointToStart(block);
   builder.create<tuples::ReturnOp>(op->getLoc());
}

void relalg::detail::inlineOpIntoBlock(mlir::Operation* vop, mlir::Operation* includeChildren, mlir::Block* newBlock, mlir::IRMapping& mapping, mlir::Operation* first) {
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
void relalg::detail::moveSubTreeBefore(mlir::Operation* op, mlir::Operation* before) {
   auto tree = mlir::dyn_cast_or_null<Operator>(op);
   if (tree->isBeforeInBlock(before)) {
      return;
   }
   tree->moveBefore(before);
   for (auto child : tree.getChildren()) {
      moveSubTreeBefore(child, tree);
   }
}

mlir::LogicalResult relalg::MapOp::foldColumns(relalg::ColumnFoldInfo& columnInfo) {
   replaceColumnUsesInLamda(getContext(), getPredicate().front(), columnInfo);
   auto returnOp = mlir::cast<tuples::ReturnOp>(getPredicate().front().getTerminator());
   for (auto z : llvm::zip(returnOp.getResults(), getComputedCols())) {
      if (auto getColumnOp = mlir::dyn_cast_or_null<tuples::GetColumnOp>(std::get<0>(z).getDefiningOp())) {
         auto* previousColumn = &getColumnOp.getAttr().getColumn();
         auto* currentColumn = &mlir::cast<tuples::ColumnDefAttr>(std::get<1>(z)).getColumn();
         columnInfo.directMappings[currentColumn] = previousColumn;
      }
   }
   return mlir::success();
}
mlir::LogicalResult relalg::MapOp::eliminateDeadColumns(relalg::ColumnSet& usedColumns, mlir::Value& newStream) {
   auto returnOp = mlir::cast<tuples::ReturnOp>(getPredicate().front().getTerminator());
   std::vector<mlir::Value> results;
   std::vector<mlir::Attribute> resultingColumns;
   for (auto z : llvm::zip(returnOp.getResults(), getComputedCols())) {
      auto defAttr = mlir::cast<tuples::ColumnDefAttr>(std::get<1>(z));
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
mlir::LogicalResult relalg::SelectionOp::foldColumns(relalg::ColumnFoldInfo& columnInfo) {
   replaceColumnUsesInLamda(getContext(), getPredicate().front(), columnInfo);
   return mlir::success();
}
namespace {
void propagateNonNull(mlir::Value val) {
   std::vector<mlir::Operation*> users(val.getUsers().begin(), val.getUsers().end());
   for (auto* user : users) {
      if (auto canChange = mlir::dyn_cast<lingodb::compiler::dialect::db::SupportsNullabilityChange>(user)) {
         mlir::Type type = canChange->getResult(0).getType();
         mlir::Type newType = canChange.getChangedResultType();
         if (type != newType) {
            canChange->getResult(0).setType(newType);
            propagateNonNull(canChange->getResult(0));
         }
      } else if (mlir::isa<tuples::ReturnOp>(user)) {
      } else {
         mlir::OpBuilder builder(val.getContext());
         builder.setInsertionPointAfter(val.getDefiningOp());
         auto asNullableOp = builder.create<db::AsNullableOp>(val.getDefiningOp()->getLoc(), db::NullableType::get(val.getType()), val);
         val.replaceUsesWithIf(asNullableOp.getRes(), [&](mlir::OpOperand& operand) { return operand.getOwner() == user; });
      }
   }
}
mlir::LogicalResult handleChangeForPredicate(PredicateOperator op, lingodb::compiler::dialect::relalg::ColumnNullableChangeInfo& columnInfo) {
   if (op.getPredicateRegion().empty()) return mlir::success();
   auto predicateArg = op.getPredicateArgument();
   for (auto* user : predicateArg.getUsers()) {
      auto getColumnOp = mlir::cast<tuples::GetColumnOp>(user);
      auto* oldCol = &getColumnOp.getAttr().getColumn();
      if (columnInfo.directMappings.contains(oldCol)) {
         auto* newCol = columnInfo.directMappings[oldCol];
         auto& colManager = op.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
         getColumnOp.setAttrAttr(colManager.createRef(newCol));
         getColumnOp.getRes().setType(newCol->type);
         propagateNonNull(getColumnOp.getRes());
      }
   }
   return mlir::success();
}
mlir::ArrayAttr updateOJMapping(mlir::ArrayAttr input, lingodb::compiler::dialect::relalg::ColumnNullableChangeInfo& columnInfo) {
   auto& colManager = input.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   std::vector<mlir::Attribute> newColDefs;
   for (auto attr : input) {
      auto colDef = mlir::cast<tuples::ColumnDefAttr>(attr);
      auto colRef = mlir::cast<tuples::ColumnRefAttr>(mlir::cast<mlir::ArrayAttr>(colDef.getFromExisting())[0]);
      if (columnInfo.directMappings.contains(&colRef.getColumn())) {
         mlir::Attribute newColRef = colManager.createRef(columnInfo.directMappings[&colRef.getColumn()]);
         auto [scope, name] = colManager.getName(&colDef.getColumn());
         newColDefs.push_back(colManager.createDef(&colDef.getColumn(), mlir::ArrayAttr::get(input.getContext(), {newColRef})));
      } else {
         newColDefs.push_back(colDef);
      }
   }
   return mlir::ArrayAttr::get(input.getContext(), newColDefs);
}
} // namespace
mlir::LogicalResult relalg::SelectionOp::changeForColumns(lingodb::compiler::dialect::relalg::ColumnNullableChangeInfo& columnInfo) {
   return handleChangeForPredicate(*this, columnInfo);
}
mlir::LogicalResult relalg::InnerJoinOp::changeForColumns(lingodb::compiler::dialect::relalg::ColumnNullableChangeInfo& columnInfo) {
   return handleChangeForPredicate(*this, columnInfo);
}
mlir::LogicalResult relalg::AntiSemiJoinOp::changeForColumns(lingodb::compiler::dialect::relalg::ColumnNullableChangeInfo& columnInfo) {
   return handleChangeForPredicate(*this, columnInfo);
}
mlir::LogicalResult relalg::CollectionJoinOp::changeForColumns(lingodb::compiler::dialect::relalg::ColumnNullableChangeInfo& columnInfo) {
   return handleChangeForPredicate(*this, columnInfo);
}
mlir::LogicalResult relalg::FullOuterJoinOp::changeForColumns(lingodb::compiler::dialect::relalg::ColumnNullableChangeInfo& columnInfo) {
   setMappingAttr(updateOJMapping(getMapping(), columnInfo));
   return handleChangeForPredicate(*this, columnInfo);
}
mlir::LogicalResult relalg::MarkJoinOp::changeForColumns(lingodb::compiler::dialect::relalg::ColumnNullableChangeInfo& columnInfo) {
   return handleChangeForPredicate(*this, columnInfo);
}
mlir::LogicalResult relalg::SemiJoinOp::changeForColumns(lingodb::compiler::dialect::relalg::ColumnNullableChangeInfo& columnInfo) {
   return handleChangeForPredicate(*this, columnInfo);
}
mlir::LogicalResult relalg::SingleJoinOp::changeForColumns(lingodb::compiler::dialect::relalg::ColumnNullableChangeInfo& columnInfo) {
   setMappingAttr(updateOJMapping(getMapping(), columnInfo));
   return handleChangeForPredicate(*this, columnInfo);
}
mlir::LogicalResult relalg::OuterJoinOp::changeForColumns(lingodb::compiler::dialect::relalg::ColumnNullableChangeInfo& columnInfo) {
   setMappingAttr(updateOJMapping(getMapping(), columnInfo));
   return handleChangeForPredicate(*this, columnInfo);
}
mlir::LogicalResult relalg::MapOp::changeForColumns(lingodb::compiler::dialect::relalg::ColumnNullableChangeInfo& columnInfo) {
   auto& colManager = getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   if (getPredicate().empty()) return mlir::success();
   auto predicateArg = getLambdaArgument();
   for (auto* user : predicateArg.getUsers()) {
      auto getColumnOp = mlir::cast<tuples::GetColumnOp>(user);
      auto* oldCol = &getColumnOp.getAttr().getColumn();
      if (columnInfo.directMappings.contains(oldCol)) {
         auto* newCol = columnInfo.directMappings[oldCol];
         getColumnOp.setAttrAttr(colManager.createRef(newCol));
         getColumnOp.getRes().setType(newCol->type);
         propagateNonNull(getColumnOp.getRes());
      }
   }

   auto returnOp = mlir::cast<tuples::ReturnOp>(getPredicate().front().getTerminator());
   std::vector<mlir::Attribute> newAttrs;
   for (auto [retVal, attr] : llvm::zip(returnOp.getResults(), getComputedCols())) {
      auto colDef = mlir::cast<tuples::ColumnDefAttr>(attr);
      if (colDef.getColumn().type != retVal.getType()) {
         auto [scope, name] = colManager.getName(&colDef.getColumn());
         auto newColDef = colManager.createDef(scope, name + "__notnull");
         newColDef.getColumn().type = getBaseType(colDef.getColumn().type);
         columnInfo.directMappings[&colDef.getColumn()] = &newColDef.getColumn();
         newAttrs.push_back(newColDef);
      } else {
         newAttrs.push_back(colDef);
      }
   }
   setComputedColsAttr(mlir::ArrayAttr::get(getContext(), newAttrs));
   return mlir::success();
}
mlir::LogicalResult relalg::AggregationOp::changeForColumns(lingodb::compiler::dialect::relalg::ColumnNullableChangeInfo& columnInfo) {
   auto& colManager = getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   std::vector<mlir::Attribute> newKeyAttrs;
   tuples::ReturnOp returnOp;
   for (auto keyAttr : getGroupByCols()) {
      auto colRef = mlir::cast<tuples::ColumnRefAttr>(keyAttr);
      if (columnInfo.directMappings.contains(&colRef.getColumn())) {
         newKeyAttrs.push_back(colManager.createRef(columnInfo.directMappings[&colRef.getColumn()]));
      } else {
         newKeyAttrs.push_back(keyAttr);
      }
   }
   getAggrFunc().walk([&](mlir::Operation* op) {
      if (mlir::isa<relalg::CountRowsOp>(op)) {
      } else if (auto aggrFunc = mlir::dyn_cast<relalg::AggrFuncOp>(op)) {
         if (columnInfo.directMappings.contains(&aggrFunc.getAttr().getColumn())) {
            aggrFunc.setAttrAttr(colManager.createRef(columnInfo.directMappings[&aggrFunc.getAttr().getColumn()]));
            if (!getGroupByCols().empty()) {
               aggrFunc.getResult().setType(getBaseType(aggrFunc.getType()));
               propagateNonNull(aggrFunc.getResult());
            }
         }
      } else if (auto projection = mlir::dyn_cast<relalg::ProjectionOp>(op)) {
         if (projection.changeForColumns(columnInfo).failed()) {
            llvm::errs() << "Failed to change ProjectionOp inside AggregationOp\n";
         }
      } else if (auto rOp = mlir::dyn_cast<tuples::ReturnOp>(op)) {
         returnOp = rOp;
      }
   });
   setGroupByColsAttr(mlir::ArrayAttr::get(getContext(), newKeyAttrs));
   std::vector<mlir::Attribute> newAttrs;
   for (auto [retVal, attr] : llvm::zip(returnOp.getResults(), getComputedCols())) {
      auto colDef = mlir::cast<tuples::ColumnDefAttr>(attr);
      if (colDef.getColumn().type != retVal.getType()) {
         auto [scope, name] = colManager.getName(&colDef.getColumn());
         auto newColDef = colManager.createDef(scope, name + "__notnull");
         newColDef.getColumn().type = getBaseType(colDef.getColumn().type);
         columnInfo.directMappings[&colDef.getColumn()] = &newColDef.getColumn();
         newAttrs.push_back(newColDef);
      } else {
         newAttrs.push_back(colDef);
      }
   }
   setComputedColsAttr(mlir::ArrayAttr::get(getContext(), newAttrs));
   return mlir::success();
}
mlir::LogicalResult relalg::ProjectionOp::changeForColumns(lingodb::compiler::dialect::relalg::ColumnNullableChangeInfo& columnInfo) {
   auto& colManager = getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   std::vector<mlir::Attribute> newKeyAttrs;
   for (auto keyAttr : getCols()) {
      auto colRef = mlir::cast<tuples::ColumnRefAttr>(keyAttr);
      if (columnInfo.directMappings.contains(&colRef.getColumn())) {
         newKeyAttrs.push_back(colManager.createRef(columnInfo.directMappings[&colRef.getColumn()]));
      } else {
         newKeyAttrs.push_back(keyAttr);
      }
   }
   setColsAttr(mlir::ArrayAttr::get(getContext(), newKeyAttrs));
   return mlir::success();
}
mlir::LogicalResult relalg::SortOp::changeForColumns(lingodb::compiler::dialect::relalg::ColumnNullableChangeInfo& columnInfo) {
   auto& colManager = getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   std::vector<mlir::Attribute> newSortSpecAttrs;
   for (auto attr : getSortspecs()) {
      auto sortSpec = mlir::cast<relalg::SortSpecificationAttr>(attr);
      auto colRef = sortSpec.getAttr();
      if (columnInfo.directMappings.contains(&colRef.getColumn())) {
         auto newRef = colManager.createRef(columnInfo.directMappings[&colRef.getColumn()]);
         newSortSpecAttrs.push_back(relalg::SortSpecificationAttr::get(getContext(), newRef, sortSpec.getSortSpec()));
      } else {
         newSortSpecAttrs.push_back(attr);
      }
   }
   setSortspecsAttr(mlir::ArrayAttr::get(getContext(), newSortSpecAttrs));
   return mlir::success();
}
mlir::LogicalResult relalg::LimitOp::changeForColumns(lingodb::compiler::dialect::relalg::ColumnNullableChangeInfo& columnInfo) {
   return mlir::success();
}
mlir::LogicalResult relalg::RenamingOp::changeForColumns(lingodb::compiler::dialect::relalg::ColumnNullableChangeInfo& columnInfo) {
   auto& colManager = getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   std::vector<mlir::Attribute> newColDefs;
   for (auto attr : getColumns()) {
      auto colDef = mlir::cast<tuples::ColumnDefAttr>(attr);
      auto colRef = mlir::cast<tuples::ColumnRefAttr>(mlir::cast<mlir::ArrayAttr>(colDef.getFromExisting())[0]);
      if (columnInfo.directMappings.contains(&colRef.getColumn())) {
         mlir::Attribute newColRef = colManager.createRef(columnInfo.directMappings[&colRef.getColumn()]);
         auto [scope, name] = colManager.getName(&colDef.getColumn());
         auto newColDef = colManager.createDef(scope, name + "__notnull", mlir::ArrayAttr::get(getContext(), {newColRef}));
         newColDef.getColumn().type = getBaseType(colDef.getColumn().type);
         columnInfo.directMappings[&colDef.getColumn()] = &newColDef.getColumn();
         newColDefs.push_back(newColDef);
      } else {
         newColDefs.push_back(colDef);
      }
   }
   setColumnsAttr(mlir::ArrayAttr::get(getContext(), newColDefs));
   return mlir::success();
}
mlir::LogicalResult relalg::CrossProductOp::foldColumns(relalg::ColumnFoldInfo& columnInfo) {
   return mlir::success();
}
mlir::LogicalResult relalg::InnerJoinOp::foldColumns(relalg::ColumnFoldInfo& columnInfo) {
   replaceColumnUsesInLamda(getContext(), getPredicate().front(), columnInfo);
   return mlir::success();
}
mlir::LogicalResult relalg::SemiJoinOp::foldColumns(relalg::ColumnFoldInfo& columnInfo) {
   replaceColumnUsesInLamda(getContext(), getPredicate().front(), columnInfo);
   return mlir::success();
}
mlir::LogicalResult relalg::AntiSemiJoinOp::foldColumns(relalg::ColumnFoldInfo& columnInfo) {
   replaceColumnUsesInLamda(getContext(), getPredicate().front(), columnInfo);
   return mlir::success();
}
ColumnSet relalg::NestedOp::getCreatedColumns() {
   AvailabilityCache cache;
   return getAvailableColumns(cache); //todo: fix
}

ColumnSet relalg::NestedOp::getUsedColumns() {
   return relalg::ColumnSet::fromArrayAttr(getUsedCols());
}
ColumnSet relalg::NestedOp::getAvailableColumns(AvailabilityCache&) {
   return relalg::ColumnSet::fromArrayAttr(getAvailableCols());
}
bool relalg::NestedOp::canColumnReach(Operator source, Operator target, const tuples::Column* col) {
   AvailabilityCache cache;
   ColumnSet available = getAvailableColumns(cache);
   if (available.contains(col)) {
      return relalg::detail::canColumnReach(this->getOperation(), source, target, col);
   }
   return false;
}

ColumnSet AvailabilityCache::getAvailableColumnsFor(Operator op) {
   auto* operation = op.getOperation();
   auto it = cache.find(operation);
   if (it != cache.end()) {
      return it->second;
   }
   auto result = op.getAvailableColumns(*this);
   cache[operation] = result;
   return result;
}

#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOpsInterfaces.cpp.inc"
