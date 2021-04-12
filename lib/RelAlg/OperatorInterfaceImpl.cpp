#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/IR/OpImplementation.h"
#include <llvm/ADT/TypeSwitch.h>
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
      .Case<mlir::relalg::CrossProductOp>([&](mlir::Operation* op) { return CP; })
      .Case<mlir::relalg::InnerJoinOp>([&](mlir::Operation* op) { return InnerJoin; })
      .Case<mlir::relalg::SemiJoinOp>([&](mlir::Operation* op) { return SemiJoin; })
      .Case<mlir::relalg::AntiSemiJoinOp>([&](mlir::Operation* op) { return AntiSemiJoin; })
      .Case<mlir::relalg::SingleJoinOp>([&](mlir::Operation* op) { return OuterJoin; })
      .Case<mlir::relalg::MarkJoinOp>([&](mlir::Operation* op) { return MarkJoin; })
      .Case<mlir::relalg::OuterJoinOp>([&](mlir::relalg::OuterJoinOp op) { return OuterJoin; })
      .Case<mlir::relalg::FullOuterJoinOp>([&](mlir::relalg::FullOuterJoinOp op) { return FullOuterJoin; })
      .Default([&](auto x) {
         return None;
      });
}

Attributes AggregationOp::getUsedAttributes() {
   auto used = mlir::relalg::detail::getUsedAttributes(getOperation());
   for (Attribute a : group_by_attrs()) {
      used.insert(&a.dyn_cast_or_null<RelationalAttributeRefAttr>().getRelationalAttribute());
   }
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
   Attributes creations;
   for (Attribute a : attributes()) {
      creations.insert(&a.dyn_cast_or_null<RelationalAttributeDefAttr>().getRelationalAttribute());
   }
   return creations;
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
      for (Attribute existing : fromExisting) {
         auto relationRefAttr = existing.dyn_cast_or_null<RelationalAttributeRefAttr>();
         used.insert(&relationRefAttr.getRelationalAttribute());
      }
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
   for (Attribute a : group_by_attrs()) {
      available.insert(&a.dyn_cast_or_null<RelationalAttributeRefAttr>().getRelationalAttribute());
   }
   return available;
}
Attributes mlir::relalg::ProjectionOp::getAvailableAttributes() {
   Attributes available;
   for (Attribute a : attrs()) {
      available.insert(&a.dyn_cast_or_null<RelationalAttributeRefAttr>().getRelationalAttribute());
   }
   return available;
}

// @formatter:off
// clang-format off
const bool mlir::relalg::detail::assoc[mlir::relalg::detail::BinaryOperatorType::LAST][mlir::relalg::detail::BinaryOperatorType::LAST] = {
   /* None =  */{},
   /* CP           =  */{/*None=*/false,/*CP=*/true, /*InnerJoin=*/true, /*SemiJoin=*/true, /*AntiSemiJoin=*/true, /*OuterJoin=*/true, /*FullOuterJoin=*/false,/*MarkJoin=*/true,},
   /* InnerJoin    =  */{/*None=*/false,/*CP=*/true, /*InnerJoin=*/true, /*SemiJoin=*/true, /*AntiSemiJoin=*/true, /*OuterJoin=*/true, /*FullOuterJoin=*/false,/*MarkJoin=*/true,},
   /* SemiJoin     =  */{/*None=*/false,/*CP=*/false,/*InnerJoin=*/false,/*SemiJoin=*/false,/*AntiSemiJoin=*/false,/*OuterJoin=*/false,/*FullOuterJoin=*/false,/*MarkJoin=*/false},
   /* AntiSemiJoin =  */{/*None=*/false,/*CP=*/false,/*InnerJoin=*/false,/*SemiJoin=*/false,/*AntiSemiJoin=*/false,/*OuterJoin=*/false,/*FullOuterJoin=*/false,/*MarkJoin=*/false},
   /* OuterJoin    =  */{/*None=*/false,/*CP=*/false,/*InnerJoin=*/false,/*SemiJoin=*/false,/*AntiSemiJoin=*/false,/*OuterJoin=*/false,/*FullOuterJoin=*/false,/*MarkJoin=*/false},
   /* FullOuterJoin=  */{/*None=*/false,/*CP=*/false,/*InnerJoin=*/false,/*SemiJoin=*/false,/*AntiSemiJoin=*/false,/*OuterJoin=*/false,/*FullOuterJoin=*/false,/*MarkJoin=*/false},
   /* MarkJoin     =  */{/*None=*/false,/*CP=*/false,/*InnerJoin=*/false,/*SemiJoin=*/false,/*AntiSemiJoin=*/false,/*OuterJoin=*/false,/*FullOuterJoin=*/false,/*MarkJoin=*/false},

};
const bool mlir::relalg::detail::lAsscom[mlir::relalg::detail::BinaryOperatorType::LAST][mlir::relalg::detail::BinaryOperatorType::LAST] = {
   /* None =  */{},
   /* CP           =  */{/*None=*/false,/*CP=*/true, /*InnerJoin=*/true, /*SemiJoin=*/true, /*AntiSemiJoin=*/true, /*OuterJoin=*/true, /*FullOuterJoin=*/false,/*MarkJoin=*/true},
   /* InnerJoin    =  */{/*None=*/false,/*CP=*/true, /*InnerJoin=*/true, /*SemiJoin=*/true, /*AntiSemiJoin=*/true, /*OuterJoin=*/true, /*FullOuterJoin=*/false,/*MarkJoin=*/true},
   /* SemiJoin     =  */{/*None=*/false,/*CP=*/true, /*InnerJoin=*/true, /*SemiJoin=*/true, /*AntiSemiJoin=*/true, /*OuterJoin=*/true, /*FullOuterJoin=*/false,/*MarkJoin=*/true},
   /* AntiSemiJoin =  */{/*None=*/false,/*CP=*/true, /*InnerJoin=*/true, /*SemiJoin=*/true, /*AntiSemiJoin=*/true, /*OuterJoin=*/true, /*FullOuterJoin=*/false,/*MarkJoin=*/true},
   /* OuterJoin    =  */{/*None=*/false,/*CP=*/true, /*InnerJoin=*/true, /*SemiJoin=*/true, /*AntiSemiJoin=*/true, /*OuterJoin=*/true, /*FullOuterJoin=*/false,/*MarkJoin=*/true},
   /* FullOuterJoin=  */{/*None=*/false,/*CP=*/false,/*InnerJoin=*/false,/*SemiJoin=*/false,/*AntiSemiJoin=*/false,/*OuterJoin=*/false,/*FullOuterJoin=*/false,/*MarkJoin=*/false},
   /* MarkJoin     =  */{/*None=*/false,/*CP=*/true, /*InnerJoin=*/true, /*SemiJoin=*/true, /*AntiSemiJoin=*/true, /*OuterJoin=*/true, /*FullOuterJoin=*/false,/*MarkJoin=*/true},
};
const bool mlir::relalg::detail::rAsscom[mlir::relalg::detail::BinaryOperatorType::LAST][mlir::relalg::detail::BinaryOperatorType::LAST] = {
   /* None =  */{},
   /* CP           =  */{/*None=*/false,/*CP=*/true, /*InnerJoin=*/true, /*SemiJoin=*/false,/*AntiSemiJoin=*/false,/*OuterJoin=*/false,/*FullOuterJoin=*/false,/*MarkJoin=*/false},
   /* InnerJoin    =  */{/*None=*/false,/*CP=*/true, /*InnerJoin=*/true, /*SemiJoin=*/false,/*AntiSemiJoin=*/false,/*OuterJoin=*/false,/*FullOuterJoin=*/false,/*MarkJoin=*/false},
   /* SemiJoin     =  */{/*None=*/false,/*CP=*/false,/*InnerJoin=*/false,/*SemiJoin=*/false,/*AntiSemiJoin=*/false,/*OuterJoin=*/false,/*FullOuterJoin=*/false,/*MarkJoin=*/false},
   /* AntiSemiJoin =  */{/*None=*/false,/*CP=*/false,/*InnerJoin=*/false,/*SemiJoin=*/false,/*AntiSemiJoin=*/false,/*OuterJoin=*/false,/*FullOuterJoin=*/false,/*MarkJoin=*/false},
   /* OuterJoin    =  */{/*None=*/false,/*CP=*/false,/*InnerJoin=*/false,/*SemiJoin=*/false,/*AntiSemiJoin=*/false,/*OuterJoin=*/false,/*FullOuterJoin=*/false,/*MarkJoin=*/false},
   /* FullOuterJoin=  */{/*None=*/false,/*CP=*/false,/*InnerJoin=*/false,/*SemiJoin=*/false,/*AntiSemiJoin=*/false,/*OuterJoin=*/false,/*FullOuterJoin=*/false,/*MarkJoin=*/false},
   /* MarkJoin     =  */{/*None=*/false,/*CP=*/false,/*InnerJoin=*/false,/*SemiJoin=*/false,/*AntiSemiJoin=*/false,/*OuterJoin=*/false,/*FullOuterJoin=*/false,/*MarkJoin=*/false},
};
// @formatter:on
// clang-format on
bool mlir::relalg::detail::binaryOperatorIs(const bool (&table)[BinaryOperatorType::LAST][BinaryOperatorType::LAST], Operation* a, Operation* b) {
   return table[getBinaryOperatorType(a)][getBinaryOperatorType(b)];
}
bool mlir::relalg::detail::isJoin(Operation* op) {
   auto opType = getBinaryOperatorType(op);
   return InnerJoin <= opType && opType <= MarkJoin;
}

#include "mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.cpp.inc"
