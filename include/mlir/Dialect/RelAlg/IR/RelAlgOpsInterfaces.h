#ifndef MLIR_DIALECT_RELALG_IR_RELALGOPSINTERFACES_H
#define MLIR_DIALECT_RELALG_IR_RELALGOPSINTERFACES_H

#include "llvm/ADT/SmallPtrSet.h"

#include "mlir/Dialect/RelAlg/ColumnSet.h"
#include "mlir/Dialect/RelAlg/FunctionalDependencies.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::relalg::detail {
void replaceUsages(mlir::Operation* op,std::function<mlir::relalg::ColumnRefAttr(mlir::relalg::ColumnRefAttr)> fn);
ColumnSet getUsedColumns(mlir::Operation* op);
ColumnSet getAvailableColumns(mlir::Operation* op);
ColumnSet getFreeColumns(mlir::Operation* op);
ColumnSet getSetOpCreatedColumns(mlir::Operation* op);
ColumnSet getSetOpUsedColumns(mlir::Operation* op);
FunctionalDependencies getFDs(mlir::Operation* op);
bool isDependentJoin(mlir::Operation* op);
void moveSubTreeBefore(mlir::Operation* tree, mlir::Operation* before);

enum class BinaryOperatorType : unsigned char {
   None = 0,
   Union,
   Intersection,
   Except,
   CP,
   InnerJoin,
   SemiJoin,
   AntiSemiJoin,
   OuterJoin,
   FullOuterJoin,
   MarkJoin,
   CollectionJoin,
   LAST
};
enum UnaryOperatorType : unsigned char {
   None = 0,
   DistinctProjection,
   Projection,
   Map,
   Selection,
   Aggregation,
   LAST
};

template <class A, class B>
class CompatibilityTable {
   static constexpr size_t sizeA = static_cast<size_t>(A::LAST);
   static constexpr size_t sizeB = static_cast<size_t>(B::LAST);

   bool table[sizeA][sizeB];

   public:
   constexpr CompatibilityTable(std::initializer_list<std::pair<A, B>> l) : table() {
      for (auto item : l) {
         auto [a, b] = item;
         table[static_cast<size_t>(a)][static_cast<size_t>(b)] = true;
      }
   }
   constexpr bool contains(const A a, const B b) const {
      return table[static_cast<size_t>(a)][static_cast<size_t>(b)];
   }
};
constexpr CompatibilityTable<BinaryOperatorType, BinaryOperatorType> assoc{
   {BinaryOperatorType::Union, BinaryOperatorType::Union},
   {BinaryOperatorType::Intersection, BinaryOperatorType::Intersection},
   {BinaryOperatorType::Intersection, BinaryOperatorType::SemiJoin},
   {BinaryOperatorType::Intersection, BinaryOperatorType::AntiSemiJoin},
   {BinaryOperatorType::CP, BinaryOperatorType::CP},
   {BinaryOperatorType::CP, BinaryOperatorType::InnerJoin},
   {BinaryOperatorType::CP, BinaryOperatorType::SemiJoin},
   {BinaryOperatorType::CP, BinaryOperatorType::AntiSemiJoin},
   {BinaryOperatorType::CP, BinaryOperatorType::OuterJoin},
   {BinaryOperatorType::CP, BinaryOperatorType::MarkJoin},
   {BinaryOperatorType::CP, BinaryOperatorType::CollectionJoin},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::CP},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::InnerJoin},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::SemiJoin},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::AntiSemiJoin},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::OuterJoin},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::MarkJoin},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::CollectionJoin},

};
constexpr CompatibilityTable<BinaryOperatorType, BinaryOperatorType> lAsscom{
   {BinaryOperatorType::Union, BinaryOperatorType::Union},
   {BinaryOperatorType::Intersection, BinaryOperatorType::Intersection},
   {BinaryOperatorType::Intersection, BinaryOperatorType::SemiJoin},
   {BinaryOperatorType::Intersection, BinaryOperatorType::AntiSemiJoin},
   {BinaryOperatorType::Except, BinaryOperatorType::SemiJoin},
   {BinaryOperatorType::Except, BinaryOperatorType::AntiSemiJoin},
   {BinaryOperatorType::CP, BinaryOperatorType::CP},
   {BinaryOperatorType::CP, BinaryOperatorType::InnerJoin},
   {BinaryOperatorType::CP, BinaryOperatorType::SemiJoin},
   {BinaryOperatorType::CP, BinaryOperatorType::AntiSemiJoin},
   {BinaryOperatorType::CP, BinaryOperatorType::OuterJoin},
   {BinaryOperatorType::CP, BinaryOperatorType::MarkJoin},
   {BinaryOperatorType::CP, BinaryOperatorType::CollectionJoin},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::CP},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::InnerJoin},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::SemiJoin},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::AntiSemiJoin},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::OuterJoin},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::MarkJoin},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::CollectionJoin},
   {BinaryOperatorType::SemiJoin, BinaryOperatorType::Intersection},
   {BinaryOperatorType::SemiJoin, BinaryOperatorType::Except},
   {BinaryOperatorType::SemiJoin, BinaryOperatorType::CP},
   {BinaryOperatorType::SemiJoin, BinaryOperatorType::InnerJoin},
   {BinaryOperatorType::SemiJoin, BinaryOperatorType::SemiJoin},
   {BinaryOperatorType::SemiJoin, BinaryOperatorType::AntiSemiJoin},
   {BinaryOperatorType::SemiJoin, BinaryOperatorType::OuterJoin},
   {BinaryOperatorType::SemiJoin, BinaryOperatorType::MarkJoin},
   {BinaryOperatorType::SemiJoin, BinaryOperatorType::CollectionJoin},
   {BinaryOperatorType::AntiSemiJoin, BinaryOperatorType::Intersection},
   {BinaryOperatorType::AntiSemiJoin, BinaryOperatorType::Except},
   {BinaryOperatorType::AntiSemiJoin, BinaryOperatorType::CP},
   {BinaryOperatorType::AntiSemiJoin, BinaryOperatorType::InnerJoin},
   {BinaryOperatorType::AntiSemiJoin, BinaryOperatorType::SemiJoin},
   {BinaryOperatorType::AntiSemiJoin, BinaryOperatorType::AntiSemiJoin},
   {BinaryOperatorType::AntiSemiJoin, BinaryOperatorType::OuterJoin},
   {BinaryOperatorType::AntiSemiJoin, BinaryOperatorType::MarkJoin},
   {BinaryOperatorType::AntiSemiJoin, BinaryOperatorType::CollectionJoin},
   {BinaryOperatorType::OuterJoin, BinaryOperatorType::CP},
   {BinaryOperatorType::OuterJoin, BinaryOperatorType::InnerJoin},
   {BinaryOperatorType::OuterJoin, BinaryOperatorType::SemiJoin},
   {BinaryOperatorType::OuterJoin, BinaryOperatorType::AntiSemiJoin},
   {BinaryOperatorType::OuterJoin, BinaryOperatorType::OuterJoin},
   {BinaryOperatorType::OuterJoin, BinaryOperatorType::MarkJoin},
   {BinaryOperatorType::OuterJoin, BinaryOperatorType::CollectionJoin},
   {BinaryOperatorType::MarkJoin, BinaryOperatorType::CP},
   {BinaryOperatorType::MarkJoin, BinaryOperatorType::InnerJoin},
   {BinaryOperatorType::MarkJoin, BinaryOperatorType::SemiJoin},
   {BinaryOperatorType::MarkJoin, BinaryOperatorType::AntiSemiJoin},
   {BinaryOperatorType::MarkJoin, BinaryOperatorType::OuterJoin},
   {BinaryOperatorType::MarkJoin, BinaryOperatorType::CollectionJoin},
   {BinaryOperatorType::MarkJoin, BinaryOperatorType::MarkJoin},

   {BinaryOperatorType::CollectionJoin, BinaryOperatorType::CP},
   {BinaryOperatorType::CollectionJoin, BinaryOperatorType::InnerJoin},
   {BinaryOperatorType::CollectionJoin, BinaryOperatorType::SemiJoin},
   {BinaryOperatorType::CollectionJoin, BinaryOperatorType::AntiSemiJoin},
   {BinaryOperatorType::CollectionJoin, BinaryOperatorType::OuterJoin},
   {BinaryOperatorType::CollectionJoin, BinaryOperatorType::MarkJoin},
   {BinaryOperatorType::CollectionJoin, BinaryOperatorType::CollectionJoin},

};
constexpr CompatibilityTable<BinaryOperatorType, BinaryOperatorType> rAsscom{
   {BinaryOperatorType::Union, BinaryOperatorType::Union},
   {BinaryOperatorType::Intersection, BinaryOperatorType::Intersection},
   {BinaryOperatorType::CP, BinaryOperatorType::CP},
   {BinaryOperatorType::CP, BinaryOperatorType::InnerJoin},
};
constexpr CompatibilityTable<UnaryOperatorType, BinaryOperatorType> lPushable{
   {UnaryOperatorType::DistinctProjection, BinaryOperatorType::Intersection},
   {UnaryOperatorType::DistinctProjection, BinaryOperatorType::Except},
   {UnaryOperatorType::DistinctProjection, BinaryOperatorType::SemiJoin},
   {UnaryOperatorType::DistinctProjection, BinaryOperatorType::AntiSemiJoin},
   {UnaryOperatorType::Projection, BinaryOperatorType::SemiJoin},
   {UnaryOperatorType::Projection, BinaryOperatorType::AntiSemiJoin},
   {UnaryOperatorType::Selection, BinaryOperatorType::Intersection},
   {UnaryOperatorType::Selection, BinaryOperatorType::Except},
   {UnaryOperatorType::Selection, BinaryOperatorType::CP},
   {UnaryOperatorType::Selection, BinaryOperatorType::InnerJoin},
   {UnaryOperatorType::Selection, BinaryOperatorType::SemiJoin},
   {UnaryOperatorType::Selection, BinaryOperatorType::AntiSemiJoin},
   {UnaryOperatorType::Selection, BinaryOperatorType::OuterJoin},
   {UnaryOperatorType::Map, BinaryOperatorType::CP},
   {UnaryOperatorType::Map, BinaryOperatorType::InnerJoin},
   {UnaryOperatorType::Map, BinaryOperatorType::SemiJoin},
   {UnaryOperatorType::Map, BinaryOperatorType::AntiSemiJoin},
   {UnaryOperatorType::Map, BinaryOperatorType::OuterJoin},
   {UnaryOperatorType::Aggregation, BinaryOperatorType::SemiJoin},
   {UnaryOperatorType::Aggregation, BinaryOperatorType::AntiSemiJoin},
};
constexpr CompatibilityTable<UnaryOperatorType, BinaryOperatorType> rPushable{
   {UnaryOperatorType::DistinctProjection, BinaryOperatorType::Intersection},
   {UnaryOperatorType::Map, BinaryOperatorType::CP},
   {UnaryOperatorType::Map, BinaryOperatorType::InnerJoin},
   {UnaryOperatorType::Selection, BinaryOperatorType::Intersection},
   {UnaryOperatorType::Selection, BinaryOperatorType::CP},
   {UnaryOperatorType::Selection, BinaryOperatorType::InnerJoin},
};
constexpr CompatibilityTable<UnaryOperatorType, UnaryOperatorType> reorderable{
   {UnaryOperatorType::DistinctProjection, UnaryOperatorType::DistinctProjection},
   {UnaryOperatorType::DistinctProjection, UnaryOperatorType::Selection},
   {UnaryOperatorType::DistinctProjection, UnaryOperatorType::Map},
   {UnaryOperatorType::Projection, UnaryOperatorType::Selection},
   {UnaryOperatorType::Projection, UnaryOperatorType::Map},
   {UnaryOperatorType::Selection, UnaryOperatorType::DistinctProjection},
   {UnaryOperatorType::Selection, UnaryOperatorType::Projection},
   {UnaryOperatorType::Selection, UnaryOperatorType::Selection},
   {UnaryOperatorType::Selection, UnaryOperatorType::Map},
   {UnaryOperatorType::Map, UnaryOperatorType::DistinctProjection},
   {UnaryOperatorType::Map, UnaryOperatorType::Projection},
   {UnaryOperatorType::Map, UnaryOperatorType::Selection},
   {UnaryOperatorType::Map, UnaryOperatorType::Map},
};

BinaryOperatorType getBinaryOperatorType(Operation* op);
UnaryOperatorType getUnaryOperatorType(Operation* op);

bool isJoin(Operation* op);

void addPredicate(mlir::Operation* op, std::function<mlir::Value(mlir::Value, mlir::OpBuilder& builder)> predicateProducer);
void initPredicate(mlir::Operation* op);

void inlineOpIntoBlock(mlir::Operation* vop, mlir::Operation* includeChildren, mlir::Operation* excludeChildren, mlir::Block* newBlock, mlir::BlockAndValueMapping& mapping, mlir::Operation* first = nullptr);
} // namespace mlir::relalg::detail
class Operator;
#define GET_OP_CLASSES
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h.inc"

#endif // MLIR_DIALECT_RELALG_IR_RELALGOPSINTERFACES_H
