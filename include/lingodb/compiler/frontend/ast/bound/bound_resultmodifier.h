#pragma once
#include "bound_expression.h"
#include "lingodb/compiler/frontend/ast/result_modifier.h"
#include "lingodb/compiler/frontend/ast/table_producer.h"

#include <cstdint>
#include <string>
namespace lingodb::ast {
class BoundResultModifier : public TableProducer {
   public:
   explicit BoundResultModifier(ResultModifierType type)
      : TableProducer(NodeType::BOUND_RESULT_MODIFIER), modifierType(type) {}
   BoundResultModifier(ResultModifierType type, std::shared_ptr<TableProducer> input)
      : TableProducer(NodeType::BOUND_RESULT_MODIFIER), modifierType(type), input(input) {}

   virtual ~BoundResultModifier() = default;

   ResultModifierType modifierType;
   std::shared_ptr<TableProducer> input = nullptr;
};
/**
 * Follows a similar logic to that of DuckDB
 * GROUP BY <element>
 */
class BoundOrderByElement {
   public:
   BoundOrderByElement(OrderType type, OrderByNullType nullOrder, std::shared_ptr<ColumnReference> columnReference) : type(type), nullOrder(nullOrder), columnReference(columnReference) {};

   /// Sort order
   OrderType type;
   /// Expression to order by
   std::shared_ptr<ColumnReference> columnReference;
   /// The NULL sort order, NULLS_FIRST or NULLS_LAST
   OrderByNullType nullOrder;

   location loc;
};
/**
 * Complete GROUP BY node
 */
class BoundOrderByModifier : public BoundResultModifier {
   public:
   BoundOrderByModifier(std::vector<std::shared_ptr<BoundOrderByElement>> orderByElements, std::shared_ptr<TableProducer> input) : BoundResultModifier(ResultModifierType::BOUND_ORDER_BY, input), orderByElements(orderByElements) {}

   std::vector<std::shared_ptr<BoundOrderByElement>> orderByElements;
};

class BoundLimitModifier : public BoundResultModifier {
   public:
   BoundLimitModifier(std::shared_ptr<BoundExpression> limitExpression, std::shared_ptr<BoundExpression> offset, std::shared_ptr<TableProducer> input) : BoundResultModifier(ResultModifierType::BOUND_LIMIT, input), limitExpression(limitExpression), offset(offset){}

   std::shared_ptr<BoundExpression> limitExpression;
   std::shared_ptr<BoundExpression> offset;
};
}