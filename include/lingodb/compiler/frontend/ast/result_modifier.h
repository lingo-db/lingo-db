#pragma once
#include "lingodb/compiler/frontend/ast/parsed_expression.h"
#include "table_producer.h"
#include <memory>
#include <vector>
namespace lingodb::ast {
enum class OrderType : uint8_t {
   ASCENDING = 1,
   DESCENDING = 2
};
enum class ResultModifierType : uint8_t {
   ORDER_BY = 1,
   LIMIT = 2,
   OFFSET = 3,

   BOUND_ORDER_BY = 11,
   BOUND_LIMIT = 12,
};
enum class OrderByNullType : uint8_t {
   INVALID = 0,
   ORDER_DEFAULT = 1,
   NULLS_FIRST = 2,
   NULLS_LAST = 3
};

class ResultModifier : public TableProducer {
   public:
   explicit ResultModifier(ResultModifierType type)
      : TableProducer(NodeType::RESULT_MODIFIER), modifierType(type) {}

   virtual ~ResultModifier() = default;

   ResultModifierType modifierType;
   std::shared_ptr<TableProducer> input = nullptr;

   virtual std::string toDotGraph(uint32_t depth);
};
/**
 * GROUP BY <element>
 */
class OrderByElement {
   public:
   OrderByElement(OrderType type, OrderByNullType nullOrder) : type(type), nullOrder(nullOrder) {};

   /// Sort order
   OrderType type;
   /// Expression to order by
   std::shared_ptr<ParsedExpression> expression;
   /// The NULL sort order, NULLS_FIRST or NULLS_LAST
   OrderByNullType nullOrder;

   location loc;
};
/**
 * Complete GROUP BY node with its orderby elements
 */
class OrderByModifier : public ResultModifier {
   public:
   OrderByModifier() : ResultModifier(ResultModifierType::ORDER_BY) {}

   std::vector<std::shared_ptr<OrderByElement>> orderByElements;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};

class LimitModifier : public ResultModifier {
   public:
   explicit LimitModifier(std::shared_ptr<ParsedExpression> limitExpression) : ResultModifier(ResultModifierType::LIMIT), limitExpression(std::move(limitExpression)) {}
   std::shared_ptr<ParsedExpression> limitExpression;
   std::shared_ptr<ParsedExpression> offset;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
   // ... implementation ...
};

class OffsetModifier : public ResultModifier {
   public:
   OffsetModifier() : ResultModifier(ResultModifierType::OFFSET) {}
   std::shared_ptr<ParsedExpression> offsetExpression;
};

} // namespace lingodb::ast