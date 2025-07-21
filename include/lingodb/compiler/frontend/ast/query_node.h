#pragma once
#include "ast_node.h"
#include "result_modifier.h"
#include "table_producer.h"

#include <cstdint>
#include <memory>
#include <string>
#define toAsciiASTPrefix                      \
   std::string ast{};                         \
   for (uint32_t i = 0; i < depth - 1; ++i) { \
      ast.append("\t");                       \
                                              \
      ast.append("");                         \
   }                                          \
   ast.append("\t");                          \
   ast.append("└──");
namespace lingodb::ast {
enum class QueryNodeType : uint8_t {
   SELECT_NODE,
   SET_OPERATION_NODE = 2,
   BOUND_SET_OPERATION_NODE = 4,
   //BOUND_SUBQUERY_NODE = 5,
   RECURSIVE_CTE_NODE = 6,
   CTE_NODE = 7,
   PIPE_NODE = 8

};
class QueryNode : public TableProducer {
   public:
   virtual ~QueryNode() override = default;

   explicit QueryNode(QueryNodeType type) : TableProducer(NodeType::QUERY_NODE), type(type) {};

   //! The type of the query node, either SetOperation or Select
   QueryNodeType type;

   /// The set of result modifiers associated with this query node
   std::vector<std::shared_ptr<ResultModifier>> modifiers{};

   std::shared_ptr<TableProducer> input;

   virtual std::string toString(uint32_t depth) = 0;

   virtual std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) = 0;
};
enum class SetOperationType {
   NONE = 0,
   UNION = 1,
   EXCEPT = 2,
   INTERSECT = 3,
   UNION_BY_NAME = 4
};
class SetOperationNode : public QueryNode {
   static constexpr QueryNodeType TYPE = QueryNodeType::SET_OPERATION_NODE;

   public:
   SetOperationNode(SetOperationType setType, std::shared_ptr<TableProducer> left, std::shared_ptr<TableProducer> right);
   SetOperationType setType;
   bool setOpAll = false;
   std::shared_ptr<TableProducer> left;
   std::shared_ptr<TableProducer> right;

   std::string toString(uint32_t depth) override;
   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};

} // namespace lingodb::ast
