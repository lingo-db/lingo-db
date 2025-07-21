#pragma once
#include "lingodb/compiler/frontend/sql_context.h"
#include "query_node.h"

namespace lingodb::ast {
class CTENode : public QueryNode {
   public:
   static constexpr const QueryNodeType TYPE = QueryNodeType::CTE_NODE;

   CTENode() : QueryNode(QueryNodeType::CTE_NODE) {}

   //TODO add missing parameters
   std::string toString(uint32_t depth) override;

   std::shared_ptr<TableProducer> query;

   //Maybe use input logic instead
   std::shared_ptr<TableProducer> child;

   std::vector<std::string> columnAliases;

   //!The scope for the query. Must be not a pointer, so a copy is required everytime the cte query is translated
   analyzer::SQLScope subQueryScope;

   std::vector<std::pair<std::shared_ptr<ast::NamedResult>, std::shared_ptr<ast::NamedResult>>> renamedResults;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};
}