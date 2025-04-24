#pragma once
#include "lingodb/compiler/frontend/ast/group_by_node.h"
#include "lingodb/compiler/frontend/ast/parsed_expression.h"
#include "lingodb/compiler/frontend/ast/pipe_operator.h"
#include "lingodb/compiler/frontend/ast/query_node.h"
#include "lingodb/compiler/frontend/ast/tableref.h"

#include "lingodb/compiler/frontend/column_semantic.h"
#include <memory>
#include <vector>
namespace lingodb::ast {
class SelectNode : public QueryNode {
   public:
   SelectNode();
   ~SelectNode() override;
   static constexpr QueryNodeType TYPE = QueryNodeType::SELECT_NODE;
   //! The projection list
   std::shared_ptr<TargetsExpression> select_list;
   //! The FROM clause
   std::shared_ptr<TableRef> from_clause;
   //! The WHERE clause
   std::shared_ptr<ParsedExpression> where_clause;

   //! list of groups
   std::shared_ptr<GroupByNode> groups;

   //! HAVING clause
   std::shared_ptr<ParsedExpression> having;

   ///For pipe operators
   std::shared_ptr<PipeOperator> startPipeOperator;
   std::shared_ptr<PipeOperator> endPipeOperator;

   /*
    * Semantic
    */
   TargetInfo targetInfo{};

   //TODO add missing parameters
   std::string toString(uint32_t depth) override;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};
} // namespace lingodb::ast
