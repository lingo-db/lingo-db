#pragma once
#include "parsed_expression.h"
#include <memory>
#include <set>
#include <string>
#include <vector>
namespace lingodb::ast {
using GroupingSet = std::set<int>;
class GroupByNode : public AstNode {
   public:
   GroupByNode() : AstNode(NodeType::GROUP_BY) {};
   //! The total set of all group expressions
   std::vector<std::shared_ptr<ParsedExpression>> group_expressions;

   //! The different grouping sets as they map to the group expressions
   std::vector<GroupingSet> grouping_sets;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};
} // namespace lingodb::ast