#ifndef LINGODB_COMPILER_FRONTEND_AST_GROUP_BY_NODE_H
#define LINGODB_COMPILER_FRONTEND_AST_GROUP_BY_NODE_H


#include "parsed_expression.h"
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>
namespace lingodb::ast {

class GroupByNode : public AstNode {
   public:
   GroupByNode() : AstNode(NodeType::GROUP_BY) {};
   //! The total set of all group expressions
   //TODO
   std::vector<std::shared_ptr<ParsedExpression>> groupByExpressions;

   std::vector<std::set<size_t>> groupingSet;

      std::unordered_set<std::shared_ptr<FunctionExpression>, ParsedExprPtrHash, ParsedExprPtrEqual> groupingFunctions;

   bool rollup = false;
   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};
} // namespace lingodb::ast
#endif
