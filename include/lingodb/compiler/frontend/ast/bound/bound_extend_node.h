#ifndef LINGODB_COMPILER_FRONTEND_AST_BOUND_BOUND_EXTEND_NODE_H
#define LINGODB_COMPILER_FRONTEND_AST_BOUND_BOUND_EXTEND_NODE_H

#include "bound_expression.h"
#include "lingodb/compiler/frontend/ast/table_producer.h"
namespace lingodb::ast {

class BoundExtendNode : public AstNode {
   public:
   static constexpr auto cType = NodeType::BOUND_EXTEND_NODE;
   BoundExtendNode(std::string mapName, std::vector<std::shared_ptr<BoundExpression>> extensions) : AstNode(cType), mapName(mapName), extensions(std::move(extensions)) {}

   std::string mapName;
   std::vector<std::shared_ptr<BoundExpression>> extensions;
   std::vector<std::shared_ptr<BoundWindowExpression>> windowExpressions;
};
} // namespace lingodb::ast
#endif
