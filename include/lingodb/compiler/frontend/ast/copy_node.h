#ifndef LINGODB_COMPILER_FRONTEND_AST_COPY_NODE_H
#define LINGODB_COMPILER_FRONTEND_AST_COPY_NODE_H

#include "ast_node.h"
#include "parsed_expression.h"

#include <memory>

namespace lingodb::ast {
class CopyInfo {
   public:
   std::string fromFileName;
   std::string table;
   std::vector<std::pair<std::string, std::string>> options;
};
/**
 * Node for the Copy statements
 */
class CopyNode : public AstNode {
   static constexpr NodeType cType = NodeType::COPY_NODE;

   public:
   CopyNode() : AstNode(cType), copyInfo(std::make_shared<CopyInfo>()) {}
   std::shared_ptr<CopyInfo> copyInfo;
};
} // namespace lingodb::ast
#endif
