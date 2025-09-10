#pragma once
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
   static constexpr NodeType TYPE = NodeType::COPY_NODE;
   public:
   CopyNode();
   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
   std::shared_ptr<CopyInfo> copyInfo;
};
} // namespace lingodb::ast