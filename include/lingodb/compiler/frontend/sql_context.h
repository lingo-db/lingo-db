#pragma once

#include "lingodb/compiler/frontend/ast/aggregation_node.h"
#include "lingodb/compiler/frontend/ast/extend_node.h"
#include "lingodb/compiler/frontend/sql_scope.h"
#include "lingodb/compiler/frontend/ast/bound/bound_query_node.h"

#include <llvm/ADT/ScopedHashTable.h>

#include <memory>
#include <stack>
#include <unordered_set>
#include <vector>
namespace lingodb::ast {
class CTENode;
}
namespace lingodb::analyzer {

struct StringInfo {
   static bool isEqual(std::string a, std::string b) { return a == b; }
   static std::string getEmptyKey() { return ""; }
   static std::string getTombstoneKey() { return "-"; }
   static size_t getHashValue(std::string str) { return std::hash<std::string>{}(str); }
};
class ASTTransformScope {
   public:
   ASTTransformScope() : aggregationNode(std::make_shared<ast::AggregationNode>()), extendNodeBeforeAggregation(std::make_shared<ast::ExtendNode>(true)), extendNodeBeforeWindowFunctions(std::make_shared<ast::ExtendNode>(true)), extendNodeAfterAggregations(std::make_shared<ast::ExtendNode>(true)), extendNodeBeforeOrderBy(std::make_shared<ast::ExtendNode>(true)), aggregationNodeForWindowFunctions(std::make_shared<ast::AggregationNode>()) {}
   std::shared_ptr<ast::AggregationNode> aggregationNode;
   std::shared_ptr<ast::AggregationNode> aggregationNodeForWindowFunctions;
   std::shared_ptr<ast::ExtendNode> extendNodeBeforeAggregation;
   std::shared_ptr<ast::ExtendNode> extendNodeBeforeWindowFunctions;
   std::shared_ptr<ast::ExtendNode> extendNodeAfterAggregations;
   std::shared_ptr<ast::ExtendNode> extendNodeBeforeOrderBy;
   std::unordered_map<std::shared_ptr<ast::ParsedExpression>, std::shared_ptr<ast::ColumnRefExpression>, ast::ParsedExprPtrHash, ast::ParsedExprPtrEqual> alreadyExtendedExpression;
   /// Container for GROUP BY expressions that ensures uniqueness, reuse in the canonicalization step
   std::unordered_set<std::shared_ptr<ast::ParsedExpression>, ast::ParsedExprPtrHash, ast::ParsedExprPtrEqual> groupedByExpressions;
};
class ASTTransformContext {
   public:
   ASTTransformContext();
   std::stack<std::shared_ptr<ASTTransformScope>> scopeStack{};
   std::shared_ptr<ASTTransformScope> currentScope;
   void pushNewScope() {
      currentScope = std::make_shared<ASTTransformScope>();
      scopeStack.push(currentScope);
   }

   void popScope() {
      currentScope = scopeStack.top();
      scopeStack.pop();
   }
};
class SQLContext;
class DefineScope {
   public:
   DefineScope(SQLContext& sqlContext);
   ~DefineScope();

   private:
   SQLContext& sqlContext;
};

class SQLContext {
   public:
   SQLContext() : definedAttributes(), resolver() {
      definedAttributes.push({});
   };

   std::shared_ptr<catalog::Catalog> catalog;
   std::vector<std::shared_ptr<SQLScope>> scopes;
   std::shared_ptr<SQLScope> currentScope;

   std::stack<std::vector<std::pair<std::string, std::shared_ptr<ast::NamedResult>>>> definedAttributes;
   using renameCte = std::vector<std::pair<std::shared_ptr<ast::NamedResult>, std::shared_ptr<ast::NamedResult>>>;
   std::unordered_map<std::string, std::pair<ast::TargetInfo, std::shared_ptr<ast::BoundCTENode>>> ctes;

   std::unordered_map<std::string, mlir::Value> translatedCtes;

   llvm::ScopedHashTable<std::string, std::shared_ptr<ast::NamedResult>, StringInfo> resolver;
   using ResolverScope = llvm::ScopedHashTable<std::string, std::shared_ptr<ast::NamedResult>, StringInfo>::ScopeTy;

   std::unordered_map<std::string, size_t> scopeUnifier;

   void pushNewScope();
   void pushNewScope(std::shared_ptr<SQLScope> newScope);
   void popCurrentScope();

   ResolverScope createResolverScope();

   DefineScope createDefineScope();

   std::vector<std::pair<std::string, std::shared_ptr<ast::NamedResult>>> getTopDefinedColumns();

   void mapAttribute(ResolverScope& scope, std::string name, std::shared_ptr<ast::NamedResult> columnInfo);
   std::vector<std::shared_ptr<ast::NamedResult>> mapAttribute(ResolverScope& scope, std::string sqlScopeName, std::string uniqueScope, std::shared_ptr<catalog::TableCatalogEntry> tableCatalogEntry);
   void mapAttribute(ResolverScope& scope, std::string name, std::vector<std::shared_ptr<ast::NamedResult>> targetInfos);
   std::shared_ptr<ast::NamedResult> getNamedResultInfo(location loc, std::string name);
   void replace(ResolverScope& scope, std::shared_ptr<ast::NamedResult> old, std::shared_ptr<ast::NamedResult> value);

   std::string getUniqueScope(std::string base);

   //std::vector<std::pair<size_t, ast::ColumnInfo>> findColumn(const std::string& columnName) const;
   //std::vector<std::pair<size_t, ast::ColumnInfo>> findColumn(const std::string& columnName, const std::string& alias) const;

   std::string toString() const;
};
} // namespace lingodb::analyzer