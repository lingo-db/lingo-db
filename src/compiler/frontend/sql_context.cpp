#include "lingodb/compiler/frontend/sql_context.h"

#include "lingodb/compiler/frontend/frontend_error.h"

#include <iostream>
namespace lingodb::analyzer {
using ResolverScope = llvm::ScopedHashTable<std::string, std::shared_ptr<ast::NamedResult>, StringInfo>::ScopeTy;
ASTTransformContext::ASTTransformContext() : currentScope(std::make_shared<ASTTransformScope>()) {
   scopeStack.push(currentScope);
}

DefineScope::DefineScope(SQLContext& sqlContext) : sqlContext(sqlContext) {
   sqlContext.definedAttributes.push({});
}
DefineScope::~DefineScope() {
   sqlContext.definedAttributes.pop();
}

void SQLContext::pushNewScope() {
   auto newScope = std::make_shared<SQLScope>();
   newScope->parent = currentScope;
   currentScope = newScope;
   scopes.push_back(newScope);
}

void SQLContext::pushNewScope(std::shared_ptr<SQLScope> newScope) {
   newScope->parent = currentScope;
   currentScope = newScope;
   scopes.push_back(newScope);
}
void SQLContext::popCurrentScope() {
   if (!currentScope->parent) {
      throw std::runtime_error("Cannot pop current scope");
      return;
   }
   currentScope = currentScope->parent;
   scopes.pop_back();
}
ResolverScope SQLContext::createResolverScope() {
   return ResolverScope(resolver);
}

DefineScope SQLContext::createDefineScope() {
   return DefineScope(*this);
}

std::vector<std::pair<std::string, std::shared_ptr<ast::NamedResult>>> SQLContext::getTopDefinedColumns() {
   return definedAttributes.top();
}
void SQLContext::mapAttribute(ResolverScope& scope, std::string name, std::shared_ptr<ast::NamedResult> columnInfo) {
   definedAttributes.top().push_back({name, columnInfo});
   resolver.insertIntoScope(&scope, name, columnInfo);
}

std::vector<std::shared_ptr<ast::NamedResult>> SQLContext::mapAttribute(ResolverScope& scope, std::string sqlScopeName, std::string uniqueScope, std::shared_ptr<catalog::TableCatalogEntry> tableCatalogEntry) {
   std::vector<std::shared_ptr<ast::NamedResult>> result;
   for (auto c : tableCatalogEntry->getColumns()) {
      auto columnInfo = std::make_shared<ast::NamedResult>(uniqueScope, c);

      mapAttribute(scope, sqlScopeName + "." + std::string(c.getColumnName()), columnInfo);
      mapAttribute(scope, std::string(c.getColumnName()), columnInfo);
      result.push_back(columnInfo);
   }
   return result;
}

void SQLContext::mapAttribute(ResolverScope& scope, std::string name, std::vector<std::shared_ptr<ast::NamedResult>> targetInfos) {
   for (auto c : targetInfos) {
      //Better
      std::string cName = c->displayName.empty() ? c->name : c->displayName;

      mapAttribute(scope, name + "." + cName, c);

      mapAttribute(scope, cName, c);
   }
}
std::shared_ptr<ast::NamedResult> SQLContext::getNamedResultInfo(location loc, std::string name) {
   if (!resolver.count(name)) {
      std::stringstream ss;

      throw frontend_error("Could not resolve: " + name, loc);
   }
   const auto res = resolver.lookup(name);
   return res;
}

void SQLContext::replace(ResolverScope& scope, std::shared_ptr<ast::NamedResult> old, std::shared_ptr<ast::NamedResult> value) {
   std::vector<std::pair<std::string, std::shared_ptr<ast::NamedResult>>> toReplace;
   std::ranges::copy_if(definedAttributes.top(), std::back_inserter(toReplace), [&](auto& p) { return p.second == old; });
   for (auto& c: toReplace) {
      mapAttribute(scope, c.first, value);
   }
}

std::string SQLContext::getUniqueScope(std::string base) {
   if (scopeUnifier.contains(std::string(base))) {
      scopeUnifier[std::string(base)] += 1;
      return std::string(base) + std::string("_u_") + std::to_string(scopeUnifier[std::string(base)]);
   } else {
      scopeUnifier[std::string(base)] = 0;
      return std::string(base);
   }
}

/*std::vector<std::pair<size_t, ast::ColumnInfo>> SQLContext::findColumn(const std::string& columnName) const {
   std::vector<std::pair<size_t, ast::ColumnInfo>> columns{};
   std::string foundTable;
   auto current = currentScope;
   while (current) {
      for (auto t : current->columns) {
         auto colsOfTable = t.second;
         foundTable = t.first;
         auto it = std::find_if(colsOfTable.begin(), colsOfTable.end(), [&columnName](std::pair<size_t, ast::ColumnInfo> col) {
            const std::string& colName = col.second.column.getColumnName();
            return colName.length() == columnName.length() &&
               std::equal(colName.begin(), colName.end(), columnName.begin(),
                          [](char a, char b) { return std::tolower(a) == std::tolower(b); });
         });
         if (it != colsOfTable.end()) {
            columns.emplace_back(*it);
         }
      }
      current = current->parent;
   }
   return columns;
}*/

std::string SQLContext::toString() const {
   return "";
}

} // namespace lingodb::analyzer