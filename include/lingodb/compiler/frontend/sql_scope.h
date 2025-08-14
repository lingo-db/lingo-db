#pragma once

#include "lingodb/catalog/Defs.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/compiler/frontend/column_semantic.h"

#include <map>
#include <memory>
namespace lingodb::ast {
class BoundExpression;
}
namespace lingodb::analyzer {

class SQLScope {
   public:
   std::map<std::string, std::pair<size_t, ast::NamedResult>> tables;
   std::map<std::string, std::vector<std::pair<size_t, ast::NamedResult>>> columns;
   std::map<std::string, std::vector<std::pair<size_t, ast::NamedResult>>> functionsEntry;
   //Stores the columns that should be produced at the end
   ast::TargetInfo targetInfo;

   //Stores the grouping functions
   std::vector<std::shared_ptr<ast::BoundFunctionExpression>> groupingFunctions;

   std::shared_ptr<SQLScope> parent;
   size_t id = 0;
};
} // namespace lingodb::analyzer