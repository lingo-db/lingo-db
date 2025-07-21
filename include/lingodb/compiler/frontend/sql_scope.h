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
   std::map<std::string, std::pair<size_t, ast::ColumnInfo>> tables;
   std::map<std::string, std::vector<std::pair<size_t, ast::ColumnInfo>>> columns;
   std::map<std::string, std::vector<std::pair<size_t, ast::FunctionInfo>>> functionsEntry;
   //Stores the columns that should be produced at the end
   ast::TargetInfo targetInfo;
   std::vector<std::shared_ptr<ast::BoundExpression>> evalBefore;

   std::shared_ptr<SQLScope> parent;
   size_t id = 0;
};
} // namespace lingodb::analyzer