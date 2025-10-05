#pragma once

#include "lingodb/catalog/Defs.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/compiler/frontend/column_semantic.h"

#include <map>
#include <memory>
namespace lingodb::ast {
class BoundExpression;
} // namespace lingodb::ast
namespace lingodb::analyzer {
/**
 * The SQLScope stores the TargetInfo for the corresponding scope.
 * Each Subquery creates a new SQL Scope
 */
class SQLScope {
   public:
   //Stores the columns that should be produced at the end
   ast::TargetInfo targetInfo;

   std::shared_ptr<SQLScope> parent;
};
} // namespace lingodb::analyzer