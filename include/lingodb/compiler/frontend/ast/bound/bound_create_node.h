#ifndef LINGODB_COMPILER_FRONTEND_AST_BOUND_BOUND_CREATE_NODE_H
#define LINGODB_COMPILER_FRONTEND_AST_BOUND_BOUND_CREATE_NODE_H

#include "lingodb/compiler/frontend/ast/create_node.h"

namespace lingodb::ast {
/**
 * Represents a column element inside a create table element
 * CREATE TABLE x (y int primary, z int). Both y and z are each represented by a BoundColumnElement
 */
class BoundColumnElement : public TableElement {
   public:
   BoundColumnElement(std::string name, NullableType datatype, bool primary)
      : TableElement(TableElementType::COLUMN), name(std::move(name)), datatype(datatype), primary(primary) {}

   std::string name;
   NullableType datatype;

   bool unique;
   bool primary;
};

} // namespace lingodb::ast
#endif
