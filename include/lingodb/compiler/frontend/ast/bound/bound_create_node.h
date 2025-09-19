#ifndef LINGODB_COMPILER_FRONTEND_AST_BOUND_BOUND_CREATE_NODE_H
#define LINGODB_COMPILER_FRONTEND_AST_BOUND_BOUND_CREATE_NODE_H



#include "lingodb/compiler/frontend/ast/create_node.h"

namespace lingodb::ast {

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
