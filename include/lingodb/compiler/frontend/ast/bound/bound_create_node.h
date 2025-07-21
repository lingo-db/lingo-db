#pragma once

#include "lingodb/compiler/frontend/ast/create_node.h"

namespace lingodb::ast {

class BoundColumnElement : public TableElement {
   public:
   BoundColumnElement(std::string name, catalog::NullableType datatype, bool primary)
      : TableElement(TableElementType::COLUMN), name(std::move(name)), datatype(datatype), primary(primary) {}

   std::string name;
   catalog::NullableType datatype;

   bool unique;
   bool primary;
};

}