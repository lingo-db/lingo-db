#pragma once

#include "lingodb/compiler/frontend/node_factory.h"
#include "lingodb/compiler/frontend/sql-parser/gen/parser.hpp"
#define YY_DECL \
   lingodb::parser::symbol_type yylex(driver& drv)
// ... and declare it for the parser's sake.
YY_DECL;

class driver {
   public:
   driver();
   ~driver() {
   };
   std::vector<std::shared_ptr<lingodb::ast::AstNode>> result;
   int parse(const std::string& f, bool isFile);
   void scan_begin(bool isFile);
   void scan_end();
   lingodb::location location;
   std::string file;
   bool trace_scanning;
   bool trace_parsing;
   lingodb::ast::NodeFactory nf;
};