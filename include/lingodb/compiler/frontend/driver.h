#ifndef LINGODB_COMPILER_FRONTEND_DRIVER_H
#define LINGODB_COMPILER_FRONTEND_DRIVER_H

#include "lingodb/compiler/frontend/generated/parser.hpp"
#include "lingodb/compiler/frontend/node_factory.h"

#define YY_DECL \
   lingodb::parser::symbol_type yylex(Driver& drv)
// ... and declare it for the parser's sake.
YY_DECL;

class Driver {
   public:
   Driver();
   ~Driver() {
   };
   std::vector<std::shared_ptr<lingodb::ast::AstNode>> result;
   int parse(const std::string& f, bool isFile);
   void scanBegin(bool isFile);
   void scanEnd();
   lingodb::location location;
   std::string file;
   bool traceScanning;
   bool traceParsing;
   lingodb::ast::NodeFactory nf;
};
#endif
