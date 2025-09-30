#include "lingodb/compiler/frontend/driver.h"

Driver::Driver()
   : traceScanning(false), traceParsing(false) {}

int Driver::parse(const std::string& f, bool isFile) {
   file = f;
   location.initialize(&file);
   scanBegin(isFile);
   lingodb::parser parse(*this);
   parse.set_debug_level(traceParsing);
   int res = parse();
   scanEnd();
   return res;
}