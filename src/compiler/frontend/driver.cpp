#include "lingodb/compiler/frontend/driver.h"

driver::driver()
   : trace_parsing(false), trace_scanning(false) {}

int driver::parse(const std::string& f, bool isFile) {
   file = f;
   location.initialize(&file);
   scan_begin(isFile);
   lingodb::parser parse(*this);
   parse.set_debug_level(trace_parsing);
   int res = parse();
   scan_end();
   return res;
}