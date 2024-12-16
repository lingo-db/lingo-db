#include "lingodb/execution/Backend.h"
#include "lingodb/runtime/helpers.h"

#include <iostream>
extern "C" lingodb::runtime::VarLen32 createVarLen32(uint8_t* ptr, uint32_t len); //NOLINT(clang-diagnostic-return-type-c-linkage)
extern "C" uint64_t hashVarLenData(lingodb::runtime::VarLen32 str);
extern "C" void dumpString(lingodb::runtime::VarLen32 str) {
   std::cout << "string(\"" << str.str() << "\")" << std::endl;
}
extern "C" void dumpI64(int64_t i) {
   std::cout << "int(" << i << ")" << std::endl;
}
extern "C" void dumpF64(double d) {
   std::cout << "float(" << d << ")" << std::endl;
}
extern "C" void dumpBool(bool b) {
   std::cout << "bool(" << (b ? "true" : "false") << ")" << std::endl;
}

void lingodb::execution::visitBareFunctions(const std::function<void(std::string, void*)>& fn) {
   fn("createVarLen32", reinterpret_cast<void*>(&createVarLen32));
   fn("hashVarLenData", reinterpret_cast<void*>(&hashVarLenData));
   fn("dumpString", reinterpret_cast<void*>(&dumpString));
   fn("dumpI64", reinterpret_cast<void*>(&dumpI64));
   fn("dumpF64", reinterpret_cast<void*>(&dumpF64));
   fn("dumpBool", reinterpret_cast<void*>(&dumpBool));
}
