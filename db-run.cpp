#include "runner/runner.h"
#include <string>

int main(int argc, char** argv) {
   std::string inputFileName = "-";
   if (argc > 1) {
      inputFileName = std::string(argv[1]);
   }
   Runner runner;
   runner.load(inputFileName);
   runner.lower();
   runner.lowerToLLVM();
   runner.runJit();
   return 0;
}