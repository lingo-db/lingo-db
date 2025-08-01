#include "lingodb/compiler/mlir-support/eval.h"
#include "lingodb/execution/CBackend.h"
#include "lingodb/execution/Execution.h"
#include "lingodb/execution/Timing.h"
#include "lingodb/scheduler/Scheduler.h"
#include "lingodb/utility/Setting.h"

#include "mlir/IR/BuiltinOps.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>

namespace {
lingodb::utility::GlobalSetting<bool> eagerLoading("system.eager_loading", false);

class CppGenerator : public lingodb::execution::ExecutionBackend {
   std::string outputPath;

   public:
   CppGenerator(std::string outputPath) : outputPath(outputPath) {}

   private:
   void execute(mlir::ModuleOp& moduleOp, lingodb::runtime::ExecutionContext* executionContext) override {
      std::string translatedModule;
      llvm::raw_string_ostream sstream(translatedModule);

      if (lingodb::execution::emitC(moduleOp.getOperation(), sstream, false, true).failed()) {
         error.emit() << "Can not translate module to c++";
         return;
      }
      std::regex r("void main\\(\\) \\{");

      translatedModule = std::regex_replace(translatedModule, r, "extern \"C\" void mainFunc() {");

      std::ofstream outputFile(outputPath);
      outputFile << "#include<cstdint>\n"
                    "#include<tuple>\n"
                    "#include<bit>\n"
                    "#include<math.h>\n"
                    "#include \"lingodb/runtime/helpers.h\"\n"
                    "#include \"lingodb/runtime/Buffer.h\"\n"
                    "namespace runtime=lingodb::runtime;\n"
                    "namespace std {\n"
                    "template <>\n"
                    "struct make_unsigned<__int128> {\n"
                    "   typedef __uint128_t type;\n"
                    "};\n"
                    "}"
                    "size_t hash_64(size_t val){\n"
                    "\tsize_t p1=11400714819323198549ull;\n"
                    "\tsize_t m1=val*p1;\n"
                    "\treturn m1 ^ __builtin_bswap64(m1);\n"
                    "}"
                    "size_t hash_combine(size_t h1, size_t h2){\n"
                    "\treturn h2 ^ __builtin_bswap64(h1);\n"
                    "}\n"
                    "extern \"C\" size_t hashVarLenData(runtime::VarLen32);\n"
                    ""
                 << std::endl;
      outputFile << translatedModule << std::endl;
      outputFile.close();
   }
};
} // namespace
int main(int argc, char** argv) {
   using namespace lingodb;
   if (argc <= 3) {
      std::cerr << "USAGE: compile-to-sql *.sql database *.cpp" << std::endl;
      return 1;
   }
   std::string inputFileName = std::string(argv[1]);
   std::string directory = std::string(argv[2]);
   std::string outputFileName = std::string(argv[3]);
   std::cout << "Loading Database from: " << directory << '\n';
   auto session = runtime::Session::createSession(directory, eagerLoading.getValue());

   lingodb::compiler::support::eval::init();
   execution::ExecutionMode runMode = execution::getExecutionMode();
   auto queryExecutionConfig = execution::createQueryExecutionConfig(runMode, true);
   unsetenv("PERF_BUILDID_DIR");
   queryExecutionConfig->timingProcessor = std::make_unique<execution::TimingPrinter>(inputFileName);
   queryExecutionConfig->executionBackend = std::make_unique<CppGenerator>(outputFileName);

   auto scheduler = scheduler::startScheduler();
   auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig), *session);
   executer->fromFile(inputFileName);
   scheduler::awaitEntryTask(std::make_unique<execution::QueryExecutionTask>(std::move(executer)));
   return 0;
}
