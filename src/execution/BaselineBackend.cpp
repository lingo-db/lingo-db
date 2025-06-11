#if BASELINE_ENABLED == 1
#if !defined(__linux__)
#error "Baseline backend is only supported on Linux systems."
#endif

#include "lingodb/execution/BaselineBackend.h"
#include "lingodb/utility/Setting.h"
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <tpde/CompilerBase.hpp>
#include <tpde/x64/CompilerX64.hpp>
#include <dlfcn.h>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>

lingodb::utility::Setting<std::string> baselineObjectFileOut("system.compilation.baseline_object_out", "");

namespace {
class SpdLogSpoof {
   // storage for the log messages
   std::ostringstream oss;
   std::shared_ptr<spdlog::sinks::ostream_sink_mt> ostream_sink;
   std::shared_ptr<spdlog::logger> logger;
   std::shared_ptr<spdlog::logger> old_logger;

   public:
   SpdLogSpoof() : ostream_sink(std::make_shared<spdlog::sinks::ostream_sink_mt>(oss)), logger(std::make_shared<spdlog::logger>("string_logger", ostream_sink)) {
      old_logger = spdlog::default_logger();
      spdlog::set_default_logger(logger);
   }

   ~SpdLogSpoof() {
      spdlog::set_default_logger(old_logger);
   }

   std::string drain_logs() {
      std::string ret = oss.str();
      oss.clear();
      return ret;
   }
};

class BaselineBackend : public lingodb::execution::ExecutionBackend {
   // adaptor mlir -> tpde
   struct IRAdaptor {
      mlir::ModuleOp* moduleOp;
      IRAdaptor(mlir::ModuleOp* moduleOp) : moduleOp(moduleOp) {}
   };

   // we will use the default config
   struct CompilerConfig : tpde::x64::PlatformConfig {
   };

   // cross-platform compiler base class
   template <typename Adaptor, typename Derived, typename Config>
   struct IRCompilerBase : tpde::CompilerBase<Adaptor, Derived, Config> {
      using Base = tpde::CompilerBase<Adaptor, Derived, Config>;

      IRCompilerBase(IRAdaptor* adaptor) : Base{adaptor} {
         static_assert(tpde::Compiler<Derived, Config>);
         // the class has to take an adaptor parameter but we know that it always will be the ir::IRAdaptor
         static_assert(std::is_same_v<Adaptor, IRAdaptor>);
      }
   };

   // x86_64 target specific compiler
   struct IRCompilerX64 : tpde::x64::CompilerX64<IRAdaptor, IRCompilerX64, IRCompilerBase, CompilerConfig> {
      using Base = tpde::x64::CompilerX64<IRAdaptor, IRCompilerX64, IRCompilerBase, CompilerConfig>;

      explicit IRCompilerX64(IRAdaptor* adaptor)
         : Base(adaptor) {
         static_assert(tpde::Compiler<IRCompilerX64, tpde::x64::PlatformConfig>);
      }
   };

   void execute(mlir::ModuleOp& moduleOp, lingodb::runtime::ExecutionContext* executionContext) override {
      SpdLogSpoof logSpoof;
      IRAdaptor adaptor{&moduleOp};

#if defined(__x86_64__)
      IRCompilerX64 compiler{&adaptor};
#else
#error "Baseline backend is only supported on x86_64 architecture."
#endif

      if (!compiler.compile()) {
         error.emit() << "Could not compile query module:\n"
                      << logSpoof.drain_logs() << "\n";
      }
      std::vector<uint8_t> objFileBytes = compiler.assembler.build_object_file();

      std::FILE* outFile;
      std::string outFileName;
      if (baselineObjectFileOut.getValue().empty()) {
         outFile = std::tmpfile();
         // be careful, this only works on Linux!
         outFileName = std::filesystem::read_symlink(
            std::filesystem::path("/proc/self/fd") / std::to_string(fileno(outFile)));
      } else {
         outFileName = baselineObjectFileOut.getValue();
         outFile = std::fopen(baselineObjectFileOut.getValue().c_str(), "wb");
         if (!outFile) {
            error.emit() << "Could not open output file for baseline object: " << baselineObjectFileOut.getValue() << " (" << strerror(errno) << ")\n";
            return;
         }
      }
      if (std::fwrite(objFileBytes.data(), 1, objFileBytes.size(), outFile) != objFileBytes.size()) {
         error.emit() << "Could not write object file to output file: " << baselineObjectFileOut.getValue() << " (" << strerror(errno) << ")\n";
         std::fclose(outFile);
         return;
      }

      void* handle = dlopen(outFileName.c_str(), RTLD_LAZY);
      const char* dlsymError = dlerror();
      if (dlsymError) {
         error.emit() << "Can not open static library: " << std::string(dlsymError) << "\nerror:" << strerror(errno) << "\n";
         return;
      }
      auto mainFunc = reinterpret_cast<lingodb::execution::mainFnType>(dlsym(handle, "main"));
      dlsymError = dlerror();
      if (dlsymError) {
         dlclose(handle);
         error.emit() << "Could not load symbol for main function: " << std::string(dlsymError) << "\nerror:" << strerror(errno) << "\n";
         return;
      }

      std::vector<size_t> measuredTimes;
      for (size_t i = 0; i < numRepetitions; i++) {
         auto executionStart = std::chrono::high_resolution_clock::now();
         mainFunc();
         auto executionEnd = std::chrono::high_resolution_clock::now();
         measuredTimes.push_back(std::chrono::duration_cast<std::chrono::microseconds>(executionEnd - executionStart).count() / 1000.0);
      }
      dlclose(handle);
      std::fclose(outFile);
   }
};
}

std::unique_ptr<lingodb::execution::ExecutionBackend> lingodb::execution::createBaselineBackend() {
   using tpde::CompilerBase;
   using tpde::x64::CompilerX64;
   return {nullptr};
}
#endif