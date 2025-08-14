#pragma once

#include <tpde/ElfMapper.hpp>

#include <cstdio>
#include <filesystem>
#include <sstream>
#include <dlfcn.h>

namespace lingodb::execution::baseline {
class DynamicLoader {
   protected:
   Error& error;

   public:
   DynamicLoader(Error& error)
      : error(error) {
   }

   virtual ~DynamicLoader() = default;

   virtual void teardown() {
   }

   virtual mainFnType getMainFunction() { return nullptr; }
   bool has_error = false;
};

template <typename Assembler>
class InMemoryLoader final : public DynamicLoader {
   tpde::ElfMapper mapper;
   tpde::SymRef mainFunc;

   public:
   InMemoryLoader(Assembler& assembler, Error& error, const tpde::SymRef mainFunc)
      : DynamicLoader(error),
        mainFunc(mainFunc) {
      mapper.map(assembler, [](const std::string_view name) {
         return dlsym(RTLD_DEFAULT, std::string(name).c_str());
      });
   }

   mainFnType getMainFunction() override {
      return reinterpret_cast<mainFnType>(mapper.get_sym_addr(mainFunc));
   }
};

template <typename Assembler>
class DebugLoader final : public DynamicLoader {
   void* handle = nullptr;

   public:
   DebugLoader(Assembler& assembler, Error& error, const std::string_view outFileName)
      : DynamicLoader(error) {
      const auto objFile = assembler.build_object_file();
      const std::string objFileName = std::string{outFileName} + ".o";
      const std::string linkedFileName = std::string{outFileName} + ".so";
      auto* outFile = std::fopen((std::string{outFileName} + ".o").c_str(), "wb");
      if (!outFile) {
         error.emit() << "Could not open output file for baseline object: " << objFileName << " (" << strerror(errno) << ")\n";
         has_error = true;
         return;
      }
      if (std::fwrite(objFile.data(), 1, objFile.size(), outFile) != objFile.size()) {
         error.emit() << "Could not write object file to output file: " << objFileName << " (" << strerror(errno)
                      << ")\n";
         has_error = true;
         return;
      }
      if (std::fclose(outFile) != 0) {
         error.emit() << "Could not close output file: " << objFileName << " (" << strerror(errno) << ")\n";
         has_error = true;
         return;
      }
      std::string cmd = std::string("cc -shared -o ") + linkedFileName + " " + objFileName;
      auto* pPipe = ::popen(cmd.c_str(), "r");
      if (pPipe == nullptr) {
         has_error = true;
         error.emit() << "Could not compile query module statically (Pipe could not be opened)";
         return;
      }
      std::array<char, 256> buffer;
      std::string result;
      while (not std::feof(pPipe)) {
         auto bytes = std::fread(buffer.data(), 1, buffer.size(), pPipe);
         result.append(buffer.data(), bytes);
      }
      auto rc = ::pclose(pPipe);
      if (WEXITSTATUS(rc)) {
         has_error = true;
         error.emit() << "Could not compile query module statically (Pipe could not be closed)";
         return;
      }
      handle = dlopen(linkedFileName.c_str(), RTLD_LAZY);
      if (const char* dlsymError = dlerror()) {
         has_error = true;
         error.emit() << "Cannot open object file: " << std::string(dlsymError) << "\nerror: " << strerror(errno) << "\n";
         return;
      }
   }

   mainFnType getMainFunction() override {
      const auto mainFunc = reinterpret_cast<mainFnType>(dlsym(handle, "main"));
      if (const char* dlsymError = dlerror()) {
         error.emit() << "Could not load symbol for main function: " << std::string(dlsymError) << "\nerror:"
                      << strerror(errno) << "\n";
         has_error = true;
         return nullptr;
      }
      return mainFunc;
   }

   void teardown() override {
      dlclose(handle);
   }
};
}
