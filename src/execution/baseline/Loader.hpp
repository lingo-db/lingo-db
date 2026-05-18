#pragma once

#if defined(__APPLE__)
#include "tpde/MachOMapper.hpp"
#else
#include "tpde/ElfMapper.hpp"
#endif

#include <cstdio>
#include <filesystem>
#include <sstream>
#include <dlfcn.h>

namespace lingodb::execution::baseline {

// In-memory JIT mapper and its assembler, selected per object format.
// The ELF and Mach-O mappers expose an identical API (map / get_sym_addr).
#if defined(__APPLE__)
using InMemoryMapper = tpde::macho::MachOMapper;
using InMemoryAssembler = tpde::macho::AssemblerMachO;
#else
using InMemoryMapper = tpde::elf::ElfMapper;
using InMemoryAssembler = tpde::elf::AssemblerElf;
#endif
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
   bool hasError = false;
};

class InMemoryLoader final : public DynamicLoader {
   InMemoryMapper mapper;
   tpde::SymRef mainFunc;

   public:
   InMemoryLoader(InMemoryAssembler& assembler, Error& error, const tpde::SymRef mainFunc)
      : DynamicLoader(error),
        mainFunc(mainFunc) {
      if (!mapper.map(assembler, [](const std::string_view name) {
             return dlsym(RTLD_DEFAULT, std::string(name).c_str());
          })) {
         hasError = true;
         error.emit() << "Could not map/link the compiled query module into memory\n";
      }
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
#if defined(__APPLE__)
      const std::string linkedFileName = std::string{outFileName} + ".dylib";
#else
      const std::string linkedFileName = std::string{outFileName} + ".so";
#endif
      auto* outFile = std::fopen((std::string{outFileName} + ".o").c_str(), "wb");
      if (!outFile) {
         error.emit() << "Could not open output file for baseline object: " << objFileName << " (" << strerror(errno) << ")\n";
         hasError = true;
         return;
      }
      if (std::fwrite(objFile.data(), 1, objFile.size(), outFile) != objFile.size()) {
         error.emit() << "Could not write object file to output file: " << objFileName << " (" << strerror(errno)
                      << ")\n";
         hasError = true;
         return;
      }
      if (std::fclose(outFile) != 0) {
         error.emit() << "Could not close output file: " << objFileName << " (" << strerror(errno) << ")\n";
         hasError = true;
         return;
      }
#if defined(__APPLE__)
      std::string cmd = std::string("cc -dynamiclib -o ") + linkedFileName + " " + objFileName;
#else
      std::string cmd = std::string("cc -shared -fPIC -o ") + linkedFileName + " " + objFileName;
#endif
      auto* pPipe = ::popen(cmd.c_str(), "r");
      if (pPipe == nullptr) {
         hasError = true;
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
         hasError = true;
         error.emit() << "Could not compile query module statically (Pipe could not be closed)";
         return;
      }
      handle = dlopen(linkedFileName.c_str(), RTLD_LAZY);
      if (const char* dlsymError = dlerror()) {
         hasError = true;
         error.emit() << "Cannot open object file: " << std::string(dlsymError) << "\nerror: " << strerror(errno) << "\n";
         return;
      }
   }

   mainFnType getMainFunction() override {
      const auto mainFunc = reinterpret_cast<mainFnType>(dlsym(handle, "main"));
      if (const char* dlsymError = dlerror()) {
         error.emit() << "Could not load symbol for main function: " << std::string(dlsymError) << "\nerror:"
                      << strerror(errno) << "\n";
         hasError = true;
         return nullptr;
      }
      return mainFunc;
   }

   void teardown() override {
      dlclose(handle);
   }
};
}
