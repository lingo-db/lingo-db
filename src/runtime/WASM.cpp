#include "lingodb/runtime/WASM.h"

#include "bh_platform.h"
#include "bh_read_file.h"
#include "lingodb/scheduler/Scheduler.h"
#include "wasm_c_api_internal.h"
#include "wasm_export.h"

#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <bits/this_thread_sleep.h>
#define WASM_STACK_SIZE 16777216
#define WASM_HEAP_SIZE (16777216)
#ifndef CPYTHON_WASM_SOURCE_DIR
#error CPYTHON_WASM_SOURCE_DIR not defined
#endif
namespace lingodb::wasm {

WASMSession WASM::initializeWASM() {
   while (!wasm_runtime_thread_env_inited()) {
      wasm_runtime_init_thread_env();
   }
   char errorBuf[128] = {0};
   uint32 size;
   /* initialize the wasm runtime by default configuration */
   wasm_runtime_init();
   /* read WASM file to memory buffer */
   uint8_t* buffer = reinterpret_cast<uint8_t*>(bh_read_file_to_buffer(CPYTHON_WASM_AOT_FILE, &size));
   if (!buffer) {
      throw std::runtime_error("Failed to read WASM file");
   }
   wasm_module_t module = wasm_runtime_load(buffer, size, errorBuf, sizeof(errorBuf));
   if (!module) {
      throw std::runtime_error(errorBuf);
   }

   /* --- 1. Setup 'dir_list' (for --dir=.) --- */
   // This sets the CWD for python.aot (This is correct)
   std::array dirList{
      CPYTHON_WASM_BUILD_DIR};

   /* --- 2. Setup 'map_dir_list' (for --map-dir=...) --- */
   std::array mapDirList{
      "/::" CPYTHON_WASM_SOURCE_DIR,
      "/lib/python3.14::" CPYTHON_WASM_SOURCE_DIR "/Lib",
   };

   // spoof fake program name (since CPython uses it to load relativ libs)
   std::string programName = "./python.aot";
   std::array wasmArgs{programName.data()};

   wasm_runtime_set_wasi_args(module,
                              dirList.data(), dirList.size(),
                              mapDirList.data(), mapDirList.size(),
                              nullptr, 0,
                              wasmArgs.data(), wasmArgs.size());

   /* create an instance of the WASM module (WASM linear memory is ready) */
   wasm_module_inst_t moduleInst = wasm_runtime_instantiate(module, WASM_STACK_SIZE, WASM_HEAP_SIZE,
                                                            errorBuf, sizeof(errorBuf));

   if (!moduleInst) {
      wasm_runtime_unload(module);
      throw std::runtime_error(errorBuf);
   }
   wasm_exec_env_t execEnv = wasm_runtime_create_exec_env(moduleInst, WASM_STACK_SIZE);
   if (!execEnv) {
      throw std::runtime_error("Create exec env failed");
   }
#ifdef ASAN_ACTIVE

#endif
   WASMSession wasmSession{execEnv, moduleInst};
#if !ASAN_ACTIVE
   wasm_runtime_set_native_stack_boundary(execEnv, scheduler::getStackBoundary());
#endif
   wasmSession.callPyFunc<void>("Py_Initialize");
   return wasmSession;
}
} // namespace lingodb::wasm