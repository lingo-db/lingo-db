#include "lingodb/runtime/WASM.h"

#include "lingodb/scheduler/Scheduler.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CPYTHON_WASM_SOURCE_DIR
#error CPYTHON_WASM_SOURCE_DIR not defined
#endif
#ifndef CPYTHON_WASM_FILE
#error CPYTHON_WASM_FILE not defined
#endif

namespace lingodb::wasm {

wasm_engine_t* WASM::engine = nullptr;

namespace {
std::once_flag engineInitFlag;

void initEngineOnce() {
   std::call_once(engineInitFlag, [] {
      WASM::engine = wasm_engine_new();
      if (!WASM::engine) {
         throw std::runtime_error("wasm_engine_new failed");
      }
   });
}

[[noreturn]] void throwLastWasmerError(const char* prefix) {
   int len = wasmer_last_error_length();
   if (len <= 0) {
      throw std::runtime_error(std::string(prefix) + ": (no error message)");
   }
   std::string buf(len, '\0');
   wasmer_last_error_message(buf.data(), len);
   throw std::runtime_error(std::string(prefix) + ": " + buf);
}

std::vector<uint8_t> readFileBytes(const char* path) {
   std::ifstream f(path, std::ios::binary | std::ios::ate);
   if (!f) {
      throw std::runtime_error(std::string("failed to open WASM file: ") + path);
   }
   auto sz = f.tellg();
   f.seekg(0);
   std::vector<uint8_t> buf(sz);
   f.read(reinterpret_cast<char*>(buf.data()), sz);
   if (!f) {
      throw std::runtime_error(std::string("short read on WASM file: ") + path);
   }
   return buf;
}
} // namespace

WASMSession* WASM::initializeWASM() {
   // Wasmer's C API builds an internal tokio multi-thread runtime per
   // wasi_env_new (one per worker). UDFs are pure single-threaded compute, so
   // tell tokio to keep one worker thread per runtime instead of one per CPU.
   // Cuts peak thread count ~85% (e.g. 4 workers: 141 -> 17). setenv with
   // overwrite=0 leaves any user-provided value in place.
   ::setenv("TOKIO_WORKER_THREADS", "1", 0);

   initEngineOnce();

   wasm_store_t* store = wasm_store_new(engine);
   if (!store) throwLastWasmerError("wasm_store_new");

   // Load the precompiled artifact produced by `wasmer compile` (the Wasmer
   // analogue of WAMR's wamrc -> python.aot). Falls back to wasm_module_new
   // for a plain .wasm if the file's first 4 bytes are the wasm magic.
   auto fileBytes = readFileBytes(CPYTHON_WASM_FILE);
   wasm_byte_vec_t bytesVec;
   wasm_byte_vec_new(&bytesVec, fileBytes.size(),
                     reinterpret_cast<const wasm_byte_t*>(fileBytes.data()));
   bool isWasm = fileBytes.size() >= 4 && fileBytes[0] == 0x00 &&
                 fileBytes[1] == 0x61 && fileBytes[2] == 0x73 && fileBytes[3] == 0x6d;
   wasm_module_t* module = isWasm ? wasm_module_new(store, &bytesVec)
                                  : wasm_module_deserialize(store, &bytesVec);
   wasm_byte_vec_delete(&bytesVec);
   if (!module) throwLastWasmerError("wasm_module_new/deserialize");

   // Configure WASI — mirrors the dirList + mapDirList we passed to
   // wasm_runtime_set_wasi_args under WAMR. argv[0] spoofs the program name
   // so CPython's relative-path lib lookup works.
   wasi_config_t* cfg = wasi_config_new("./python.wasm");
   if (!wasi_config_preopen_dir(cfg, CPYTHON_WASM_BUILD_DIR)) {
      throw std::runtime_error("wasi_config_preopen_dir(BUILD_DIR) failed");
   }
   if (!wasi_config_mapdir(cfg, "/", CPYTHON_WASM_SOURCE_DIR)) {
      throw std::runtime_error("wasi_config_mapdir(/) failed");
   }
   if (!wasi_config_mapdir(cfg, "/lib/python3.14",
                           CPYTHON_WASM_SOURCE_DIR "/Lib")) {
      throw std::runtime_error("wasi_config_mapdir(/lib/python3.14) failed");
   }
   wasi_config_inherit_stdout(cfg);
   wasi_config_inherit_stderr(cfg);

   wasi_env_t* wasiEnv = wasi_env_new(store, cfg);
   if (!wasiEnv) throwLastWasmerError("wasi_env_new");

   wasm_extern_vec_t imports;
   if (!wasi_get_imports(store, wasiEnv, module, &imports)) {
      throwLastWasmerError("wasi_get_imports");
   }

   wasm_instance_t* instance = wasm_instance_new(store, module, &imports, nullptr);
   if (!instance) throwLastWasmerError("wasm_instance_new");

   if (!wasi_env_initialize_instance(wasiEnv, store, instance)) {
      throwLastWasmerError("wasi_env_initialize_instance");
   }
   wasm_extern_vec_delete(&imports);

   wasm_extern_vec_t exports;
   wasm_instance_exports(instance, &exports);
   wasm_exporttype_vec_t exportTypes;
   wasm_module_exports(module, &exportTypes);

   auto* session = new WASMSession{store, module, instance, wasiEnv, exports, exportTypes};
   session->callPyFunc<void>(CommonPyFunc::Py_Initialize);
   return session;
}

} // namespace lingodb::wasm
