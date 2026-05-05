#include "lingodb/runtime/WASM.h"

#include "lingodb/scheduler/Scheduler.h"

#include <stdexcept>
#include <string>

#ifndef LINGODB_WASIX_WEBC_FILE
#error LINGODB_WASIX_WEBC_FILE not defined (set by vendored/cpython-wasm/CMakeLists.txt)
#endif
#ifndef LINGODB_WASIX_SITE_PACKAGES_DIR
#error LINGODB_WASIX_SITE_PACKAGES_DIR not defined
#endif
#ifndef LINGODB_WASIX_MODULE_CACHE_DIR
#error LINGODB_WASIX_MODULE_CACHE_DIR not defined
#endif

namespace lingodb::wasm {

WASMSession* WASM::initializeWASM() {
   // Hand off to the Rust shim. It owns the wasmer engine + module cache,
   // loads the webc, mounts the site-packages dir, instantiates, runs
   // __wasm_call_ctors / __wasi_init_tp / Py_Initialize.
   lingodb_wasix_session_t* shim = lingodb_wasix_session_new(
      LINGODB_WASIX_WEBC_FILE,
      LINGODB_WASIX_SITE_PACKAGES_DIR,
      LINGODB_WASIX_MODULE_CACHE_DIR);
   if (!shim) {
      const char* msg = lingodb_wasix_last_error();
      throw std::runtime_error(std::string("wasix session_new failed: ") +
                               (msg ? msg : "<no message>"));
   }
   return new WASMSession{shim};
}

} // namespace lingodb::wasm
