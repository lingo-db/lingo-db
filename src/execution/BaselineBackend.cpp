#if BASELINE_ENABLED == 1

#include "lingodb/execution/BaselineBackend.h"
#include <tpde/CompilerBase.hpp>
#include <tpde/x64/CompilerX64.hpp>

namespace {

}

std::unique_ptr<lingodb::execution::ExecutionBackend> lingodb::execution::createBaselineBackend() {
   using tpde::CompilerBase;
   using tpde::x64::CompilerX64;
   return {nullptr};
}
#endif