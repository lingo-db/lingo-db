#pragma once

#if defined(__x86_64__) && false
#include <tpde/x64/CompilerX64.hpp>
#elif defined(__aarch64__) || true
#include <tpde/arm64/CompilerA64.hpp>
#endif

namespace lingodb::execution::baseline {
// NOLINTBEGIN(readability-identifier-naming)

// use the default config
#if defined(__x86_64__) && false
struct CompilerConfig : tpde::x64::PlatformConfig {
};
#elif defined(__aarch64__) || true
struct CompilerConfig : tpde::a64::PlatformConfig {
};
#endif


// NOLINTEND(readability-identifier-naming)
}
