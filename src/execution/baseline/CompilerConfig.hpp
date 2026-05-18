#pragma once

#if defined(__x86_64__)
#include <tpde/x64/CompilerX64.hpp>
#elif defined(__aarch64__)
#include <tpde/arm64/CompilerA64.hpp>
#endif

namespace lingodb::execution::baseline {
// NOLINTBEGIN(readability-identifier-naming)

// use the default config
#if defined(__x86_64__)
struct CompilerConfig : tpde::x64::PlatformConfig {
};
#elif defined(__aarch64__)
#if defined(__APPLE__)
// macOS / Apple Silicon: Mach-O assembler + darwinpcs calling convention.
struct CompilerConfig : tpde::a64::PlatformConfigDarwin {
};
#else
struct CompilerConfig : tpde::a64::PlatformConfig {
};
#endif
#endif

// NOLINTEND(readability-identifier-naming)
}
