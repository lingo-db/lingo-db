#pragma once

namespace garel {

#define GEN_PASS_DECL
#include "lingodb/compiler/Dialect/garel/GARelPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "lingodb/compiler/Dialect/garel/GARelPasses.h.inc"

} // namespace garel
