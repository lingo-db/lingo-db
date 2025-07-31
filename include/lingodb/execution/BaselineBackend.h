#ifndef LINGODB_EXECUTION_BASELINEBACKEND_H
#define LINGODB_EXECUTION_BASELINEBACKEND_H

#include "Backend.h"

namespace lingodb::execution {
std::unique_ptr<ExecutionBackend> createBaselineBackend();
} // namespace lingodb::execution
#endif
