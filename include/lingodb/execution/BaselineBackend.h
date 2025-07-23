#ifndef LINGODB_EXECUTION_BASELINE_BACKEND_H
#define LINGODB_EXECUTION_BASELINE_BACKEND_H

#include "Backend.h"

namespace lingodb::execution {
std::unique_ptr<ExecutionBackend> createBaselineBackend();
} // namespace lingodb::execution
#endif //LINGODB_EXECUTION_BASELINE_BACKEND_H
