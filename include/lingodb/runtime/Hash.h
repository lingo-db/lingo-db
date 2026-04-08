#ifndef LINGODB_RUNTIME_HASH_H
#define LINGODB_RUNTIME_HASH_H

#include "lingodb/runtime/helpers.h"

#include <arrow/array.h>

#include <vector>

EXPORT uint64_t hashVarLenData(lingodb::runtime::VarLen32 str);

namespace lingodb::runtime {

/// Fold one column into the per-row running hash in running (size must equal num_rows)
void dbHashApplyColumn(std::vector<uint64_t>& running, const arrow::Array& arr, int64_t num_rows, bool isFirstColumn);

} // namespace lingodb::runtime

#endif
