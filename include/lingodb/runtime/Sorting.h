#ifndef LINGODB_RUNTIME_SORTING_H
#define LINGODB_RUNTIME_SORTING_H
#include "lingodb/runtime/GrowingBuffer.h"

namespace lingodb::runtime {

bool canParallelSort(const size_t valueSize);
Buffer parallelSort(FlexibleBuffer& values, bool (*compareFn)(uint8_t*, uint8_t*));

} // end namespace lingodb::runtime
#endif //LINGODB_RUNTIME_SORTING_H