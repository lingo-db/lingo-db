#ifndef LINGODB_RUNTIME_STORAGE_RESTRICTIONS_H
#define LINGODB_RUNTIME_STORAGE_RESTRICTIONS_H
#include "lingodb/runtime/ArrowView.h"
#include "lingodb/runtime/storage/TableStorage.h"
#include <cstddef>
#include <memory>
#include <vector>
namespace lingodb::runtime {
class Filter {
   public:
   virtual size_t filter(size_t len, uint16_t* currSelVec, uint16_t* nextSelVec, const lingodb::runtime::ArrayView* arrayView, size_t offset) = 0;
   virtual ~Filter() {}
};
class Restrictions {
   std::vector<std::pair<std::unique_ptr<lingodb::runtime::Filter>, size_t>> filters;

   public:
   std::pair<size_t, uint16_t*> applyFilters(size_t offset, size_t length, uint16_t* selVec1, uint16_t* selVec2, std::function<const ArrayView*(size_t)> getArrayView);
   static std::unique_ptr<Restrictions> create(std::vector<FilterDescription> filterDescs, const arrow::Schema& schema);
};
};

#endif //LINGODB_RUNTIME_STORAGE_RESTRICTIONS_H
