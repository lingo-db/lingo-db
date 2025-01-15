#ifndef LINGODB_STORAGE_ROWGROUP_H
#define LINGODB_STORAGE_ROWGROUP_H
#include <vector>
namespace lingodb::storage {
class Buffer {
   std::byte* data;
   size_t size;
};
class ColumnInRowGroup {

};
class RowGroup {
   size_t numRows;
   std::vector<ColumnInRowGroup> columns;
};
} // end namespace lingodb::storage

#endif //LINGODB_STORAGE_ROWGROUP_H
