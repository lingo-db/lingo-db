#ifndef LINGODB_RUNTIME_STRUCTRUNTIME_H
#define LINGODB_RUNTIME_STRUCTRUNTIME_H
#include "lingodb/runtime/Buffer.h"

#include <lingodb/runtime/helpers.h>
namespace lingodb::runtime {
class Struct {
   StorageClass storageClass;

   public:
   std::vector<uint8_t> values;
   Struct(size_t totalSize, StorageClass storageClass = StorageClass::REFCOUNTED) : storageClass(storageClass), values(totalSize) {}
   static Struct* create(size_t totalSize);
   uint8_t* data();
   static void cleanupUse(Struct* s);
   static void addUse(Struct* s);
   static Struct* promoteToGlobal(Struct* s);
};

}; // namespace lingodb::runtime

#endif //LINGODB_RUNTIME_STRUCTRUNTIME_H
