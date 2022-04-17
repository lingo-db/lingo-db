
#ifndef RUNTIME_VECTOR_H
#define RUNTIME_VECTOR_H
#include "runtime/helpers.h"
namespace runtime {
class Vector {
   size_t len;
   size_t cap;
   uint8_t* ptr;
   size_t typeSize;

   public:
   Vector(size_t cap, size_t typeSize) : len(0), cap(cap), ptr((uint8_t*) malloc(typeSize * cap)), typeSize(typeSize) {}
   template <class T>
   T* ptrAt(size_t i) {
      return (T*) &ptr[i * typeSize];
   }
   void resize();
   static Vector* create(size_t sizeOfType, size_t initialCapacity);
   size_t getLen() const;
   size_t getCap() const;
   uint8_t* getPtr() const;
   size_t getTypeSize() const;
   void sort(bool (*compareFn)(uint8_t*, uint8_t*));
   static void destroy(Vector* vec);
   ~Vector(){
      free(ptr);
   }
};
} // end namespace runtime
#endif // RUNTIME_VECTOR_H
