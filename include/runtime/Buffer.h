#ifndef LINGODB_BUFFER_H
#define LINGODB_BUFFER_H
#include <cstdint>
#include <vector>
#include <stddef.h>
namespace runtime {
/*
 * Buffer: continuous memory area, which is directly accessed by generated code
 */
struct Buffer {
   size_t numElements;
   uint8_t* ptr;
};
struct BufferIterator {
   virtual bool isValid() = 0;
   virtual void next() = 0;
   virtual Buffer getCurrentBuffer() = 0;
   static bool isIteratorValid(BufferIterator* iterator);
   static void iteratorNext(BufferIterator* iterator);
   static Buffer iteratorGetCurrentBuffer(BufferIterator* iterator);
   static void destroy(BufferIterator* iterator);
   virtual ~BufferIterator(){}
};
class FlexibleBuffer {
   size_t totalLen;
   size_t currCapacity;
   std::vector<Buffer> buffers;
   size_t typeSize;

   void nextBuffer() {
      size_t nextCapacity = currCapacity * 2;
      buffers.push_back(Buffer{0, (uint8_t*) malloc(nextCapacity * typeSize)});
      currCapacity = nextCapacity;
   }

   public:
   FlexibleBuffer(size_t initialCapacity, size_t typeSize) : totalLen(0), currCapacity(initialCapacity), typeSize(typeSize) {
      buffers.push_back(Buffer{0, (uint8_t*) malloc(initialCapacity * typeSize)});
   }
   uint8_t* insert() {
      if (buffers.back().numElements == currCapacity) {
         nextBuffer();
      }
      totalLen++;
      auto* res = &buffers.back().ptr[typeSize * (buffers.back().numElements)];
      buffers.back().numElements++;
      return res;
   }
   template<class Fn>
   void iterate(const Fn& fn){
      for (auto buffer : buffers) {
         for (size_t i = 0; i < buffer.numElements; i++) {
            fn(&buffer.ptr[i * typeSize]);
         }
      }
   }
   const std::vector<Buffer>& getBuffers(){
      return buffers;
   }
   BufferIterator* createIterator();
   size_t getTypeSize() const {
      return typeSize;
   }
   size_t getLen() const;
};

}

#endif //LINGODB_BUFFER_H
