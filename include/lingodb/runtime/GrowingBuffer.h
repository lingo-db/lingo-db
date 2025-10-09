#ifndef LINGODB_RUNTIME_GROWINGBUFFER_H
#define LINGODB_RUNTIME_GROWINGBUFFER_H
#include "Buffer.h"
#include "ExecutionContext.h"
#include "ThreadLocal.h"
namespace lingodb::runtime {

class GrowingBuffer;
class GrowingBufferAllocator {
   public:
   virtual GrowingBuffer* create(ExecutionContext* executionContext, size_t sizeOfType, size_t initialCapacity) = 0;
   static GrowingBufferAllocator* getDefaultAllocator();
   static GrowingBufferAllocator* getGroupAllocator(size_t groupId);
   virtual ~GrowingBufferAllocator() {}
};
class GrowingBuffer {
   runtime::FlexibleBuffer values;

   public:
   GrowingBuffer(size_t cap, size_t typeSize) : values(cap, typeSize) {}
   uint8_t* insert();
   static GrowingBuffer* create(GrowingBufferAllocator* allocator, size_t sizeOfType, size_t initialCapacity);
   static ThreadLocal* createThreadLocal(size_t sizeOfType);
   size_t getLen() const;
   size_t getTypeSize() const;
   runtime::Buffer sort(bool (*compareFn)(uint8_t*, uint8_t*));
   runtime::Buffer asContinuous();
   static void destroy(GrowingBuffer* vec);
   BufferIterator* createIterator();
   runtime::FlexibleBuffer& getValues() { return values; }
   static GrowingBuffer* merge(ThreadLocal* threadLocal);
};

} // end namespace lingodb::runtime
#endif //LINGODB_RUNTIME_GROWINGBUFFER_H
