#ifndef LINGODB_RUNTIME_GROWINGBUFFER_H
#define LINGODB_RUNTIME_GROWINGBUFFER_H
#include "ExecutionContext.h"
#include "ThreadLocal.h"
#include "lingodb/runtime/Buffer.h"
namespace lingodb::runtime {

class GrowingBuffer;
class GrowingBufferAllocator {
   public:
   virtual GrowingBuffer* create(ExecutionContext* executionContext,size_t sizeOfType, size_t initialCapacity) = 0;
   static GrowingBufferAllocator* getDefaultAllocator();
   static GrowingBufferAllocator* getGroupAllocator(ExecutionContext* executionContext, size_t groupId);
   virtual ~GrowingBufferAllocator(){}
};
class GrowingBuffer {
   runtime::FlexibleBuffer values;

   public:
   GrowingBuffer(size_t cap, size_t typeSize) : values(cap, typeSize) {}
   uint8_t* insert();
   static GrowingBuffer* create(GrowingBufferAllocator* allocator, ExecutionContext* executionContext, size_t sizeOfType, size_t initialCapacity);
   size_t getLen() const;
   size_t getTypeSize() const;
   runtime::Buffer sort(runtime::ExecutionContext*, bool (*compareFn)(uint8_t*, uint8_t*));
   runtime::Buffer asContinuous(ExecutionContext* executionContext);
   static void destroy(GrowingBuffer* vec);
   BufferIterator* createIterator();
   runtime::FlexibleBuffer& getValues() { return values; }
   static GrowingBuffer* merge(ThreadLocal* threadLocal);
};

} // end namespace lingodb::runtime
#endif //LINGODB_RUNTIME_GROWINGBUFFER_H
