#ifndef LINGODB_RUNTIME_GPU_DEVICERTWRAPPER_H
#define LINGODB_RUNTIME_GPU_DEVICERTWRAPPER_H

#include "lingodb/runtime/ExecutionContext.h"

#include "DeviceMemoryManager.h"

namespace cudaRT {
class HashIndexedView;
class GrowingBuffer;
class PreAggregationHashtableFragment;
class PreAggregationHashtable;
} // end namespace cudaRT
namespace lingodb::runtime {
struct DeviceMemoryFuncs {
   static uint8_t* getPtrForArray(const uint32_t arraySize);
   static void freePtrForArray(uint8_t* arrayPtr);
   static void syncStream(uint8_t* streamPtr);

   static uint8_t* getThreadStream();
   static void setThreadStream(uint8_t* streamPtr);
   static uint8_t* getThreadStreamEvent();
   static uint8_t* threadFetchArrayToGPU(uint8_t* hostPtr, size_t numBytes);
   static void threadSendToGPUSync(uint8_t* hostPtr, uint8_t* devicePtr, size_t numBytes);
   static void threadCopyFromGPUSync(uint8_t* hostPtr, uint8_t* devicePtr, size_t numBytes);

   static void threadFetchArrayFromGPU(uint8_t* hostPtr, size_t numBytes);

   static void initializeGrowingBufferOnCPU(cudaRT::GrowingBuffer* buf, int32_t capacity, int32_t typeSize);
   static void initializePreAggrFragmentOnCPU(cudaRT::PreAggregationHashtableFragment* frag, int32_t typeSize);
   static cudaRT::HashIndexedView* createHashIndexedView(cudaRT::GrowingBuffer* buf);
   static cudaRT::PreAggregationHashtable* initHtFromFragMetadata(cudaRT::PreAggregationHashtableFragment* frag);
   static uint8_t* createHTiteratorOnCPU(cudaRT::PreAggregationHashtable* buf, int32_t typeSize);
};
} // namespace lingodb::runtime

#endif //LINGODB_RUNTIME_GPU_DEVICERTWRAPPER_H
