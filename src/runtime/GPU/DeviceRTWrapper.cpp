#include "lingodb/runtime/GPU/DeviceRTWrapper.h"
#if GPU_ENABLED == 1
#include "lingodb/runtime/GPU/CUDA/GrowingBuffer.cuh"
#include "lingodb/runtime/GPU/CUDA/LazyJoinHashtable.cuh"
#include "lingodb/runtime/GPU/CUDA/PreAggregationHashTable.cuh"
#endif
#include <iostream>
namespace runtime = lingodb::runtime;

uint8_t* runtime::DeviceMemoryFuncs::getPtrForArray(const uint32_t arraySize) {
   auto* executionContext = runtime::getCurrentExecutionContext();
   // std::cout << "[DeviceMemoryFuncs::getPtrForArray()]ExecCTX_ptr = " << (void*) executionContext << " " << arraySize << "\n";
   assert(executionContext);
   return executionContext->getGPUMemManager().getPtrForArray(arraySize);
}

void runtime::DeviceMemoryFuncs::freePtrForArray(uint8_t* arrayPtr) {
   auto* executionContext = runtime::getCurrentExecutionContext();
   // std::cout << "[DeviceMemoryFuncs::getPtrForArray()]ExecCTX_ptr = " << (void*) executionContext << " " << arraySize << "\n";
   assert(executionContext);
   return executionContext->getGPUMemManager().freeArray(arrayPtr);
}

uint8_t* runtime::DeviceMemoryFuncs::getThreadStream() {
   auto* executionContext = runtime::getCurrentExecutionContext();
   assert(executionContext);
   return executionContext->getGPUMemManager().getThreadStream();
}

void runtime::DeviceMemoryFuncs::syncStream(uint8_t* streamPtr) {
   auto* executionContext = runtime::getCurrentExecutionContext();
   executionContext->getGPUMemManager().syncStream(streamPtr);
}

void runtime::DeviceMemoryFuncs::setThreadStream(uint8_t* streamPtr) {
   auto* executionContext = runtime::getCurrentExecutionContext();
   assert(executionContext);
   return executionContext->getGPUMemManager().setThreadStream(streamPtr);
}

uint8_t* runtime::DeviceMemoryFuncs::getThreadStreamEvent() {
   auto* executionContext = runtime::getCurrentExecutionContext();
   assert(executionContext);
   return executionContext->getGPUMemManager().getThreadStreamEvent();
}

uint8_t* runtime::DeviceMemoryFuncs::threadFetchArrayToGPU(uint8_t* hostPtr, size_t numBytes) {
   auto* executionContext = runtime::getCurrentExecutionContext();
   assert(executionContext);
   return executionContext->getGPUMemManager().fetchArray(hostPtr, numBytes);
}

void runtime::DeviceMemoryFuncs::threadSendToGPUSync(uint8_t* hostPtr, uint8_t* devicePtr, size_t numBytes) {
   auto* executionContext = runtime::getCurrentExecutionContext();
   assert(executionContext);
   executionContext->getGPUMemManager().moveToDevice(hostPtr, devicePtr, numBytes, true);
}

void runtime::DeviceMemoryFuncs::threadCopyFromGPUSync(uint8_t* hostPtr, uint8_t* devicePtr, size_t numBytes) {
   auto* executionContext = runtime::getCurrentExecutionContext();
   assert(executionContext);
   executionContext->getGPUMemManager().moveFromDevice(hostPtr, devicePtr, numBytes);
}

void runtime::DeviceMemoryFuncs::threadFetchArrayFromGPU(uint8_t* hostPtr, size_t numBytes) {
   auto* executionContext = runtime::getCurrentExecutionContext();
   assert(executionContext);
   return executionContext->getGPUMemManager().fetchArrayFromGPU(hostPtr, numBytes);
}

#if GPU_ENABLED == 1
void runtime::DeviceMemoryFuncs::initializeGrowingBufferOnCPU(cudaRT::GrowingBuffer* buf, int32_t initCapacity, int32_t typeSize) {
   new (buf) cudaRT::GrowingBuffer(initCapacity, typeSize, false);
}

void runtime::DeviceMemoryFuncs::initializePreAggrFragmentOnCPU(cudaRT::PreAggregationHashtableFragment* buf, int32_t typeSize) {
   // std::cout << "[HOST] initializePreAggrFragmentOnCPU = " << typeSize << "\n";
   new (buf) cudaRT::PreAggregationHashtableFragment(typeSize);
}

cudaRT::HashIndexedView* runtime::DeviceMemoryFuncs::createHashIndexedView(cudaRT::GrowingBuffer* buf) {
   auto* executionContext = runtime::getCurrentExecutionContext();
   assert(executionContext);
   uint8_t* stream = executionContext->getGPUMemManager().getThreadStream();
   return cudaRT::buildHashIndexedView(buf, stream);
}

cudaRT::PreAggregationHashtable* runtime::DeviceMemoryFuncs::initHtFromFragMetadata(cudaRT::PreAggregationHashtableFragment* frag) {
   auto* executionContext = runtime::getCurrentExecutionContext();
   assert(executionContext);
   uint8_t* stream = executionContext->getGPUMemManager().getThreadStream();
   return cudaRT::initHtFromFrag(frag, stream);
}

uint8_t* runtime::DeviceMemoryFuncs::createHTiteratorOnCPU(cudaRT::PreAggregationHashtable* buf, int32_t typeSize) {
   // std::cout << "createHTiteratorOnCPU typeSize=" << typeSize << "\n";
   return getCPUIteratorHt(buf, typeSize);
}
#else
void runtime::DeviceMemoryFuncs::initializeGrowingBufferOnCPU(cudaRT::GrowingBuffer* buf, int32_t initCapacity, int32_t typeSize) {
   assert(false);
}

void runtime::DeviceMemoryFuncs::initializePreAggrFragmentOnCPU(cudaRT::PreAggregationHashtableFragment* buf, int32_t typeSize) {
   assert(false);
}

cudaRT::HashIndexedView* runtime::DeviceMemoryFuncs::createHashIndexedView(cudaRT::GrowingBuffer* buf) {
   assert(false);
   return nullptr;
}

cudaRT::PreAggregationHashtable* runtime::DeviceMemoryFuncs::initHtFromFragMetadata(cudaRT::PreAggregationHashtableFragment* frag) {
   assert(false);
   return nullptr;
}

uint8_t* runtime::DeviceMemoryFuncs::createHTiteratorOnCPU(cudaRT::PreAggregationHashtable* buf, int32_t typeSize) {
   assert(false);
   return nullptr;
}

#endif
