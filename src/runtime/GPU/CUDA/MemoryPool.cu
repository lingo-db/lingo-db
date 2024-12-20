#include "lingodb/runtime/GPU/CUDA/CudaMemPool.cuh"

#include <cstdint>
#include <iostream>

CUDAMemoryPool::CUDAMemoryPool(int32_t device_id, size_t retainBytesThreshold) : DeviceMemoryPool(device_id), retainBytesThreshold(retainBytesThreshold) {
   CHECK_CUDA_ERROR(cudaStreamCreate(&allocationStream));
   // Conveniently set reserved bytes to zero
   memset(&poolProps, 0, sizeof(poolProps));
   poolProps.allocType = cudaMemAllocationTypePinned;
   poolProps.handleTypes = cudaMemHandleTypePosixFileDescriptor;
   poolProps.location.type = cudaMemLocationTypeDevice;
   poolProps.location.id = device_id;
   // Maximum size of the pool. If exceeded (e.g., 2GB), throws OOM.
   // poolProps.maxSize = threshold;

   // Retain threshold. We can exceed it, but at least this amount will not be released back to OS.
   CHECK_CUDA_ERROR(cudaMemPoolCreate(&poolHandle, &poolProps));
   CHECK_CUDA_ERROR(cudaMemPoolSetAttribute(poolHandle, cudaMemPoolAttrReleaseThreshold, &retainBytesThreshold));
   CHECK_CUDA_ERROR(cudaDeviceSetMemPool(device_id, poolHandle));

   // Warmup: eagerly prepare the needed structures in runtime (if any)
   freeArrayPtr(getPtrForArray(retainBytesThreshold));
   constexpr uint64_t HEAP_SIZE{1024 * 1024 * 1024};
   cudaDeviceSetLimit(cudaLimitMallocHeapSize, HEAP_SIZE);
}

CUDAMemoryPool::~CUDAMemoryPool() {
   CHECK_CUDA_ERROR(cudaStreamSynchronize(allocationStream));
   CHECK_CUDA_ERROR(cudaMemPoolDestroy(poolHandle));
   CHECK_CUDA_ERROR(cudaStreamDestroy(allocationStream));
   if (!!threadStream.has_value()) {
      CHECK_CUDA_ERROR(cudaStreamDestroy(threadStream.value()));
   }
}

void CUDAMemoryPool::moveToDevice(uint8_t* hostPtr, uint8_t* devicePtr, int64_t numBytes, bool sync) {
   if (!threadStream.has_value()) {
      threadStream = cudaStream_t{};
      CHECK_CUDA_ERROR(cudaStreamCreate(&threadStream.value()));
   }
   // std::cout <<"[TO] Copy " << numBytes << " bytes from host at " << (void*) hostPtr << " to device at " << (void*) devicePtr << "\n";

   CHECK_CUDA_ERROR(cudaMemcpyAsync(devicePtr, hostPtr, numBytes, cudaMemcpyHostToDevice, threadStream.value()));
   if (sync) {
      syncThreadStream();
   }
}

void CUDAMemoryPool::registerHostMemAsPinned(uint8_t* hostPtr, int64_t numBytes) {
   // cudaHostRegisterReadOnly
   CHECK_CUDA_ERROR(cudaHostRegister(hostPtr, numBytes, cudaHostRegisterDefault));
}

void CUDAMemoryPool::moveFromDevice(uint8_t* hostPtr, uint8_t* devicePtr, int64_t numBytes) {
   if (!threadStream.has_value()) {
      threadStream = cudaStream_t{};
      CHECK_CUDA_ERROR(cudaStreamCreate(&threadStream.value()));
   }
   auto it = cudaGetLastError();
   CHECK_CUDA_ERROR(it);
   // std::cout <<"[FROM] Copy " << numBytes << " bytes to host at " << (void*) hostPtr << " from device at " << (void*) devicePtr << "\n";
   CHECK_CUDA_ERROR(cudaMemcpyAsync(hostPtr, devicePtr, numBytes, cudaMemcpyDeviceToHost, threadStream.value()));
   syncThreadStream();
}

void CUDAMemoryPool::syncThreadStream(uint8_t* streamPtr) {
   if (streamPtr) {
      CHECK_CUDA_ERROR(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(streamPtr)));
   } else {
      if (!threadStream.has_value()) {
         threadStream = cudaStream_t{};
         CHECK_CUDA_ERROR(cudaStreamCreate(&threadStream.value()));
      }
      CHECK_CUDA_ERROR(cudaStreamSynchronize(threadStream.value()));
   }
}

uint8_t* CUDAMemoryPool::getThreadStream() {
   if (!threadStream.has_value()) {
      threadStream = cudaStream_t{};
      CHECK_CUDA_ERROR(cudaStreamCreate(&threadStream.value()));
   }
   return reinterpret_cast<uint8_t*>(threadStream.value());
}

void CUDAMemoryPool::setThreadStream(uint8_t* stream) {
   if (!!threadStream.has_value()) {
      syncThreadStream();
      CHECK_CUDA_ERROR(cudaStreamDestroy(threadStream.value()));
   }
   threadStream.value() = reinterpret_cast<cudaStream_t>(stream);
}

uint8_t* CUDAMemoryPool::getThreadStreamEvent() {
   if (!threadStream.has_value()) {
      threadStream = cudaStream_t{};
      CHECK_CUDA_ERROR(cudaStreamCreate(&threadStream.value()));
   }
   cudaEvent_t event;
   CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&event, cudaEventBlockingSync));
   return reinterpret_cast<uint8_t*>(event);
}

uint8_t* CUDAMemoryPool::getPtrForArray(const size_t arraySize) {
   // printPoolState(std::string("Allocating ").append(std::to_string(arraySize/MB)).append("MB"));
   // eventToken event;
   // CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&event, cudaEventBlockingSync));
   uint8_t* devicePtr{nullptr};
   // std::cout << cudaGetLastError() << " : " << cudaGetErrorString(cudaGetLastError() ) << "\n";
   while (cudaGetLastError() != cudaSuccess) {}
   CHECK_CUDA_ERROR_ALLOC(cudaMallocAsync(&devicePtr, arraySize, allocationStream));
   // CHECK_CUDA_ERROR(cudaEventRecord(event, hStream));
   // std::cout <<"[Alloc] " << arraySize << " bytes at " << (void*) devicePtr << "\n";

   return devicePtr;
}

void CUDAMemoryPool::freeArrayPtr(uint8_t* devicePtr, bool possiblyReleaseToOS) {
   // printPoolState(std::string("freeArray: ").append(std::to_string(reinterpret_cast<uintptr_t>(devicePtr))).append(", release=").append(std::to_string(possiblyReleaseToOS)));
   CHECK_CUDA_ERROR(cudaFreeAsync(reinterpret_cast<void*>(devicePtr), allocationStream));
   if (possiblyReleaseToOS) {
      // Above we released memory to the pool, but not to OS, so memory is still visible as allocated by the driver/OS.
      // It will be released to OS only on the synchronization point.
      CHECK_CUDA_ERROR(cudaMemPoolTrimTo(poolHandle, 0));
      CHECK_CUDA_ERROR(cudaStreamSynchronize(allocationStream));
   }
   // printPoolState(std::string("After freeArray: ").append(std::to_string(reinterpret_cast<uintptr_t>(devicePtr))).append(", release=").append(std::to_string(possiblyReleaseToOS)));
}

uint32_t CUDAMemoryPool::getAvailableSMEMSize(const uint32_t deviceId) {
   uint32_t smemPerBlock{0};
   CHECK_CUDA_ERROR(cudaDeviceGetAttribute(reinterpret_cast<int*>(&smemPerBlock), cudaDevAttrMaxSharedMemoryPerBlock, deviceId));
   return smemPerBlock;
}
uint32_t CUDAMemoryPool::getSMCount(const uint32_t deviceId) {
   uint32_t smCount{0};
   CHECK_CUDA_ERROR(cudaDeviceGetAttribute(reinterpret_cast<int*>(&smCount), cudaDevAttrMultiProcessorCount, deviceId));
   return smCount;
}
