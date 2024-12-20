#ifndef CUDA_MEMPOOL_H
#define CUDA_MEMPOOL_H

#include <cuda_runtime.h>

#include "lingodb/runtime/GPU/MemoryPool.h"
#include "lingodb/runtime/GPU/CUDA/CudaUtils.cuh"

#include<optional>
#include<iostream>


class CUDAMemoryPool : public DeviceMemoryPool {
    cudaMemPoolProps poolProps;
    cudaMemPool_t poolHandle;
    cudaStream_t allocationStream;
    uint64_t retainBytesThreshold;
    //tbb::enumerable_thread_specific<cudaStream_t> threadStream;
   std::optional<cudaStream_t> threadStream;

    cudaMemPool_t getPoolHandle() const { return poolHandle; }

    void printPoolState(std::string_view prefix = ""){
        uint64_t nbytes_used, nbytes_reserved;
        CHECK_CUDA_ERROR(cudaMemPoolGetAttribute(poolHandle, cudaMemPoolAttrUsedMemCurrent, &nbytes_used));
        CHECK_CUDA_ERROR(cudaMemPoolGetAttribute(poolHandle, cudaMemPoolAttrReservedMemCurrent, &nbytes_reserved));
        std::cout << "["<< prefix << "] Threshold=" << retainBytesThreshold/mb << "MB < nbytes_reserved=" << nbytes_reserved/mb << "MB, nbytes_used = " << nbytes_used/mb << "MB\n";
    }

    public:
    CUDAMemoryPool(int32_t device_id, size_t retainBytesThreshold = 0);
    virtual ~CUDAMemoryPool();
    virtual uint8_t* getPtrForArray(const size_t arraySize);
    virtual void freeArrayPtr(uint8_t* devicePtr, bool possiblyReleaseToOS = false);
    virtual void moveToDevice(uint8_t* hostPtr, uint8_t* devicePtr, int64_t numBytes, bool sync);
    virtual void moveFromDevice(uint8_t* hostPtr, uint8_t* devicePtr, int64_t numBytes);
    virtual void syncThreadStream(uint8_t* streamPtr = nullptr);
    virtual uint8_t* getThreadStream();
    virtual void setThreadStream(uint8_t* stream);
    virtual uint8_t* getThreadStreamEvent();
    virtual void registerHostMemAsPinned(uint8_t* hostPtr, int64_t numBytes);
    virtual uint32_t getAvailableSMEMSize(const uint32_t deviceId);
    virtual uint32_t getSMCount(const uint32_t deviceId);
};


#endif // CUDA_MEMPOOL_H
