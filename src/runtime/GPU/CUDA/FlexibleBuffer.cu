#include "lingodb/runtime/GPU/CUDA/FlexibleBuffer.cuh"
#include "lingodb/runtime/GPU/CUDA/Locking.cuh"
#include "lingodb/runtime/GPU/CUDA/DeviceHeap.cuh"

namespace cudaRT{

__device__ uint8_t* FlexibleBuffer::allocCurrentCapacity() {
    return reinterpret_cast<uint8_t*>(cudaRT::heapAlloc(currCapacity * typeSize));
}
__device__ FlexibleBuffer::FlexibleBuffer(int32_t initialCapacity, int32_t typeSize, bool initialAlloc) : currCapacity(initialCapacity), typeSize(typeSize) {
    // if (initialAlloc) {
    //     uint8_t* ptr = allocCurrentCapacity();
    //     buffers.push_back(Buffer{ptr, 0});
    // }
}
__device__ void FlexibleBuffer::destroy() { // Support at most one thread block!
    for (int32_t i = threadIdx.x; i < buffers.size(); i+=blockDim.x) {
        if (buffers[i].ptr) {
            freeHeapPtr(buffers[i].ptr);
        }
    }
    __syncthreads(); // "Global" sync
    if(threadIdx.x == 0){
        buffers.destroy();
    }
}
__device__ uint8_t* FlexibleBuffer::insertWarpOpportunistic(){
    const int mask{static_cast<int>(__activemask())};
    const int numWriters{__popc(mask)};
    const int leader{__ffs(mask)-1};
    uint8_t* res{nullptr};
    const int lane{static_cast<int>(threadIdx.x % warpSize)};
    if(lane == leader){
        // A warp can diverge, so sub-warps will share the warp-level data structure.
        acquireLockBlock(getLock());
        res = insert(numWriters);
        // printf("[GPU][FlexibleBuffer::insertWarpOpportunistic, addr=%p][LEADER BID=%d, TID=%d, WID=%d] NumWriters: %d, Len after insert: %d!\n", this, blockIdx.x, threadIdx.x, threadIdx.x/32, numWriters, getLen());
        relLockBlock(getLock());
    }
    
    if(numWriters > 1){
        res = reinterpret_cast<uint8_t*>(__shfl_sync(mask, (unsigned long long)res, leader));
        const int laneOffset = __popc(mask & ((1U << lane) - 1));
        res = &res[laneOffset * typeSize];
    }
    // printf("[GPU][FlexibleBuffer::insertWarpOpportunistic, addr=%p][TID=%d] my entry address %p!\n", this, threadIdx.x, res);
    return res;
}


}
