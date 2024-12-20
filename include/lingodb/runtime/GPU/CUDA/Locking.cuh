#ifndef CUDA_RUNTIME_LOCKING_H
#define CUDA_RUNTIME_LOCKING_H
#include <cuda_runtime.h>

/*
 We need `__threadfence` to flush any "cached" modifications of non-atomic variables, 
 because non-atomics have no visibility guarantees.
 Example: we lock a data structure, modify member, unlock -> 
    member modification could be only visible at L1, other SMs do not see it.
*/
__device__ __forceinline__ void acquireLock(int* lock) {
    while (atomicCAS(lock, 0, 1) != 0);
    __threadfence(); // flush to L2 (globally visible)
}
__device__ __forceinline__ void relLock(int* lock){
    __threadfence();
    atomicExch(lock, 0);
}
__device__ __forceinline__ void acquireLockBlock(int* lock){
    while (atomicCAS(lock, 0, 1) != 0);
    __threadfence_block(); // flush to L1 (SM visible)
}
__device__ __forceinline__ void relLockBlock(int* lock) {
    __threadfence_block();
    atomicExch(lock, 0); 
}

#endif //CUDA_RUNTIME_LOCKING_H
