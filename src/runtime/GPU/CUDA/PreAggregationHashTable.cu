#include "lingodb/runtime/GPU/CUDA/PreAggregationHashTable.cuh"
#include "lingodb/runtime/GPU/CUDA/Locking.cuh"
#include "lingodb/runtime/GPU/CUDA/helpers.cuh"
#include "lingodb/runtime/GPU/CUDA/CudaUtils.cuh"
#include <iostream>


namespace cudaRT{
__device__ PreAggregationHashtableFragment::PreAggregationHashtableFragment(int32_t typeSize, uint8_t* cachePtr, int32_t cacheSize) : typeSize(typeSize){
    // printf("INITIALIZING %p, typeSize=%d\n", this, typeSize);
    for(int i = 0; i < 1024; i++){
        cache[i] = nullptr;
    }
    // printf("INITIALIZED shared ptr\n");

    for (uint32_t i = 0; i < numPartitions; ++i) {
        new(&partitions[i]) FlexibleBuffer(typeSize); 
    }
}  

__device__ PreAggregationHashtableFragment::Entry* PreAggregationHashtableFragment::insertWarp(const uint64_t hash, const int32_t maskExternal) {
    const int32_t partitionID{static_cast<int32_t>(hash & partitionMask)};
    const int32_t mask= __match_any_sync(maskExternal, partitionID);
    const int32_t numWriters{__popc(mask)};
    const int32_t leader{__ffs(mask)-1};
    const int32_t lane = threadIdx.x % warpSize;
    FlexibleBuffer* myPartition = &partitions[partitionID];
    Entry* newEntry{nullptr};
    if(lane == leader){
        acquireLockBlock(myPartition->getLock());
        newEntry = reinterpret_cast<Entry*>(myPartition->insert(numWriters));
        relLockBlock(myPartition->getLock());
    }
    if(numWriters > 1){
        const int laneOffset = __popc(mask & ((1U << lane) - 1));
        uint8_t* bytes = reinterpret_cast<uint8_t*>(__shfl_sync(mask, (unsigned long long)newEntry, leader)); // barrier stalls
        newEntry = reinterpret_cast<Entry*>(&bytes[laneOffset * typeSize]);
    }
    // printf("[GPU][PreAggregationHashtableFragment::insertWarp, addr=%p][TID=%d][PartitionID=%d] my entry address %p!\n", this, threadIdx.x, partitionID, newEntry);
    newEntry->hashValue = hash;
    newEntry->next = nullptr;
    return newEntry;
}
__device__ PreAggregationHashtableFragment::Entry* PreAggregationHashtableFragment::insertWarpOpportunistic(const uint64_t hash) {
    return insertWarp(hash, __activemask());
}

static CUDA_HOST uint32_t getSMEMRequirements(){
    
}

__device__ void PreAggregationHashtableFragment::append(PreAggregationHashtableFragment* other){
    for(int i = 0; i < numPartitions; i++){
        FlexibleBuffer* flexBufPtr = other->getPartitionPtr(i);
        // printf("Partition %d, len = %d, typeSize=%d \n", i, flexBufPtr->getLen(), flexBufPtr->getTypeSize());
        if(flexBufPtr->getLen()){
            acquireLockBlock(partitions[i].getLock());
            partitions[i].merge(flexBufPtr);
            relLockBlock(partitions[i].getLock());
        }
        // printf("--Partition %d, len = %d, typeSize=%d \n", i, partitions[i].getLen(), partitions[i].getTypeSize());
    }
}

__device__ void PreAggregationHashtable::printMetaInfo(){
        printf("---------------------PreAggregationHashtable [%p]-------------------------\n", ht);
        int resCnt{0};
        for(int p = 0; p < PreAggregationHashtableFragment::numPartitions; p++){
            for(int i = 0; i < ht[p].hashMask+1; i++){
                Entry* curr = reinterpret_cast<Entry*>(untag(ht[p].ht[i]));
                if(!curr){continue;}
                // printf("[PARTITION %d, htEntryIdx=%d]", p, i);
                while(curr){
                    // printf(", {ptr=%p, next=%p, KEY1: %d, KEY2: %d, AGG: %lld}", curr, curr->next, curr->key[0], curr->key[1], curr->value);
                    curr = curr->next;
                    resCnt++;
                }
                // printf("\n");
            }
        }
        printf("Res count: %d, size=%d\n", resCnt, size);
        printf("------------------[END] PreAggregationHashtable [%p]----------------------\n", ht);
}


__global__ void linearizeOnDevice(PreAggregationHashtable* ht, uint8_t* continuousEntryAlloc, int32_t typeSize, int32_t* counter){
    for(int p = 0; p < PreAggregationHashtableFragment::numPartitions; p++){
        for(int i = 0; i < ht->ht[p].hashMask+1; i++){
            PreAggregationHashtable::Entry* curr = reinterpret_cast<PreAggregationHashtable::Entry*>(untag(ht->ht[p].ht[i]));
            while(curr){
                int myIdx = atomicAdd(counter, 1);
                memcpy(&continuousEntryAlloc[myIdx * typeSize], curr, typeSize);
                curr = curr->next;
            }
        }
    }
    // printf("COUNTER = %d\n", *counter);/usr/local/cuda-12.4
}
extern "C" uint8_t* getCPUIteratorHt(PreAggregationHashtable* ht, int32_t typeSize){
    // Step 1: Linearize GPU heap (can't copy from default CUDA's malloc)
    uint8_t* linearEntriesDevice;
    CHECK_CUDA_ERROR(cudaMalloc(&linearEntriesDevice, typeSize * ht->size));
    PreAggregationHashtable* htOnDevice;
    CHECK_CUDA_ERROR(cudaMalloc(&htOnDevice, sizeof(PreAggregationHashtable)));
    CHECK_CUDA_ERROR(cudaMemcpy(htOnDevice, ht, sizeof(PreAggregationHashtable), cudaMemcpyHostToDevice)); 
    int32_t cnt = 0;
    int32_t* d_cnt;
    CHECK_CUDA_ERROR(cudaMalloc(&d_cnt, sizeof(int32_t)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_cnt, &cnt, sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaGetLastError());
    std::cout << "linearizeOnDevice size = " << typeSize * ht->size << "\n";
    std::cout << "Num merges = " << ht->numMerges << "\n";

    linearizeOnDevice<<<1,1>>>(htOnDevice, linearEntriesDevice, typeSize, d_cnt);
    cudaStreamSynchronize(0);
    // Step 2: Get linear entries back and construct FlexibleBuffer
    uint8_t* linearEntriesHost;
    CHECK_CUDA_ERROR(cudaMallocHost(&linearEntriesHost, typeSize * ht->size));
    CHECK_CUDA_ERROR(cudaMemcpy(linearEntriesHost, linearEntriesDevice, typeSize * ht->size, cudaMemcpyDeviceToHost));
    uint8_t* linearEntriesPtrHost;
    CHECK_CUDA_ERROR(cudaMallocHost(&linearEntriesPtrHost, sizeof(uint8_t*) * ht->size));
    for(int i =0; i < ht->size; i++){
        reinterpret_cast<uint8_t**>(linearEntriesPtrHost)[i] = &linearEntriesHost[i * typeSize];
    }
    return reinterpret_cast<uint8_t*>(new OneBufferIterator({static_cast<int64_t>(ht->size * sizeof(uint8_t*)), linearEntriesPtrHost}));
}


extern "C" cudaRT::PreAggregationHashtable* initHtFromFrag(cudaRT::PreAggregationHashtableFragment* globalFragDevice, uint8_t* stream) {
    // Here we retrieve fragment class back to CPU and have its metadata that is used for allocations of the real HT.
    cudaRT::PreAggregationHashtableFragment* globalFragHost;
    CHECK_CUDA_ERROR(cudaMallocHost(&globalFragHost, sizeof(cudaRT::PreAggregationHashtableFragment)));
    CHECK_CUDA_ERROR(cudaMemcpy(globalFragHost, globalFragDevice, sizeof(cudaRT::PreAggregationHashtableFragment), cudaMemcpyDeviceToHost)); 

    cudaRT::PreAggregationHashtable::PartitionHt* preAllocatedPartitionsHost;
    CHECK_CUDA_ERROR(cudaMallocHost(&preAllocatedPartitionsHost, sizeof(cudaRT::PreAggregationHashtable::PartitionHt) * cudaRT::PreAggregationHashtableFragment::numPartitions));

    for(int partitionID = 0; partitionID < cudaRT::PreAggregationHashtableFragment::numPartitions; partitionID++){
        uint64_t partitionSize = globalFragHost->getPartitionPtr(partitionID)->getLen();
        auto [htAllocSize, htMask] = getHtSizeMask(partitionSize, sizeof(cudaRT::PreAggregationHashtableFragment::Entry*));
        preAllocatedPartitionsHost[partitionID].hashMask = htMask;
        CHECK_CUDA_ERROR(cudaMalloc(&preAllocatedPartitionsHost[partitionID].ht, htAllocSize));
        CHECK_CUDA_ERROR(cudaMemset(preAllocatedPartitionsHost[partitionID].ht, 0, htAllocSize));
        // std::cout << "partitionID=" << partitionID << ", htAllocSize=" << htAllocSize << ", htMask=" <<  htMask << "\n";
    }

    cudaRT::PreAggregationHashtable* preAggrHTDevice;
    CHECK_CUDA_ERROR(cudaMalloc(&preAggrHTDevice, sizeof(cudaRT::PreAggregationHashtable)));

    cudaRT::PreAggregationHashtable* preAggrHTHost;
    CHECK_CUDA_ERROR(cudaMallocHost(&preAggrHTHost, sizeof(cudaRT::PreAggregationHashtable)));

    new(preAggrHTHost) cudaRT::PreAggregationHashtable(preAllocatedPartitionsHost); // copies preAllocatedPartitionsDevice byte-by-byte
    CHECK_CUDA_ERROR(cudaMemcpy(preAggrHTDevice, preAggrHTHost, sizeof(cudaRT::PreAggregationHashtable), cudaMemcpyHostToDevice)); 
    cudaStreamSynchronize(0);

    // All payload is now on device, none is needed at host 
    CHECK_CUDA_ERROR(cudaFreeHost(globalFragHost));
    CHECK_CUDA_ERROR(cudaFreeHost(preAggrHTHost));
    CHECK_CUDA_ERROR(cudaFreeHost(preAllocatedPartitionsHost));

    return preAggrHTDevice;
}



}

