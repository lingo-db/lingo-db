#ifndef LINGODB_RUNTIME_GPU_CUDA_PREAGGREGATIONHASHTABLE_CUH
#define LINGODB_RUNTIME_GPU_CUDA_PREAGGREGATIONHASHTABLE_CUH

#include "DefinitionHelper.cuh"
#include "FlexibleBuffer.cuh"

#include <cstring>
#include <new>
#include <stdio.h>
namespace cudaRT {

class PreAggregationHashtableFragment {
   public:
   static constexpr uint32_t partitionShift{6}; //2^6=64
   static constexpr uint32_t numPartitions{1 << partitionShift};
   static constexpr uint64_t partitionMask{numPartitions - 1};
   static constexpr int32_t initCapacity{256};

   struct Entry {
      Entry* next;
      uint64_t hashValue;
      uint8_t content[];
      //kv follows
   };

   private:
   Entry* cache[1024]; // first member, since we load its entries by class ptr (e.g., lookupOrInsert)
   FlexibleBuffer partitions[numPartitions];
   int32_t cacheMask{1023};
   int32_t typeSize{0};

   public:
   CUDA_HOST_DEVICE PreAggregationHashtableFragment() {}
   CUDA_HOST_DEVICE PreAggregationHashtableFragment(int32_t typeSize) : typeSize(typeSize) {
      for (uint32_t i = 0; i < numPartitions; ++i) {
         new (&partitions[i]) FlexibleBuffer(typeSize);
      }
   }
   CUDA_DEVICE PreAggregationHashtableFragment(int32_t typeSize, uint8_t* cachePtr, int32_t cacheSize);
   CUDA_HOST_DEVICE ~PreAggregationHashtableFragment() {}
   CUDA_HOST_DEVICE FlexibleBuffer* getPartitionPtr(int32_t partitionID) { return &partitions[partitionID]; }
   static CUDA_HOST uint32_t getSMEMRequirements();
   CUDA_DEVICE Entry* insertWarp(const uint64_t hash, const int32_t maskExternal);
   CUDA_DEVICE Entry* insertWarpOpportunistic(const uint64_t hash);
   CUDA_DEVICE void append(PreAggregationHashtableFragment* other);
   // CUDA_DEVICE Entry** getCachePtr() const {return cache;};
   CUDA_DEVICE int32_t getCacheMask() const { return cacheMask; };
   CUDA_DEVICE void printMetaInfo(void (*printEntry)(uint8_t*) = nullptr) {
      printf("--------------------PreAggregationHashtableFragmentSMEM [%p]--------------------\n", this);
      int32_t countValidPartitions{0};
      for (uint32_t i = 0; i < numPartitions; i++) {
         countValidPartitions += (partitions[i].getTypeSize() != 0);
      }
      auto entryPrinter = [](uint8_t* entry) {
         Entry* e = reinterpret_cast<Entry*>(entry);
         printf("{entryPtr=%p, next=%p, hashValue=%lu}", e, e->next, e->hashValue);
      };
      printf("typeSize=%d, %d non-empty partitions out of %d\n", typeSize, countValidPartitions, numPartitions);
      for (size_t i = 0; i < numPartitions; i++) {
         if (partitions[i].getLen()) {
            printf("[Partition %lu] ", i);
            partitions[i].printMetaInfo(entryPrinter);
            printf("[END Partition %lu] \n", i);
         }
      }
      printf("---------------[END] PreAggregationHashtableFragmentSMEM [%p]--------------------\n", this);
   }
};

class PreAggregationHashtable {
   public:
   using Entry = PreAggregationHashtableFragment::Entry;
   struct PartitionHt {
      Entry** ht;
      uint64_t hashMask;
   };
   PartitionHt ht[PreAggregationHashtableFragment::numPartitions];
   FlexibleBuffer buffer;
   int32_t size{0};
   int32_t numMerges{0}; // debug member
   CUDA_HOST_DEVICE PreAggregationHashtable() : ht(), buffer(sizeof(PreAggregationHashtableFragment::Entry*)) {}
   CUDA_HOST PreAggregationHashtable(PartitionHt* preAllocatedPartitions) : buffer(sizeof(PreAggregationHashtableFragment::Entry*)) {
      memcpy(ht, preAllocatedPartitions, sizeof(PartitionHt) * PreAggregationHashtableFragment::numPartitions);
   }

   public:
   CUDA_DEVICE void printMetaInfo();
};

extern "C" cudaRT::PreAggregationHashtable* initHtFromFrag(PreAggregationHashtableFragment* frag, uint8_t* stream);
extern "C" uint8_t* getCPUIteratorHt(PreAggregationHashtable*, int32_t typeSize);

} // namespace cudaRT
#endif //LINGODB_RUNTIME_GPU_CUDA_PREAGGREGATIONHASHTABLE_CUH
