#include "lingodb/runtime/GPU/CUDA/LazyJoinHashtable.cuh"
#include "lingodb/runtime/GPU/CUDA/helpers.cuh"
#include "lingodb/runtime/GPU/CUDA/CudaUtils.cuh"
#include <iostream>
namespace cudaRT{
   /*
      To expose CUDA code to C++ (e.g., in form of a kernel call wrapper),
      we have to declare the wrapper as `extern "C"`.
   */
   __device__ void HashIndexedView::printMetaInfo(){
      printf("--------------------HashIndexedView [%p]--------------------\n", this);
      int32_t totalEntries{0};
      int32_t chainedEntries{0};
      printf("htMask= %lu, ht**=%p\n", htMask, ht);
      for(int32_t i = 0; i < htMask+1; i++){
         if(ht[i]){
            auto htEntry = untag(ht[i]); // tag stores a bloom filter in the upper bits, can't dereference such a pointer
            printf("[HT SLOT %d]: ht[i] = %p : {next = %p, hash = %lu}\n", i, htEntry, untag(htEntry->next), htEntry->hashValue);
            Entry* cur = untag(htEntry->next);
            totalEntries++;
            while(cur){
               printf("[HT SLOT %d chain]: current %p : {next = %p, hash = %lu}\n", i, cur, untag(cur->next), cur->hashValue);
               cur = untag(cur->next);
               chainedEntries++;
               totalEntries++;
            }
         } else{
            printf("[HT SLOT %d]: %p \n", i, ht[i]);
         }
      }
      printf("Total valid entries: %d, out of them %d are not directly accessible via ht directory\n", totalEntries, chainedEntries);
      printf("-----------------------------------------------------------\n");
   }

   __global__ void build(GrowingBuffer* buffer, HashIndexedView* view) {
      const int globalTID = blockDim.x * blockIdx.x + threadIdx.x;
      const int numThreadsTotal = blockDim.x * gridDim.x;
      const uint64_t globalMask{view->htMask}; 
      HashIndexedView::Entry** globalHt{view->ht};
      FlexibleBufferIterator myIterator(buffer->getValuesPtr(), globalTID, numThreadsTotal);
      HashIndexedView::Entry* entry = (HashIndexedView::Entry*)myIterator.initialize();
      while(entry){
         uint64_t hash = (uint64_t) entry->hashValue;
         const uint64_t pos = hash & globalMask;
         HashIndexedView::Entry* newEntry;
         HashIndexedView::Entry* current;
         HashIndexedView::Entry* exchanged;
         // Try to put entry to the head of the linked list at globalHt[pos]. 
         // Success iff while preparing the entry to point to current head, the current head didn't change.
         do {
            current = globalHt[pos]; // read head (TODO: do we need to read globalHt[pos] or can we reuse exchanged?)
            entry->next = untag(current); // entry point to head
            newEntry = tag(entry, current, hash); // update bloom filter aggregate
            exchanged = (HashIndexedView::Entry*) atomicCAS((unsigned long long*)&globalHt[pos], (unsigned long long)current, (unsigned long long)newEntry);
         } while (exchanged != current); // retry if the head changed, otherwise the atomic write happened -> proceed to next entry.
         entry = (HashIndexedView::Entry*)myIterator.step();
      }
      // if(!threadIdx.x){view->printMetaInfo();}  // only for <<<1,X>>> debug
   }

   extern "C" HashIndexedView* buildHashIndexedView(GrowingBuffer* buffer, uint8_t* stream) {
      // Get a shallow copy of growing buffer back to CPU (need just the length)
      GrowingBuffer* bufCPU = reinterpret_cast<GrowingBuffer*>(malloc(sizeof(GrowingBuffer)));
      HashIndexedView* viewCPU = reinterpret_cast<HashIndexedView*>(malloc(sizeof(HashIndexedView)));
      CHECK_CUDA_ERROR(cudaMemcpy(bufCPU, buffer, sizeof(GrowingBuffer), cudaMemcpyDeviceToHost)); // shallow copy, only some metadata

      // Handle hashtable creation, allocate straight on GPU.
      auto [htAllocSize, htMask] = getHtSizeMask(bufCPU->getValuesPtr()->getLen(), sizeof(uint8_t*));
      std::cout << "Filter out: " << bufCPU->getValuesPtr()->getLen() << "\n";
      uint8_t* htPtrGPU{nullptr};
      CHECK_CUDA_ERROR(cudaMalloc(&htPtrGPU, htAllocSize));
      CHECK_CUDA_ERROR(cudaMemset(htPtrGPU, 0, htAllocSize));
      new(viewCPU) HashIndexedView(htPtrGPU, htMask);

      // Get HashIndexedView class on GPU 
      HashIndexedView* viewGPU{nullptr};
      CHECK_CUDA_ERROR(cudaMalloc(&viewGPU, sizeof(HashIndexedView)));
      CHECK_CUDA_ERROR(cudaMemcpy(viewGPU, viewCPU, sizeof(HashIndexedView), cudaMemcpyHostToDevice));
       
      build<<<30, 256>>>(buffer, viewGPU);

      cudaStreamSynchronize(0);
      cudaGetLastError();
      return viewGPU;
   }
}
