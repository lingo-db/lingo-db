#ifndef LINGODB_RUNTIME_GPU_DEVICEMEMORYMANAGER_H
#define LINGODB_RUNTIME_GPU_DEVICEMEMORYMANAGER_H

#include "lingodb/runtime/GPU/MemoryPool.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace lingodb::runtime {

using stateID = uint8_t*;

class DeviceMemoryManager {
   using stateTrackerType = std::unordered_map<stateID, std::pair<DeviceStateStatus, std::list<stateID>::iterator>>;

   std::unique_ptr<DeviceMemoryPool> memoryPool;
   stateTrackerType statesOnDevice;
   std::list<stateID> fifoCache;
   std::mutex cacheMtx;
   std::mutex evictionMtx;
   std::condition_variable cv;

   public:
   DeviceMemoryManager() : memoryPool(nullptr) {}
   DeviceMemoryManager(std::unique_ptr<DeviceMemoryPool> pool) : memoryPool(std::move(pool)) {}

   ~DeviceMemoryManager() {
      for (const auto& [state, status] : statesOnDevice) {
         freeArray(status.first.devicePtr);
      }
   }

   uint8_t* fetchArray(uint8_t* hostPtr, const uint32_t arraySize);
   void fetchArrayFromGPU(uint8_t* hostPtr, const uint32_t arraySize);
   void moveToDevice(uint8_t* hostPtr, uint8_t* devicePtr, int64_t numBytes, bool sync);

   void moveFromDevice(uint8_t* hostPtr, uint8_t* devicePtr, int64_t numBytes);

   uint8_t* getThreadStream();

   uint8_t* getThreadStreamEvent();

   void setThreadStream(uint8_t* streamPtr);

   void syncStream(uint8_t* streamPtr = nullptr);

   void registerHostMemAsPinned(uint8_t* hostPtr, int64_t numBytes);

   uint8_t* getPtrForArray(const uint32_t arraySize);

   uint8_t** getPtrsForNArrays(const uint32_t* arraySizes, const size_t lengthSizes);

   void freeArray(uint8_t* devicePtr, bool possiblyRelease = false);

   uint32_t getAvailableSMEMSize(const uint32_t deviceId = 0);

   uint32_t getSMCount(const uint32_t deviceId = 0);

   DeviceArrayToken getPtrForStateArray(stateID id, const uint32_t arraySize);

   DeviceArrayToken* getPtrsForStateArrays(const stateID* ids, const uint32_t* arraySizes, const size_t length);

   void freeStateArray(stateID id, bool possiblyRelease = false);

   void freeStateArrayByIter(stateTrackerType::iterator it, bool possiblyRelease = false);

   void unpinState(stateID id);

   void unpinStates(stateID* ids, const size_t lengthIds);

   void unpinAllStates();

   void evict(const uint64_t evictForSize);

   void printCache() {
      std::unique_lock<std::mutex> guard(cacheMtx);
      std::cout << "Map :\n";
      for (const auto& [stateId, d] : statesOnDevice) {
         std::cout << "id = " << stateId << ", ptr=" << (void*) d.first.devicePtr << ", pinned=" << d.first.pinned << "\n";
      }
      std::cout << "_____________\n";
   }
};

} // namespace lingodb::runtime

#endif //LINGODB_RUNTIME_GPU_DEVICEMEMORYMANAGER_H
