#include "lingodb/runtime/GPU/DeviceMemoryManager.h"

#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
namespace runtime = lingodb::runtime;

uint8_t* runtime::DeviceMemoryManager::fetchArray(uint8_t* hostPtr, const uint32_t arraySize) {
   DeviceArrayToken deviceArrayStatus = getPtrForStateArray(hostPtr, arraySize);
   if (!deviceArrayStatus.cached) {
      memoryPool->moveToDevice(hostPtr, deviceArrayStatus.devicePtr, arraySize, true);
   } else {
      // std::cout << "Host addr " << (void*)hostPtr << " is cached \n";
   }
   return deviceArrayStatus.devicePtr;
}

void runtime::DeviceMemoryManager::fetchArrayFromGPU(uint8_t* hostPtr, const uint32_t arraySize) {
   DeviceArrayToken deviceArrayStatus = getPtrForStateArray(hostPtr, arraySize);
   assert(deviceArrayStatus.cached);
   memoryPool->moveFromDevice(hostPtr, deviceArrayStatus.devicePtr, arraySize);
   memoryPool->syncThreadStream();
}

void runtime::DeviceMemoryManager::moveToDevice(uint8_t* hostPtr, uint8_t* devicePtr, int64_t numBytes, bool sync) {
   memoryPool->moveToDevice(hostPtr, devicePtr, numBytes, sync);
}

void runtime::DeviceMemoryManager::moveFromDevice(uint8_t* hostPtr, uint8_t* devicePtr, int64_t numBytes) {
   memoryPool->moveFromDevice(hostPtr, devicePtr, numBytes);
}

void runtime::DeviceMemoryManager::registerHostMemAsPinned(uint8_t* hostPtr, int64_t numBytes) {
   memoryPool->registerHostMemAsPinned(hostPtr, numBytes);
}

uint8_t* runtime::DeviceMemoryManager::getThreadStream() {
   return memoryPool->getThreadStream();
}

void runtime::DeviceMemoryManager::setThreadStream(uint8_t* streamPtr) {
   return memoryPool->setThreadStream(streamPtr);
}

uint8_t* runtime::DeviceMemoryManager::getThreadStreamEvent() {
   return memoryPool->getThreadStreamEvent();
}

void runtime::DeviceMemoryManager::syncStream(uint8_t* streamPtr) {
   memoryPool->syncThreadStream(streamPtr);
}

uint8_t* runtime::DeviceMemoryManager::getPtrForArray(const uint32_t arraySize) {
   assert(arraySize && "Requesting empty allocation?");
   uint8_t* result{nullptr};
   try {
      result = memoryPool->getPtrForArray(arraySize);
   } catch (const std::bad_alloc& e) {
      std::unique_lock<std::mutex> guardEviction(evictionMtx);
      bool success{false};
      while (!success) {
         std::unique_lock<std::mutex> guardAllocate(cacheMtx);
         try {
            evict(arraySize);
            result = memoryPool->getPtrForArray(arraySize);
            success = true;
         } catch (const std::bad_alloc& e) { // Erase more unpinned
            // std::cout << "Erasing  unpinned\n";
         } catch (const std::runtime_error& e) { // Wait until pinned get unpinned
            // std::cout << "Waiting\n";
            cv.wait(guardAllocate);
         }
      }
      if (!success) {
         std::this_thread::sleep_for(std::chrono::milliseconds(2000));
         std::cout << "statesOnDevice.size(): " << statesOnDevice.size() << ", fifo Size = " << fifoCache.size() << "\n";

         std::cerr << "Eviction failed for the required size: " << arraySize / mb << "MB, execute on CPU" << std::endl;
         throw std::runtime_error("[getPtrForArray] GPU OOM");
      }
   }
   assert(result);
   return result;
}

uint32_t runtime::DeviceMemoryManager::getAvailableSMEMSize(const uint32_t deviceId) {
   return memoryPool->getAvailableSMEMSize(deviceId);
}

uint32_t runtime::DeviceMemoryManager::getSMCount(const uint32_t deviceId) {
   return memoryPool->getSMCount(deviceId);
}

DeviceArrayToken runtime::DeviceMemoryManager::getPtrForStateArray(runtime::stateID id, const uint32_t arraySize) {
   // std::unique_lock lock(cache_mtx);
   auto it = statesOnDevice.find(id);
   if (it != statesOnDevice.end()) {
      it->second.first.pinned++;
      if (it->second.second != fifoCache.end()) {
         fifoCache.erase(it->second.second); // Cache only contains states eligible for eviction, pinned are not eligible.
         it->second.second = fifoCache.end();
      }
      return {it->second.first.devicePtr, true};
   }
   uint8_t* allocation = getPtrForArray(arraySize);
   if (allocation) {
      std::unique_lock<std::mutex> guard(cacheMtx);
      // std::cout << "[insert statesOnDevice] " << id << "\n";
      // Timer t{"Insert"};
      statesOnDevice.emplace(id, std::make_pair(DeviceStateStatus{allocation, arraySize, 1}, fifoCache.end()));
      return {allocation, false};
   }
   assert(0 && "By now we must either return or throw in getPtrForArray");
   return {nullptr, false};
}

void runtime::DeviceMemoryManager::freeArray(uint8_t* devicePtr, bool possiblyRelease) {
   memoryPool->freeArrayPtr(devicePtr, possiblyRelease);
}

void runtime::DeviceMemoryManager::freeStateArray(runtime::stateID id, bool possiblyRelease) {
   auto it = statesOnDevice.find(id);
   assert(it != statesOnDevice.end() && "Cannot free: state array is not tracked");
   assert(!it->second.first.pinned);
   // std::cout << "[freeStateArray] " << id << ", ptr="<< (void*)it->second.devicePtr <<  "\n";
   freeArray(it->second.first.devicePtr, possiblyRelease);
   fifoCache.erase(it->second.second);
   statesOnDevice.erase(it);
}

void runtime::DeviceMemoryManager::freeStateArrayByIter(stateTrackerType::iterator it, bool possiblyRelease) {
   freeArray(it->second.first.devicePtr, possiblyRelease);
   fifoCache.erase(it->second.second);
   statesOnDevice.erase(it);
}

void runtime::DeviceMemoryManager::unpinState(runtime::stateID id) {
   std::lock_guard<std::mutex> guard(cacheMtx);
   auto it = statesOnDevice.find(id);
   assert(it != statesOnDevice.end() && "Cannot unpin: state array is not tracked");
   // std::cout << "Unpinning " << id << " at " << (void*)it->second.devicePtr << "\n";
   assert(it->second.first.pinned);
   it->second.first.pinned = false;
   fifoCache.push_front(id);
   it->second.second = fifoCache.begin();
   cv.notify_one();
}

void runtime::DeviceMemoryManager::unpinAllStates() {
   std::lock_guard<std::mutex> guard(cacheMtx);
   for (auto& [id, stateIter] : statesOnDevice) {
      if (stateIter.first.pinned) {
         assert(stateIter.second == fifoCache.end());
         stateIter.first.pinned = false;
         fifoCache.push_front(id);
         stateIter.second = fifoCache.begin();
      }
   }
}

void runtime::DeviceMemoryManager::evict(uint64_t evictSize) {
   uint64_t accumulatedSize{0};
   std::vector<stateTrackerType::iterator> statesToEvict;
   // std::cout << "Trying to evict buffers to fit " << evictSize/MB << " MB \n";
   for (auto cacheIt = fifoCache.begin(); cacheIt != fifoCache.end(); cacheIt++) {
      auto it = statesOnDevice.find(*cacheIt);
      assert(!it->second.first.pinned);
      if (it != statesOnDevice.end()) {
         statesToEvict.push_back(it);
         accumulatedSize += it->second.first.size;
         if (accumulatedSize >= evictSize) {
            break;
         }
      }
   }
   if (accumulatedSize < evictSize) {
      // std::cout << "For a " << evictSize/MB << " MB data, accumulated " << statesToEvict.size() << " unpinned states of total size " << accumulated_size/MB << "MB, visited " << cnt << " elements \n";
      // std::cout << "statesOnDevice.size(): " << statesOnDevice.size() << "\n";
      throw std::runtime_error("[evict] Not enough unused states to evict for the requested size!");
   }

   for (auto cacheIt : statesToEvict) {
      freeStateArrayByIter(cacheIt);
   }
}
