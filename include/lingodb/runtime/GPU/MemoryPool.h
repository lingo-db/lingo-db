#ifndef LINGODB_RUNTIME_GPU_MEMORYPOOL_H
#define LINGODB_RUNTIME_GPU_MEMORYPOOL_H

#include <cstdint>
#include <future>
#include <random>

constexpr uint64_t kb{1024};
constexpr uint64_t mb{kb * 1024};
constexpr uint64_t poolThreshold{1024 * mb};

struct DeviceArrayToken {
   uint8_t* devicePtr;
   bool cached;
};

struct DeviceStateStatus {
   uint8_t* devicePtr;
   uint32_t size;
   uint32_t pinned;
};

class DeviceMemoryPool {
   public:
   DeviceMemoryPool(const int32_t deviceId) {}

   virtual ~DeviceMemoryPool() {};

   virtual uint8_t* getPtrForArray(const size_t arraySize) = 0;
   virtual void freeArrayPtr(uint8_t* devicePtr, bool possiblyReleaseToOS) = 0;
   virtual void moveToDevice(uint8_t* hostPtr, uint8_t* devicePtr, int64_t numBytes, bool sync) = 0;
   virtual void moveFromDevice(uint8_t* hostPtr, uint8_t* devicePtr, int64_t numBytes) = 0;
   virtual void syncThreadStream(uint8_t* streamPtr = nullptr) = 0;
   virtual uint8_t* getThreadStream() = 0;
   virtual void setThreadStream(uint8_t*) = 0;
   virtual uint8_t* getThreadStreamEvent() = 0;
   virtual void registerHostMemAsPinned(uint8_t* hostPtr, int64_t numBytes) = 0;
   virtual uint32_t getAvailableSMEMSize(const uint32_t deviceId) = 0;
   virtual uint32_t getSMCount(const uint32_t deviceId) = 0;
};

#endif //LINGODB_RUNTIME_GPU_MEMORYPOOL_H
