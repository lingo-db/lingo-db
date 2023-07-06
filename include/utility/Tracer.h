#ifndef UTILITY_TRACER_H
#define UTILITY_TRACER_H
#include <cassert>
#include <chrono>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace utility {

class Tracer {
   public:
#ifdef TRACER
   struct Event {
      unsigned id;
      Event(std::string_view category, std::string_view name) : id(registerEvent(category, name)) {}
   };
#else
   /// An event
   struct Event {
      constexpr Event(std::string_view /*category*/, std::string_view /*name*/) {}
   };
#endif

   struct TraceRecord {
      unsigned threadId;
      unsigned eventId;
      uint64_t traceBegin;
      uint64_t traceEnd;
      uint64_t metaData;
   };
   class TraceRecordList {
      struct Chunk {
         static constexpr unsigned size = 4096;
         TraceRecord traceRecords[size];
         Chunk* next;
      };
      //thread information
      unsigned threadId;
      std::string threadName;

      //list of chunks
      Chunk *first = nullptr, *last = nullptr;

      //current write position
      size_t currentPos = Chunk::size;

      friend class Tracer;

      public:
      explicit TraceRecordList(unsigned threadId, std::string threadName) : threadId(threadId), threadName(threadName) {}
      ~TraceRecordList();

      TraceRecord* createRecord() {
         if (currentPos == Chunk::size) {
            auto* nextChunk = new Chunk();
            nextChunk->next = nullptr;
            if (last) {
               last->next = nextChunk;
            } else {
               first = nextChunk;
            }
            last = nextChunk;
            currentPos = 0;
         }
         auto* record = &last->traceRecords[currentPos++];
         record->threadId = threadId;
         return record;
      }
   };

   private:
   std::mutex mutex;
   std::vector<std::unique_ptr<TraceRecordList>> traceRecordLists;
   //{category, name}
   std::vector<std::pair<std::string, std::string>> eventDescriptions;

   void dumpInternal();

   static unsigned registerEvent(std::string_view category, std::string_view name);
   static void ensureThreadLocalTraceRecordList();
   static void recordTrace(unsigned eventId, std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end, uint64_t metaData);

   public:
   Tracer() {}
   ~Tracer() {}

#ifdef TRACER
   class Trace {
      private:
      unsigned eventId;
      bool alreadyRecorded = false;
      std::chrono::steady_clock::time_point begin;
      uint64_t metaData;

      public:
      Trace(const Event& event) : eventId(event.id), begin(std::chrono::steady_clock::now()), metaData(-1) {}
      ~Trace() { stop(); }
      void setMetaData(uint64_t metaData) {
         this->metaData = metaData;
      }
      void stop() {
         if (!alreadyRecorded) {
            recordTrace(eventId, begin, std::chrono::steady_clock::now(), metaData);
            alreadyRecorded = true;
         }
      }
   };

#else
   class Trace {
      public:
      constexpr Trace(const Event& /*event*/) {}
      void setMetaData(uint64_t metaData) {}
      constexpr ~Trace() {}
      void stop() {}
   };
#endif

   static void dump();
};
} // end namespace utility
#endif // UTILITY_TRACER_H
