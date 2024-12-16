#ifndef LINGODB_UTILITY_TRACER_H
#define LINGODB_UTILITY_TRACER_H
#include "json_fwd.h"
#include <cassert>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
namespace utility {

class Tracer {
   public:
   struct TraceRecord {
      unsigned threadId;
      unsigned eventId;
      uint64_t traceBegin;
      uint64_t traceEnd;
      uint64_t metaData;
   };
#ifdef TRACER
   struct Event {
      unsigned id;
      std::string category;
      std::string name;
      Event(std::string_view category, std::string_view name) : id(registerEvent(this)), category(category), name(name) {}

      virtual void writeOut(TraceRecord* record, nlohmann::json& j);
   };
#else
   /// An event
   struct Event {
      constexpr Event(std::string_view /*category*/, std::string_view /*name*/) {}

   };
#endif

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
   std::vector<Event*> eventDescriptions;

   void dumpInternal();

   static unsigned registerEvent(Event* event);
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

      protected:
      uint64_t metaData;

      public:
      Trace(const Event& event, uint64_t metaData) : eventId(event.id), begin(std::chrono::steady_clock::now()), metaData(metaData) {}
      Trace(const Event& event) : eventId(event.id), begin(std::chrono::steady_clock::now()), metaData(-1) {}
      ~Trace() { stop(); }
      void stop() {
         if (!alreadyRecorded) {
            recordTrace(eventId, begin, std::chrono::steady_clock::now(), metaData);
            alreadyRecorded = true;
         }
      }
   };

   template <class E, class M>
   class MetaDataTrace : public Trace {
      public:
      MetaDataTrace(E& event, M metaData) : Trace(event, event.serializeMetaData(metaData)) {}
   };
   class StringMetaDataEvent : public Event {
      std::vector<std::string> strings;
      std::mutex mutex;
      std::string metaName;

      public:
      StringMetaDataEvent(std::string category, std::string name, std::string metaName) : utility::Tracer::Event(category, name), metaName(metaName) {}
      void writeOut(utility::Tracer::TraceRecord* record, nlohmann::json& j) override;
      uint64_t serializeMetaData(std::string pass);
   };
#else
   class Trace {
      public:
      constexpr Trace(const Event& /*event*/) {}
      constexpr Trace(const Event& /*event*/,uint64_t) {}
      void setMetaData(uint64_t metaData) {}
      constexpr ~Trace() {}
      void stop() {}
   };
   template <class E, class M>
   class MetaDataTrace : public Trace {
      public:
      MetaDataTrace(const E& event, M metaData) : Trace(event,0) {}
   };
   class StringMetaDataEvent : public Event {

      public:
      StringMetaDataEvent(std::string category, std::string name, std::string metaName) : utility::Tracer::Event(category, name) {}
   };
#endif

   static void dump();
};
} // end namespace utility
#endif // LINGODB_UTILITY_TRACER_H
