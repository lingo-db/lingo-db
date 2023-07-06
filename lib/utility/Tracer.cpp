#include "utility/Tracer.h"

#include "json.h"

#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <unistd.h>

namespace {
thread_local utility::Tracer::TraceRecordList* threadLocalTraceRecordList = nullptr;
utility::Tracer* singleton() {
#ifdef TRACER
   static utility::Tracer perfTracer;
   return &perfTracer;
#else
   return nullptr;
#endif
}
utility::Tracer* getTracer() {
   auto* tracer = singleton();
   assert(tracer);
   return tracer;
}
} // end namespace
namespace utility {
Tracer::TraceRecordList::~TraceRecordList() {
   auto* curr = first;
   while (curr) {
      auto* next = curr->next;
      delete curr;
      curr = next;
   }
}

unsigned Tracer::registerEvent(std::string_view category, std::string_view name) {
   auto* tracer = getTracer();
   std::unique_lock<std::mutex> lock(tracer->mutex);
   auto id = tracer->eventDescriptions.size();
   tracer->eventDescriptions.emplace_back(category, name);

   return id;
}
void Tracer::ensureThreadLocalTraceRecordList() {
   auto* tracer = getTracer();
   if (threadLocalTraceRecordList == nullptr) {
      std::unique_lock<std::mutex> lock(tracer->mutex);
      auto threadId = tracer->traceRecordLists.size();
      char name[16];
      pthread_getname_np(pthread_self(), name, 16);
      tracer->traceRecordLists.emplace_back(std::make_unique<TraceRecordList>(tracer->traceRecordLists.size(), std::string(name)));
      threadLocalTraceRecordList = tracer->traceRecordLists[threadId].get();
   }
}
void Tracer::recordTrace(unsigned eventId, std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end,uint64_t metaData) {
   static std::chrono::steady_clock::time_point initial = std::chrono::steady_clock::now();
   ensureThreadLocalTraceRecordList();
   auto diffInMicroSecond = [](auto a, auto b) {
      return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
   };
   auto* record = threadLocalTraceRecordList->createRecord();
   record->eventId = eventId;
   record->traceBegin = diffInMicroSecond(initial, begin);
   record->traceEnd = diffInMicroSecond(initial, end);
   record->metaData=metaData;
}
void Tracer::dump() {
   getTracer()->dumpInternal();
}

void Tracer::dumpInternal() {
   std::unique_lock<std::mutex> lock(mutex);
   std::string filename = "lingodb.trace";
   std::ofstream out(filename);
   int pid = getpid();
   auto result = nlohmann::json::object();
   result["traceEvents"] = nlohmann::json::array();
   auto& eventList = result["traceEvents"];
   std::vector<TraceRecord> traceRecords;
   for (auto& list : traceRecordLists) {
      auto threadObject = nlohmann::json::object();
      threadObject["name"] = "thread_name";
      threadObject["ph"] = "M";
      threadObject["pid"] = pid;
      threadObject["tid"] = list->threadId;
      {
         threadObject["args"] = nlohmann::json::object();
         auto& args = threadObject["args"];
         args["name"] = list->threadName;
      }
      eventList.push_back(threadObject);
      for (auto* it = list->first; it != list->last; it = it->next) {
         traceRecords.insert(traceRecords.end(), &it->traceRecords[0], &it->traceRecords[TraceRecordList::Chunk::size]);
      }
      if (list->last) {
         for (auto i = 0ull; i < list->currentPos; i++) {
            traceRecords.push_back(list->last->traceRecords[i]);
         }
      }
   }

   std::stable_sort(traceRecords.begin(), traceRecords.end(), [](const TraceRecord& a, const TraceRecord& b) { //workaround, std::sort crashes :(
      return a.traceBegin < b.traceEnd;
   });
   for (auto& r : traceRecords) {
      assert(r.eventId < eventDescriptions.size());
      auto& eventDescription = eventDescriptions[r.eventId];
      auto recordObject = nlohmann::json::object();
      recordObject["name"] = eventDescription.second;
      recordObject["cat"] = eventDescription.first;
      recordObject["ph"] = "X";
      recordObject["pid"] = pid;
      recordObject["tid"] = r.threadId;
      recordObject["ts"] = r.traceBegin;
      recordObject["dur"] = r.traceEnd - r.traceBegin;
      {
         recordObject["args"] = nlohmann::json::object();
         auto& args = recordObject["args"];
         args["meta"] = r.metaData;
      }
      eventList.push_back(recordObject);
   }

   result["displayTimeUnit"] = "ms";
   out << to_string(result) << std::endl;
   std::cout << "trace file written to file:" << filename << std::endl;
}
} // end namespace utility
