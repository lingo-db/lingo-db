#include "lingodb/utility/Tracer.h"

#include "json.h"

#include "lingodb/utility/Setting.h"
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <unistd.h>

namespace {
utility::GlobalSetting<std::string> traceOutputDir("system.trace_dir", ".");

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
#ifdef TRACER
void Tracer::Event::writeOut(utility::Tracer::TraceRecord* record, nlohmann::json& j) {
   j["tid"] = record->threadId;
   j["name"] = name;
   j["category"] = category;
   j["start"] = record->traceBegin;
   j["duration"] = record->traceEnd - record->traceBegin;
}
void Tracer::StringMetaDataEvent::writeOut(utility::Tracer::TraceRecord* record, nlohmann::json& j) {
   utility::Tracer::Event::writeOut(record, j);
   j["extra"] = nlohmann::json::object();
   j["extra"][metaName] = strings[record->metaData];
}
uint64_t Tracer::StringMetaDataEvent::serializeMetaData(std::string pass) {
   std::unique_lock<std::mutex> lock(mutex);
   uint64_t res = strings.size();
   strings.push_back(pass);
   return res;
}
#endif

Tracer::TraceRecordList::~TraceRecordList() {
   auto* curr = first;
   while (curr) {
      auto* next = curr->next;
      delete curr;
      curr = next;
   }
}

unsigned Tracer::registerEvent(Event* event) {
   auto* tracer = getTracer();
   std::unique_lock<std::mutex> lock(tracer->mutex);
   auto id = tracer->eventDescriptions.size();
   tracer->eventDescriptions.emplace_back(event);
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
const std::chrono::steady_clock::time_point initial = std::chrono::steady_clock::now();
void Tracer::recordTrace(unsigned eventId, std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end, uint64_t metaData) {
   ensureThreadLocalTraceRecordList();
   auto diffInMicroSecond = [](auto a, auto b) {
      return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
   };
   auto* record = threadLocalTraceRecordList->createRecord();
   record->eventId = eventId;
   record->traceBegin = diffInMicroSecond(initial, begin);
   record->traceEnd = diffInMicroSecond(initial, end);
   record->metaData = metaData;
}
void Tracer::dump() {
   getTracer()->dumpInternal();
}

void Tracer::dumpInternal() {
   std::unique_lock<std::mutex> lock(mutex);
   std::string filename = traceOutputDir.getValue() + "/lingodb.trace";
   std::ofstream out(filename);
   auto result = nlohmann::json::array();
   std::vector<TraceRecord> traceRecords;
   for (auto& list : traceRecordLists) {
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
#ifdef TRACER
      auto* event = eventDescriptions[r.eventId];
      auto recordObject = nlohmann::json::object();
      event->writeOut(&r, recordObject);
      result.push_back(recordObject);
#endif
   }
   auto fileContent = nlohmann::json::object();
   fileContent["fileType"] = "traceOnly";
   fileContent["version"] = "0.0.3";
   fileContent["trace"] = result;
   out << to_string(fileContent) << std::endl;
}
} // end namespace utility
