#ifndef LINGODB_UTILITY_PERFEVENT_H
#define LINGODB_UTILITY_PERFEVENT_H

#if defined(__linux__)

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <asm/unistd.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <unistd.h>

struct PerfEvent {
   struct Event {
      struct ReadFormat {
         uint64_t value;
         uint64_t timeEnabled;
         uint64_t timeRunning;
         uint64_t id;
      };

      perf_event_attr pe;
      int fd;
      ReadFormat prev;
      ReadFormat data;

      double readCounter() {
         double multiplexingCorrection = static_cast<double>(data.timeEnabled - prev.timeEnabled) / static_cast<double>(data.timeRunning - prev.timeRunning);
         return static_cast<double>(data.value - prev.value) * multiplexingCorrection;
      }
   };

   enum EventDomain : uint8_t { USER = 0b1,
                                KERNEL = 0b10,
                                HYPERVISOR = 0b100,
                                ALL = 0b111 };

   std::vector<Event> events;
   std::vector<std::string> names;
   std::chrono::time_point<std::chrono::steady_clock> startTime;
   std::chrono::time_point<std::chrono::steady_clock> stopTime;

   PerfEvent() {
      registerCounter("cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
      registerCounter("kcycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, KERNEL);
      registerCounter("instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
      registerCounter("L1-misses", PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
      registerCounter("LLC-misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);
      registerCounter("branch-misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);
      registerCounter("task-clock", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK);
      registerCounter("pf", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS);
      registerCounter("pf-min", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MIN);
      registerCounter("pf-maj", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MAJ);
      registerCounter("alignment faults", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_ALIGNMENT_FAULTS);
      // additional counters can be found in linux/perf_event.h

      for (unsigned i = 0; i < events.size(); i++) {
         auto& event = events[i];
         event.fd = static_cast<int>(syscall(__NR_perf_event_open, &event.pe, 0, -1, -1, 0));
         if (event.fd < 0) {
            std::cerr << "Error opening counter " << names[i] << std::endl;
            events.resize(0);
            names.resize(0);
            return;
         }
      }
   }

   void registerCounter(const std::string& name, uint64_t type, uint64_t eventID, EventDomain domain = ALL) {
      names.push_back(name);
      events.push_back(Event());
      auto& event = events.back();
      auto& pe = event.pe;
      memset(&pe, 0, sizeof(struct perf_event_attr));
      pe.type = static_cast<uint32_t>(type);
      pe.size = sizeof(struct perf_event_attr);
      pe.config = eventID;
      pe.disabled = true;
      pe.inherit = 1;
      pe.inherit_stat = 0;
      pe.exclude_user = !(domain & USER);
      pe.exclude_kernel = !(domain & KERNEL);
      pe.exclude_hv = !(domain & HYPERVISOR);
      pe.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
   }

   void startCounters() {
      for (unsigned i = 0; i < events.size(); i++) {
         auto& event = events[i];
         ioctl(event.fd, PERF_EVENT_IOC_RESET, 0);
         ioctl(event.fd, PERF_EVENT_IOC_ENABLE, 0);
         if (read(event.fd, &event.prev, sizeof(uint64_t) * 3) != sizeof(uint64_t) * 3)
            std::cerr << "Error reading counter " << names[i] << std::endl;
      }
      startTime = std::chrono::steady_clock::now();
   }

   ~PerfEvent() {
      for (auto& event : events) {
         close(event.fd);
      }
   }

   void stopCounters() {
      stopTime = std::chrono::steady_clock::now();
      for (unsigned i = 0; i < events.size(); i++) {
         auto& event = events[i];
         if (read(event.fd, &event.data, sizeof(uint64_t) * 3) != sizeof(uint64_t) * 3)
            std::cerr << "Error reading counter " << names[i] << std::endl;
         ioctl(event.fd, PERF_EVENT_IOC_DISABLE, 0);
      }
   }

   double getDuration() {
      return std::chrono::duration<double>(stopTime - startTime).count();
   }

   double getIPC() {
      return getCounter("instructions") / getCounter("cycles");
   }

   double getCPUs() {
      return getCounter("task-clock") / (getDuration() * 1e9);
   }

   double getGHz() {
      return getCounter("cycles") / getCounter("task-clock");
   }

   double getCounter(const std::string& name) {
      for (unsigned i = 0; i < events.size(); i++)
         if (names[i] == name)
            return events[i].readCounter();
      return -1;
   }

   static void printCounter(std::ostream& headerOut, std::ostream& dataOut, std::string name, std::string counterValue, bool addComma = true) {
      auto width = std::max(name.length(), counterValue.length());
      headerOut << std::setw(static_cast<int>(width)) << name << (addComma ? "," : "") << " ";
      dataOut << std::setw(static_cast<int>(width)) << counterValue << (addComma ? "," : "") << " ";
   }

   template <typename T>
   static void printCounter(std::ostream& headerOut, std::ostream& dataOut, std::string name, T counterValue, bool addComma = true) {
      std::stringstream stream;
      stream << std::fixed << std::setprecision(2) << counterValue;
      PerfEvent::printCounter(headerOut, dataOut, name, stream.str(), addComma);
   }

   void printReport(std::ostream& out, uint64_t normalizationConstant) {
      std::stringstream header;
      std::stringstream data;
      printReport(header, data, normalizationConstant);
      out << header.str() << std::endl;
      out << data.str() << std::endl;
   }

   void printReport(std::ostream& headerOut, std::ostream& dataOut, uint64_t normalizationConstant) {
      if (!events.size())
         return;

      // print all metrics
      for (unsigned i = 0; i < events.size(); i++) {
         printCounter(headerOut, dataOut, names[i], events[i].readCounter() / static_cast<double>(normalizationConstant));
      }

      printCounter(headerOut, dataOut, "scale", normalizationConstant);

      // derived metrics
      printCounter(headerOut, dataOut, "IPC", getIPC());
      printCounter(headerOut, dataOut, "CPUs", getCPUs());
      printCounter(headerOut, dataOut, "GHz", getGHz(), false);
   }
};

struct BenchmarkParameters {
   void setParam(const std::string& name, const std::string& value) {
      params[name] = value;
   }

   void setParam(const std::string& name, const char* value) {
      params[name] = value;
   }

   template <typename T>
   void setParam(const std::string& name, T value) {
      setParam(name, std::to_string(value));
   }

   void printParams(std::ostream& header, std::ostream& data) {
      for (auto& p : params) {
         PerfEvent::printCounter(header, data, p.first, p.second);
      }
   }

   BenchmarkParameters(std::string name = "") {
      if (name.length())
         setParam("name", name);
   }

   private:
   std::map<std::string, std::string> params;
};

struct PerfRef {
   union {
      PerfEvent instance;
      PerfEvent* pointer;
   };
   bool hasInstance;

   PerfRef() : instance(), hasInstance(true) {}
   PerfRef(PerfEvent* ptr) : pointer(ptr), hasInstance(false) {}
   PerfRef(const PerfRef&) = delete;

   ~PerfRef() {
      if (hasInstance)
         instance.~PerfEvent();
   }

   PerfEvent* operator->() {
      return hasInstance ? &instance : pointer;
   }
};

struct PerfEventBlock {
   PerfRef e;
   uint64_t scale;
   BenchmarkParameters parameters;
   bool printHeader;

   PerfEventBlock(uint64_t scale = 1, BenchmarkParameters params = {}, bool printHeader = true)
      : scale(scale),
        parameters(params),
        printHeader(printHeader) {
      e->startCounters();
   }

   PerfEventBlock(PerfEvent& perf, uint64_t scale = 1, BenchmarkParameters params = {}, bool printHeader = true)
      : e(&perf),
        scale(scale),
        parameters(params),
        printHeader(printHeader) {
      e->startCounters();
   }

   ~PerfEventBlock() {
      e->stopCounters();
      std::stringstream header;
      std::stringstream data;
      parameters.printParams(header, data);
      PerfEvent::printCounter(header, data, "time sec", e->getDuration());
      e->printReport(header, data, scale);
      if (printHeader)
         std::cout << header.str() << std::endl;
      std::cout << data.str() << std::endl;
   }
};

#else
#include <ostream>
struct PerfEvent {
   void startCounters() {}
   void stopCounters() {}
   void printReport(std::ostream&, uint64_t) {}
   template <class T>
   void setParam(const std::string&, const T&){};
};

struct BenchmarkParameters {
};

struct PerfEventBlock {
   PerfEventBlock(uint64_t = 1, BenchmarkParameters = {}, bool = true){};
   PerfEventBlock(PerfEvent e, uint64_t = 1, BenchmarkParameters = {}, bool = true){};
};
#endif
#endif