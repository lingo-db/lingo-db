#ifndef LINGODB_BRIDGE_H
#define LINGODB_BRIDGE_H
#include <memory>
#include <arrow/c/bridge.h>
namespace runtime {
class ExecutionContext;
}
namespace bridge {
__attribute__((visibility("default"))) void createInMemory();
__attribute__((visibility("default"))) void loadFromDisk(const char* directory);
__attribute__((visibility("default"))) void addTable(const char* name, ArrowArrayStream* recordBatchStream);
__attribute__((visibility("default"))) void run(const char* module);
__attribute__((visibility("default"))) void runSQL(const char* query);

}
#endif //LINGODB_BRIDGE_H
