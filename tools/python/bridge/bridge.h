#ifndef LINGODB_BRIDGE_H
#define LINGODB_BRIDGE_H
#include <memory>
#include <arrow/c/bridge.h>
namespace bridge {
class Connection;
__attribute__((visibility("default"))) Connection* createInMemory();
__attribute__((visibility("default"))) Connection* loadFromDisk(const char* directory);
__attribute__((visibility("default"))) void createTable(Connection* con, const char* name, const char* metaData);
__attribute__((visibility("default"))) void appendTable(Connection* con, const char* name, ArrowArrayStream* recordBatchStream);
__attribute__((visibility("default"))) bool run(Connection* con, const char* module, ArrowArrayStream* res);
__attribute__((visibility("default"))) bool runSQL(Connection* con, const char* query, ArrowArrayStream* res);
__attribute__((visibility("default"))) double getTiming(Connection* con, const char* type);
__attribute__((visibility("default"))) void closeConnection(Connection* con);

}
#endif //LINGODB_BRIDGE_H
