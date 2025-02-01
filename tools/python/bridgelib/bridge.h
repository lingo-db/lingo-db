#ifndef LINGODB_BRIDGE_H
#define LINGODB_BRIDGE_H
#include <memory>
#include <arrow/c/bridge.h>
#include "mlir-c/IR.h"
namespace bridge {
class Connection;
struct MLIRValueRangeImpl;
struct MLIRValueRange{
   MLIRValueRangeImpl* impl;
};
struct UDF{
   void* data;
   MlirValue (*call)(UDF*,bridge::MLIRValueRange,MlirLocation,MlirOperation);
};
;

__attribute__((visibility("default"))) Connection* createInMemory();
__attribute__((visibility("default"))) Connection* loadFromDisk(const char* directory);
__attribute__((visibility("default"))) void createTable(Connection* con, const char* name, const char* metaData);
__attribute__((visibility("default"))) void appendTable(Connection* con, const char* name, ArrowArrayStream* recordBatchStream);
__attribute__((visibility("default"))) bool run(Connection* con, const char* module, ArrowArrayStream* res);
__attribute__((visibility("default"))) bool runSQL(Connection* con, const char* query, ArrowArrayStream* res);
__attribute__((visibility("default"))) double getTiming(Connection* con, const char* type);
__attribute__((visibility("default"))) void closeConnection(Connection* con);
__attribute__((visibility("default"))) void initContext(MlirContext context);

__attribute__((visibility("default"))) void addUDF(bridge::Connection* con, const char* name, UDF* udf);

__attribute__((visibility("default"))) void addValueToRange(MLIRValueRange,MlirValue);
__attribute__((visibility("default"))) size_t valueRangeGetLen(MLIRValueRange);
__attribute__((visibility("default"))) MlirValue valueRangeGet(MLIRValueRange,size_t offset);
__attribute__((visibility("default"))) MLIRValueRange createValueRange();

}
#endif //LINGODB_BRIDGE_H
