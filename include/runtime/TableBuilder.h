#ifndef RUNTIME_TABLEBUILDER_H
#define RUNTIME_TABLEBUILDER_H
#include "ExecutionContext.h"
#include "runtime/helpers.h"
#include "runtime/ArrowSchema.h"
#include <cassert>

#include "ThreadLocal.h"
#include <arrow/type_fwd.h>
class TableBuilder;
namespace runtime {

class ResultTable {
   std::shared_ptr<arrow::Table> resultTable;
   TableBuilder* builder;

   public:
   std::shared_ptr<arrow::Table> get();
   //interface for generated code
   static ResultTable* create(ExecutionContext*,ArrowSchema* schema);
   static ResultTable* merge(ThreadLocal* threadLocal);
   void addBool(bool isValid, bool value);
   void addInt8(bool isValid, int8_t);
   void addInt16(bool isValid, int16_t);
   void addInt32(bool isValid, int32_t);
   void addInt64(bool isValid, int64_t);
   void addFloat32(bool isValid, float);
   void addFloat64(bool isValid, double);
   void addDecimal(bool isValid, __int128);
   void addFixedSized(bool isValid, int64_t);
   void addBinary(bool isValid, runtime::VarLen32);
   void nextRow();
};
} // end namespace runtime
#endif //RUNTIME_TABLEBUILDER_H
