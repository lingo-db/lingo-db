#ifndef RUNTIME_EXECUTION_CONTEXT_H
#define RUNTIME_EXECUTION_CONTEXT_H
#include "database.h"
namespace runtime {
class ExecutionContext {
   public:
   int id;
   std::unique_ptr<Database> db;
};
} // end namespace runtime
using db_execution_context=uint8_t*;
using db_table=uint8_t*;
using db_table_chunk=uint8_t*;
using db_table_column_buffer=uint8_t*;

using db_table_chunk_iterator=uint8_t*;
using db_column_id=uint64_t;
using db_buffer_id=uint64_t;

#endif // RUNTIME_EXECUTION_CONTEXT_H
