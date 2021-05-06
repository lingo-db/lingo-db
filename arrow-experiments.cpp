
#include "runtime/database.h"
#include <iostream>
#include <arrow/array.h>
#include <arrow/pretty_print.h>

int main() {
   auto database = runtime::Database::load("resources/data/tpch");
   arrow::PrettyPrint(*database->getTable("region"), {}, &std::cerr);
   arrow::PrettyPrint(*database->getTable("nation"), {}, &std::cerr);

   auto table=database->getTable("region");
   arrow::TableBatchReader test(*table);
   test.set_chunksize(3);
   std::shared_ptr<arrow::RecordBatch> batch;
   while(test.ReadNext(&batch)==arrow::Status::OK()&&batch){
      std::cout<<batch->column_data(0)->buffers.size()<<std::endl;
      std::cout<<batch->column_data(1)->buffers.size()<<std::endl;
      std::cout<<batch->column_data(2)->buffers.size()<<std::endl;

   }

   auto universitydb = runtime::Database::load("resources/data/uni");
   arrow::PrettyPrint(*universitydb->getTable("studenten"), {}, &std::cerr);
   arrow::PrettyPrint(*universitydb->getTable("vorlesungen"), {}, &std::cerr);
   return 0;
}