#ifndef RUNTIME_ARROWSCHEMA_H
#define RUNTIME_ARROWSCHEMA_H
#include "helpers.h"

#include <arrow/type_fwd.h>
namespace runtime {
class ArrowSchema {
   std::shared_ptr<arrow::Schema> schema;
   ArrowSchema(std::shared_ptr<arrow::Schema> schema):schema(schema){}
   public:
   static ArrowSchema* createFromString(runtime::VarLen32 schemaDescription);
   [[nodiscard]] std::shared_ptr<arrow::Schema> getSchema() const { return schema; }
};
} // namespace runtime
#endif //RUNTIME_ARROWSCHEMA_H
