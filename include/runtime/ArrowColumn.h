
#ifndef RUNTIME_ARROWCOLUMN_H
#define RUNTIME_ARROWCOLUMN_H
#include "helpers.h"

#include <arrow/type_fwd.h>
namespace runtime {
class ArrowColumn {
   std::shared_ptr<arrow::ChunkedArray> column;

   public:
   ArrowColumn(std::shared_ptr<arrow::ChunkedArray> column) : column(column) {}
   std::shared_ptr<arrow::ChunkedArray> getColumn() const { return column; }
};
class ArrowColumnBuilder {
   std::unique_ptr<arrow::ArrayBuilder> builderUnique;
   arrow::ArrayBuilder* builder;
   ArrowColumnBuilder* childBuilder;
   std::shared_ptr<arrow::DataType> type;
   std::vector<std::shared_ptr<arrow::Array>> additionalArrays;
   ArrowColumnBuilder(std::shared_ptr<arrow::DataType> type);
   ArrowColumnBuilder(arrow::ArrayBuilder* childBuilder);

   public:
   static ArrowColumnBuilder* create(VarLen32 type);
   ArrowColumnBuilder* getChildBuilder();
   void addBool(bool isValid, bool value);
   void addInt8(bool isValid, int8_t);
   void addInt16(bool isValid, int16_t);
   void addInt32(bool isValid, int32_t);
   void addInt64(bool isValid, int64_t);
   void addFloat32(bool isValid, float);
   void addFloat64(bool isValid, double);
   void addDecimal(bool isValid, __int128);
   void addFixedSized(bool isValid, int64_t);
   void addList(bool isValid);
   void addBinary(bool isValid, runtime::VarLen32);
   void merge(ArrowColumnBuilder* other);
   ArrowColumn* finish();
   ~ArrowColumnBuilder();
};

} // namespace runtime
#endif //RUNTIME_ARROWCOLUMN_H
