#include "runtime/ArrowSchema.h"
#include<algorithm>
#include <arrow/table.h>

namespace {
std::shared_ptr<arrow::DataType> createType(std::string name, uint32_t p1, uint32_t p2) {
   if (name == "int") {
      switch (p1) {
         case 8: return arrow::int8();
         case 16: return arrow::int16();
         case 32: return arrow::int32();
         case 64: return arrow::int64();
      }
   } else if (name == "uint") {
      switch (p1) {
         case 8: return arrow::uint8();
         case 16: return arrow::uint16();
         case 32: return arrow::uint32();
         case 64: return arrow::uint64();
      }
   } else if (name == "float") {
      switch (p1) {
         case 16: return arrow::float16();
         case 32: return arrow::float32();
         case 64: return arrow::float64();
      }
   } else if (name == "string") {
      return arrow::utf8();
   } else if (name == "fixed_sized") {
      return arrow::fixed_size_binary(p1);
   } else if (name == "date") {
      return p1 == 32 ? arrow::date32() : arrow::date64();
   } else if (name == "interval_months") {
      return arrow::month_interval();
   } else if (name == "interval_daytime") {
      return arrow::day_time_interval();
   } else if (name == "timestamp") {
      return arrow::timestamp(static_cast<arrow::TimeUnit::type>(p1));
   } else if (name == "decimal") {
      return arrow::decimal(p1, p2);
   } else if (name == "bool") {
      return arrow::boolean();
   }
   throw std::runtime_error("unknown type");
}
static std::shared_ptr<arrow::Schema> parseSchema(std::string str) {
   if (str.empty()) {
      return std::make_shared<arrow::Schema>(std::vector<std::shared_ptr<arrow::Field>>{});
   }
   std::vector<std::shared_ptr<arrow::Field>> fields;

   str.erase(std::remove_if(str.begin(), str.end(), [](char c) { return c == ' '; }), str.end());
   auto parseEntry = [&fields](std::string token) {
      size_t colonPos = token.find(":");
      if (colonPos == std::string::npos) throw std::runtime_error("expected ':'");
      std::string colName = token.substr(0, colonPos);
      std::string typeDescr = token.substr(colonPos + 1);
      size_t lParamPos = typeDescr.find("[");
      std::string p1 = "0";
      std::string p2 = "0";
      std::string typeName = typeDescr;
      if (lParamPos != std::string ::npos) {
         typeName = typeDescr.substr(0, lParamPos);
         assert(typeDescr.ends_with(']'));
         std::string paramString = typeDescr.substr(lParamPos + 1, typeDescr.size() - lParamPos - 2);
         size_t commaPos = paramString.find(",");
         if (commaPos == std::string::npos) {
            p1 = paramString;
         } else {
            p1 = paramString.substr(0, commaPos);
            p2 = paramString.substr(commaPos + 1);
         }
      }
      fields.push_back(std::make_shared<arrow::Field>(colName, createType(typeName, std::stoi(p1), std::stoi(p2))));
   };
   size_t pos = 0;
   std::string token;
   while ((pos = str.find(";")) != std::string::npos) {
      token = str.substr(0, pos);
      str.erase(0, pos + 1);
      parseEntry(token);
   }
   parseEntry(str);
   return std::make_shared<arrow::Schema>(fields);
}
}
runtime::ArrowSchema* runtime::ArrowSchema::createFromString(runtime::VarLen32 schemaDescription) {
   return new ArrowSchema(parseSchema(schemaDescription.str()));
}
