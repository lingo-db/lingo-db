#include <iostream>
#include <regex>

#include <arrow/buffer.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>

#include "json.h"

#include "runtime/metadata.h"

namespace {
std::string serializeRecordBatch(std::shared_ptr<arrow::RecordBatch> batch) {
   std::shared_ptr<arrow::ResizableBuffer> buffer = arrow::AllocateResizableBuffer(0).ValueOrDie();
   std::unique_ptr<arrow::io::BufferOutputStream> bufferOutputStream = std::make_unique<arrow::io::BufferOutputStream>(buffer);
   std::shared_ptr<arrow::ipc::RecordBatchWriter> recordBatchWriter = arrow::ipc::MakeStreamWriter(bufferOutputStream.get(), batch->schema()).ValueOrDie();

   assert(recordBatchWriter->WriteRecordBatch(*batch) == arrow::Status::OK());
   assert(recordBatchWriter->Close() == arrow::Status::OK());
   auto resBuffer = bufferOutputStream->Finish();
   return resBuffer.ValueOrDie()->ToHexString();
}
//adapted from stack overflow: https://stackoverflow.com/a/30606613
std::shared_ptr<arrow::ResizableBuffer> hexToBytes(const std::string& hex) {
   auto bytes = arrow::AllocateResizableBuffer(hex.size() / 2).ValueOrDie();
   for (unsigned int i = 0; i < hex.length(); i += 2) {
      std::string byteString = hex.substr(i, 2);
      char byte = (char) strtol(byteString.c_str(), NULL, 16);
      bytes->mutable_data()[i / 2] = byte;
   }

   return bytes;
}
std::shared_ptr<arrow::RecordBatch> deserializeRecordBatch(std::string str) {
   auto resizableBuffer = hexToBytes(str);
   auto reader = arrow::ipc::RecordBatchStreamReader::Open(std::make_shared<arrow::io::BufferReader>(resizableBuffer)).ValueOrDie();
   std::shared_ptr<arrow::RecordBatch> batch;
   assert(reader->ReadNext(&batch) == arrow::Status::OK());
   return batch;
}
} // end namespace
std::shared_ptr<runtime::ColumnMetaData> createColumnMetaData(const nlohmann::json& info) {
   auto res = std::make_shared<runtime::ColumnMetaData>();
   if (info.contains("distinct_values")) {
      res->setDistinctValues(info["distinct_values"].get<size_t>());
   }
   if (info.contains("type")) {
      runtime::ColumnType columnType;
      const nlohmann::json& typeInfo = info["type"];
      if (typeInfo.contains("base")) {
         columnType.base = typeInfo["base"];
      }
      if (typeInfo.contains("nullable")) {
         columnType.nullable = typeInfo["nullable"];
      } else {
         columnType.nullable = false;
      }
      if (typeInfo.contains("props")) {
         for (auto x : typeInfo["props"]) {
            std::variant<size_t, std::string> prop;
            if (x.is_number()) {
               prop = x.get<size_t>();
            } else {
               prop = x.get<std::string>();
            }
            columnType.modifiers.push_back(prop);
         }
      }
      res->setColumnType(columnType);
   }
   return res;
}
std::shared_ptr<runtime::TableMetaData> runtime::TableMetaData::create(const std::string& jsonMetaData, const std::string& name = "", std::shared_ptr<arrow::RecordBatch> sample = std::shared_ptr<arrow::RecordBatch>()) {
   auto res = std::make_shared<runtime::TableMetaData>();
   res->present = !jsonMetaData.empty() || sample;
   res->sample = sample;
   if (jsonMetaData.empty()) return res;
   auto json = nlohmann::json::parse(jsonMetaData);
   if (json.contains("tables")) {
      json = json["tables"];
      if (!json.contains(name)) return res;
      json = json[name];
   }

   if (json.contains("num_rows")) {
      res->numRows = json["num_rows"];
   }
   if (json.contains("pkey")) {
      res->primaryKey = std::vector<std::string>();
      for (auto c : json["pkey"]) {
         res->primaryKey.push_back(c);
      }
   }
   if (json.contains("sample")) {
      res->sample = deserializeRecordBatch(json["sample"]);
   }
   if (!json.contains("columns")) return res;
   for (auto c : json["columns"].get<nlohmann::json::array_t>()) {
      auto columnName = c["name"];
      res->orderedColumns.push_back(columnName);
      res->columns[columnName] = createColumnMetaData(c);
   }
   return res;
}
nlohmann::json::object_t serializeColumn(std::shared_ptr<runtime::ColumnMetaData> column) {
   nlohmann::json::object_t res;
   if (column->getDistinctValues()) {
      res["distinct_values"] = column->getDistinctValues().value();
   }
   auto columnType = column->getColumnType();
   res["type"] = nlohmann::json::object_t();
   res["type"]["base"] = columnType.base;
   res["type"]["nullable"] = columnType.nullable;
   auto props = nlohmann::json::array_t();
   for (auto x : columnType.modifiers) {
      if (std::holds_alternative<size_t>(x)) {
         props.push_back(std::get<size_t>(x));
      } else {
         props.push_back(std::get<std::string>(x));
      }
   }
   res["type"]["props"] = props;
   return res;
}
std::string runtime::TableMetaData::serialize(bool serializeSample) const {
   nlohmann::json json;
   json["num_rows"] = numRows;
   json["pkey"] = primaryKey;
   if (sample && serializeSample) {
      json["sample"] = serializeRecordBatch(sample);
   }
   json["columns"] = nlohmann::json::array_t();
   for (auto c : orderedColumns) {
      auto serializedColumn = serializeColumn(columns.at(c));
      serializedColumn["name"] = c;
      json["columns"].push_back(serializedColumn);
   }
   std::string str = json.dump();
   return str;
}
std::shared_ptr<runtime::TableMetaData> runtime::TableMetaData::deserialize(std::string str) {
   return create(str);
}
bool runtime::TableMetaData::isPresent() const {
   return present;
}
const std::vector<std::string>& runtime::TableMetaData::getOrderedColumns() const {
   return orderedColumns;
}
const std::optional<size_t>& runtime::ColumnMetaData::getDistinctValues() const {
   return distinctValues;
}
void runtime::ColumnMetaData::setDistinctValues(const std::optional<size_t>& distinctValues) {
   ColumnMetaData::distinctValues = distinctValues;
}
const runtime::ColumnType& runtime::ColumnMetaData::getColumnType() const {
   return columnType;
}
void runtime::ColumnMetaData::setColumnType(const runtime::ColumnType& columnType) {
   ColumnMetaData::columnType = columnType;
}
