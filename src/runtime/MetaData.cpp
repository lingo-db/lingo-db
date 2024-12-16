#include "lingodb/runtime/metadata.h"

#include <arrow/buffer.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>

#include <iostream>
#include <regex>

#include "json.h"

namespace {
std::string serializeRecordBatch(std::shared_ptr<arrow::RecordBatch> batch) {
   std::shared_ptr<arrow::ResizableBuffer> buffer = arrow::AllocateResizableBuffer(0).ValueOrDie();
   std::unique_ptr<arrow::io::BufferOutputStream> bufferOutputStream = std::make_unique<arrow::io::BufferOutputStream>(buffer);
   std::shared_ptr<arrow::ipc::RecordBatchWriter> recordBatchWriter = arrow::ipc::MakeStreamWriter(bufferOutputStream.get(), batch->schema()).ValueOrDie();

   if (!recordBatchWriter->WriteRecordBatch(*batch).ok() || !recordBatchWriter->Close().ok()) {
      throw std::runtime_error("MetaData: Failed to write record batch");
   }
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
   if (!reader->ReadNext(&batch).ok()) {
      throw std::runtime_error("could not deserialize batch");
   }
   return batch;
}
std::shared_ptr<lingodb::runtime::ColumnMetaData> createColumnMetaData(const nlohmann::json& info) {
   auto res = std::make_shared<lingodb::runtime::ColumnMetaData>();
   if (info.contains("distinct_values")) {
      res->setDistinctValues(info["distinct_values"].get<size_t>());
   }
   if (info.contains("type")) {
      lingodb::runtime::ColumnType columnType;
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

nlohmann::json::object_t serializeColumn(std::shared_ptr<lingodb::runtime::ColumnMetaData> column) {
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
nlohmann::json::object_t serializeIndex(std::shared_ptr<lingodb::runtime::IndexMetaData> indexMetaData) {
   nlohmann::json::object_t res;
   res["name"] = indexMetaData->name;
   res["columns"] = nlohmann::json::array_t();
   for (auto c : indexMetaData->columns) {
      res["columns"].push_back(c);
   }
   res["type"] = indexMetaData->type;
   return res;
}
} // end namespace

std::shared_ptr<lingodb::runtime::TableMetaData> lingodb::runtime::TableMetaData::create(const std::string& jsonMetaData, const std::string& name = "", std::shared_ptr<arrow::RecordBatch> sample = std::shared_ptr<arrow::RecordBatch>()) {
   auto res = std::make_shared<lingodb::runtime::TableMetaData>();
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
   if (json.contains("indices")) {
      for (auto idx : json["indices"].get<nlohmann::json::array_t>()) {
         auto metaData = std::make_shared<IndexMetaData>();
         metaData->type = idx["type"];
         metaData->name = idx["name"];
         for (auto c : idx["columns"].get<nlohmann::json::array_t>()) {
            metaData->columns.push_back(c);
         }
         res->indices.push_back(metaData);
      }
   }
   if (!json.contains("columns")) return res;
   for (auto c : json["columns"].get<nlohmann::json::array_t>()) {
      auto columnName = c["name"];
      res->orderedColumns.push_back(columnName);
      res->columns[columnName] = createColumnMetaData(c);
   }
   return res;
}
std::string lingodb::runtime::TableMetaData::serialize(bool serializeSample) const {
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
   json["indices"] = nlohmann::json::array_t();
   for (auto idx : indices) {
      json["indices"].push_back(serializeIndex(idx));
   }
   std::string str = json.dump();
   return str;
}
std::shared_ptr<lingodb::runtime::TableMetaData> lingodb::runtime::TableMetaData::deserialize(std::string str) {
   return create(str);
}
bool lingodb::runtime::TableMetaData::isPresent() const {
   return present;
}
const std::vector<std::string>& lingodb::runtime::TableMetaData::getOrderedColumns() const {
   return orderedColumns;
}
const std::optional<size_t>& lingodb::runtime::ColumnMetaData::getDistinctValues() const {
   return distinctValues;
}
void lingodb::runtime::ColumnMetaData::setDistinctValues(const std::optional<size_t>& distinctValues) {
   ColumnMetaData::distinctValues = distinctValues;
}
const lingodb::runtime::ColumnType& lingodb::runtime::ColumnMetaData::getColumnType() const {
   return columnType;
}
void lingodb::runtime::ColumnMetaData::setColumnType(const lingodb::runtime::ColumnType& columnType) {
   ColumnMetaData::columnType = columnType;
}
