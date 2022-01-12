#include "runtime/metadata.h"
#include "runtime/database.h"
#include "vendored/json.h"
#include <iostream>
#include <regex>
std::shared_ptr<runtime::ColumnMetaData> createColumnMetaData(const nlohmann::json& info) {
   auto res = std::make_shared<runtime::ColumnMetaData>();
   if (info.contains("distinct_values")) {
      res->setDistinctValues(info["distinct_values"].get<size_t>());
   }
   return res;
}
std::shared_ptr<runtime::TableMetaData> runtime::TableMetaData::create(const std::string& jsonMetaData, const std::string& name = "", std::shared_ptr<arrow::RecordBatch> sample = std::shared_ptr<arrow::RecordBatch>()) {
   auto res = std::make_shared<runtime::TableMetaData>();
   res->present = !jsonMetaData.empty()||sample;
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
      res->sample = runtime::Database::deserializeRecordBatch(json["sample"]);
   }
   if (!json.contains("columns")) return res;
   for (auto c : json["columns"].get<nlohmann::json::object_t>()) {
      auto columnName = c.first;
      res->columns[columnName] = createColumnMetaData(c.second);
   }
   return res;
}
nlohmann::json::object_t serializeColumn(std::shared_ptr<runtime::ColumnMetaData> column) {
   nlohmann::json::object_t res;
   if (column->getDistinctValues()) {
      res["distinct_values"] = column->getDistinctValues().value();
   }
   return res;
}
std::string runtime::TableMetaData::serialize() const {
   nlohmann::json json;
   json["num_rows"] = numRows;
   json["pkey"] = primaryKey;
   if (sample) {
      json["sample"] = Database::serializeRecordBatch(sample);
   }
   json["columns"] = nlohmann::json::object_t();
   for (auto c : columns) {
      json["columns"][c.first] = serializeColumn(c.second);
   }
   std::string str = json.dump();
   str = std::regex_replace(str, std::regex("\""), std::string("''"));
   return str;
}
std::shared_ptr<runtime::TableMetaData> runtime::TableMetaData::deserialize(std::string str) {
   str = std::regex_replace(str, std::regex("''"), std::string("\""));
   return create(str);
}
bool runtime::TableMetaData::isPresent() const {
   return present;
}
const std::optional<size_t>& runtime::ColumnMetaData::getDistinctValues() const {
   return distinctValues;
}
void runtime::ColumnMetaData::setDistinctValues(const std::optional<size_t>& distinctValues) {
   ColumnMetaData::distinctValues = distinctValues;
}
