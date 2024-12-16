#include "lingodb/runtime/RelationHelper.h"

#include <iostream>

#include "json.h"

#include "lingodb/runtime/ArrowTable.h"
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
namespace lingodb::runtime {
void RelationHelper::createTable(lingodb::runtime::ExecutionContext* context, lingodb::runtime::VarLen32 name, lingodb::runtime::VarLen32 meta) {
   auto& session = context->getSession();
   auto catalog = session.getCatalog();
   catalog->addTable(name.str(), lingodb::runtime::TableMetaData::deserialize(meta.str()));
}
void RelationHelper::appendTableFromResult(lingodb::runtime::VarLen32 tableName, lingodb::runtime::ExecutionContext* context, size_t resultId) {
   {
      auto resultTable = context->getResultOfType<lingodb::runtime::ArrowTable>(resultId);
      if (!resultTable) {
         throw std::runtime_error("appending result table failed: no result table");
      }
      auto& session = context->getSession();
      auto catalog = session.getCatalog();
      if (auto relation = catalog->findRelation(tableName)) {
         relation->append(resultTable.value()->get());
      } else {
         throw std::runtime_error("appending result table failed: no such table");
      }
   }
}
void RelationHelper::copyFromIntoTable(lingodb::runtime::ExecutionContext* context, lingodb::runtime::VarLen32 tableName, lingodb::runtime::VarLen32 fileName, lingodb::runtime::VarLen32 delimiter, lingodb::runtime::VarLen32 escape) {
   auto& session = context->getSession();
   auto catalog = session.getCatalog();
   if (auto relation = catalog->findRelation(tableName)) {
      arrow::io::IOContext ioContext = arrow::io::default_io_context();
      auto inputFile = arrow::io::ReadableFile::Open(fileName.str()).ValueOrDie();
      std::shared_ptr<arrow::io::InputStream> input = inputFile;

      auto readOptions = arrow::csv::ReadOptions::Defaults();

      auto parseOptions = arrow::csv::ParseOptions::Defaults();
      parseOptions.delimiter = delimiter.str().front();
      if (escape.getLen() > 0) {
         parseOptions.escape_char = escape.str().front();
         parseOptions.escaping = true;
      }
      parseOptions.newlines_in_values = true;
      auto convertOptions = arrow::csv::ConvertOptions::Defaults();
      auto schema = relation->getArrowSchema();
      convertOptions.null_values.push_back("");
      convertOptions.strings_can_be_null = true;
      for (auto f : schema->fields()) {
         if (f->name().find("primaryKeyHashValue") != std::string::npos) continue;
         readOptions.column_names.push_back(f->name());
         convertOptions.column_types.insert({f->name(), f->type()});
      }

      // Instantiate TableReader from input stream and options
      auto maybeReader = arrow::csv::TableReader::Make(ioContext,
                                                       input,
                                                       readOptions,
                                                       parseOptions,
                                                       convertOptions);
      if (!maybeReader.ok()) {
         // Handle TableReader instantiation error...
      }
      std::shared_ptr<arrow::csv::TableReader> reader = *maybeReader;

      // Read table from CSV file
      auto maybeTable = reader->Read();
      if (!maybeTable.ok()) {
         // Handle CSV read error
         // (for example a CSV syntax error or failed type conversion)
      }
      std::shared_ptr<arrow::Table> table = *maybeTable;
      relation->append(table);
   } else {
      throw std::runtime_error("copy failed: no such table");
   }
}
void RelationHelper::setPersist(lingodb::runtime::ExecutionContext* context, bool value) {
   auto& session = context->getSession();
   auto catalog = session.getCatalog();
   catalog->setPersist(value);
}
HashIndexAccess* RelationHelper::getIndex(lingodb::runtime::ExecutionContext* context, lingodb::runtime::VarLen32 description) {
   auto json = nlohmann::json::parse(description.str());
   std::string relationName = json["relation"];
   std::string index = json["index"];
   auto& session = context->getSession();
   auto catalog = session.getCatalog();
   if (auto relation = catalog->findRelation(relationName)) {
      auto* hashIndex = static_cast<HashIndex*>(relation->getIndex(index).get());
      std::vector<std::string> cols;
      for (auto m : json["mapping"].get<nlohmann::json::object_t>()) {
         cols.push_back(m.second.get<std::string>());
      }
      return new HashIndexAccess(*hashIndex, cols);
   } else {
      throw std::runtime_error("no such table");
   }
}
} // end namespace lingodb::runtime