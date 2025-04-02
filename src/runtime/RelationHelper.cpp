#include "lingodb/runtime/RelationHelper.h"

#include <iostream>

#include "json.h"

#include "lingodb/catalog/IndexCatalogEntry.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/runtime/ArrowTable.h"
#include "lingodb/runtime/storage/TableStorage.h"
#include "lingodb/utility/Serialization.h"
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <lingodb/catalog/Defs.h>

#include <lingodb/runtime/storage/LingoDBTable.h>
namespace lingodb::runtime {
void RelationHelper::createTable(lingodb::runtime::VarLen32 meta) {
   auto* context = getCurrentExecutionContext();
   auto& session = context->getSession();
   auto catalog = session.getCatalog();
   auto def = utility::deserializeFromHexString<lingodb::catalog::CreateTableDef>(meta.str());
   auto relation = lingodb::catalog::LingoDBTableCatalogEntry::createFromCreateTable(def);
   catalog->insertEntry(relation);
   if (!def.primaryKey.empty()) {
      auto index = lingodb::catalog::LingoDBHashIndexEntry::createForPrimaryKey(def.name, def.primaryKey);
      catalog->insertEntry(index);
      relation->addIndex(index->getName());
   }
   catalog->persist();
}
void RelationHelper::appendToTable(runtime::Session& session, std::string tableName, std::shared_ptr<arrow::Table> table) {
   auto catalog = session.getCatalog();
   if (auto relation = catalog->getTypedEntry<catalog::TableCatalogEntry>(tableName)) {
      auto startRowId = relation.value()->getTableStorage().nextRowId();
      relation.value()->getTableStorage().append(table);
      for (auto idx : relation.value()->getIndices()) {
         if (auto index = catalog->getTypedEntry<catalog::IndexCatalogEntry>(idx.first)) {
            index.value()->getIndex().bulkInsert(startRowId, table);
         }
      }
      catalog->persist();
   } else {
      throw std::runtime_error("appending result table failed: no such table");
   }
}
void RelationHelper::appendTableFromResult(lingodb::runtime::VarLen32 tableName, size_t resultId) {
   auto* context = getCurrentExecutionContext();
   {
      auto resultTable = context->getResultOfType<lingodb::runtime::ArrowTable>(resultId);
      if (!resultTable) {
         throw std::runtime_error("appending result table failed: no result table");
      }
      auto& session = context->getSession();
      appendToTable(session, tableName.str(), resultTable.value()->get());
   }
}
void RelationHelper::copyFromIntoTable(lingodb::runtime::VarLen32 tableName, lingodb::runtime::VarLen32 fileName, lingodb::runtime::VarLen32 delimiter, lingodb::runtime::VarLen32 escape) {
   auto* context = getCurrentExecutionContext();
   auto& session = context->getSession();
   auto catalog = session.getCatalog();
   if (auto relation = catalog->getTypedEntry<lingodb::catalog::TableCatalogEntry>(tableName)) {
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
      convertOptions.null_values.push_back("");
      convertOptions.strings_can_be_null = true;
      auto& storage = relation.value()->getTableStorage();
      for (auto n : relation.value()->getColumnNames()) {
         readOptions.column_names.push_back(n);
         convertOptions.column_types.insert({n, storage.getColumnStorageType(n)});
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
      appendToTable(session, tableName.str(), table);
      catalog->persist();
   } else {
      throw std::runtime_error("copy failed: no such table");
   }
}
void RelationHelper::setPersist(bool value) {
   auto* context = getCurrentExecutionContext();
   auto& session = context->getSession();
   auto catalog = session.getCatalog();
   catalog->setShouldPersist(value);
}
HashIndexAccess* RelationHelper::accessHashIndex(lingodb::runtime::VarLen32 description) {
   auto* context = runtime::getCurrentExecutionContext();
   auto json = nlohmann::json::parse(description.str());
   std::string indexName = json["index"];
   auto& session = context->getSession();
   auto catalog = session.getCatalog();
   if (auto relation = catalog->getTypedEntry<catalog::LingoDBHashIndexEntry>(indexName)) {
      auto* hashIndex = static_cast<LingoDBHashIndex*>(&relation.value()->getIndex());
      std::vector<std::string> cols;
      for (auto m : json["mapping"].get<nlohmann::json::object_t>()) {
         cols.push_back(m.second.get<std::string>());
      }
      return new HashIndexAccess(*hashIndex, cols);
   } else {
      throw std::runtime_error("no such table");
   }

   throw std::runtime_error("index unsupported for now");
}
} // end namespace lingodb::runtime