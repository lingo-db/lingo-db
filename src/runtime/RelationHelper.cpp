#include "lingodb/runtime/RelationHelper.h"

#include "json.h"
#include "lingodb/catalog/FunctionCatalogEntry.h"

#include "lingodb/catalog/IndexCatalogEntry.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/runtime/ArrowTable.h"
#include "lingodb/runtime/ExternalDataSourceProperty.h"
#include "lingodb/runtime/storage/LingoDBTable.h"
#include "lingodb/runtime/storage/TableStorage.h"
#include "lingodb/utility/Serialization.h"

#include <arrow/builder.h>
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <dlfcn.h>
#include <lingodb/catalog/Defs.h>
#include <parquet/arrow/writer.h>
#include <parquet/properties.h>

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
void RelationHelper::createFunction(runtime::VarLen32 meta) {
   auto* context = getCurrentExecutionContext();
   auto& session = context->getSession();
   auto catalog = session.getCatalog();
   auto def = utility::deserializeFromHexString<lingodb::catalog::CreateFunctionDef>(meta.str());
   std::shared_ptr<lingodb::catalog::FunctionCatalogEntry> func;
   if (def.language == "c") {
      func = std::make_shared<lingodb::catalog::CFunctionCatalogEntry>(def.name, def.code, def.returnType, def.argumentTypes);
      //Remove possible so file
      auto find = catalog::FunctionCatalogEntry::getUdfFunctions().find(def.name);
      if (find != catalog::FunctionCatalogEntry::getUdfFunctions().end()) {
         dlclose(find->second.handle);
         catalog::FunctionCatalogEntry::getUdfFunctions().erase(find);
      }
      if (!catalog->getDbDir().empty() && std::filesystem::exists(catalog->getDbDir() + "/udf/" + def.name + ".so")) {
         std::filesystem::remove(catalog->getDbDir() + "/udf/" + def.name + ".so");
      }
   } else if (def.language == "python") {
      func = std::make_shared<lingodb::catalog::PythonFunctionCatalogEntry>(def.name, def.code, def.returnType, def.argumentTypes);
   } else {
      throw std::runtime_error("unsupported function language: " + def.language);
   }
   catalog->insertEntry(func, true);
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
void RelationHelper::copyFromIntoTableCSV(lingodb::runtime::VarLen32 tableName, lingodb::runtime::VarLen32 fileName, lingodb::runtime::VarLen32 delimiter, lingodb::runtime::VarLen32 escape, bool header) {
   auto* context = getCurrentExecutionContext();
   auto& session = context->getSession();
   auto catalog = session.getCatalog();
   if (auto relation = catalog->getTypedEntry<lingodb::catalog::TableCatalogEntry>(tableName)) {
      arrow::io::IOContext ioContext = arrow::io::default_io_context();
      auto inputFile = arrow::io::ReadableFile::Open(fileName.str()).ValueOrDie();
      std::shared_ptr<arrow::io::InputStream> input = inputFile;

      auto readOptions = arrow::csv::ReadOptions::Defaults();
      if (header) {
         readOptions.skip_rows = 1;
      }
      auto parseOptions = arrow::csv::ParseOptions::Defaults();
      parseOptions.delimiter = delimiter.str().front();
      if (size_t escapeStringLen = escape.getLen(); escapeStringLen > 0) {
         if (escapeStringLen > 1) {
            throw std::runtime_error("escape character must be a single character");
         }
         const char escapeChar = escape.str().front();
         if (escapeChar == '"') {
            parseOptions.escaping = false;
            parseOptions.double_quote = true;
         } else {
            parseOptions.escape_char = escapeChar;
            parseOptions.escaping = true;
            parseOptions.double_quote = false;
         }
      }
      parseOptions.newlines_in_values = true;
      auto convertOptions = arrow::csv::ConvertOptions::Defaults();
      convertOptions.null_values.push_back("");
      convertOptions.strings_can_be_null = true;
      auto& storage = relation.value()->getTableStorage();
      std::vector<std::string> fixedSizeBinaryColumns;
      for (auto n : relation.value()->getColumnNames()) {
         readOptions.column_names.emplace_back(n);
         auto arrowType = storage.getColumnStorageType(n);
         // we store db::char<1> as arrow::fixed_size_binary<4> in the table storage, but arrow CSV reader doesn't allow reading fixed_size_binary<4> for single chars in the csv
         if (arrowType->id() == arrow::Type::FIXED_SIZE_BINARY) {
            fixedSizeBinaryColumns.emplace_back(n);
            arrowType = arrow::utf8();
         }
         convertOptions.column_types.emplace(n, arrowType);
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

      // correct single char columns from arrow::utf8 to arrow::fixed_size_binary<4>
      for (auto n : fixedSizeBinaryColumns) {
         auto column = table->GetColumnByName(n);
         auto fsbBuilder = std::make_unique<arrow::FixedSizeBinaryBuilder>(arrow::fixed_size_binary(4), arrow::default_memory_pool());
         std::array<uint8_t, 4> buf;

         auto chunks = column->chunks();
         for (const auto& chunk : chunks) {
            const auto chunkStrArray = std::static_pointer_cast<arrow::StringArray>(chunk);
            for (int64_t i = 0; i < chunkStrArray->length(); i++) {
               const std::string_view str = chunkStrArray->GetView(i);
               if (str.empty()) {
                  if (!fsbBuilder->AppendNull().ok()) {
                     throw std::runtime_error("failed to append null fixed-size binary data");
                  }
               } else {
                  buf.fill(0);
                  std::ranges::copy(str.begin(), str.end(), buf.begin());
                  if (!fsbBuilder->Append(buf).ok()) {
                     throw std::runtime_error("failed to append fixed-size binary data");
                  }
               }
            }
         }
         const auto fsbArray = fsbBuilder->Finish().ValueOrDie();
         const auto newColumnChunked = std::make_shared<arrow::ChunkedArray>(fsbArray);

         auto field = arrow::field(n, arrow::fixed_size_binary(4));
         table = table->SetColumn(table->schema()->GetFieldIndex(n), field, newColumnChunked).ValueOrDie();
      }

      appendToTable(session, tableName.str(), table);
      catalog->persist();
   } else {
      throw std::runtime_error("copy failed: no such table");
   }
}
void RelationHelper::copyToFromTableCSV(runtime::VarLen32 tableName, runtime::VarLen32 fileName, runtime::VarLen32 delimiter, bool header) {
   std::shared_ptr<arrow::io::FileOutputStream> outfile;
   auto table = getArrowTableFromName(tableName.str());

   auto openResult = arrow::io::FileOutputStream::Open(fileName.str());
   if (!openResult.ok()) {
      throw std::runtime_error("Error opening file" + openResult.status().ToString());
   }
   outfile = openResult.ValueOrDie();
   auto writeOptions = arrow::csv::WriteOptions::Defaults();
   writeOptions.delimiter = delimiter.str().front();
   writeOptions.include_header = header;
   auto status = arrow::csv::WriteCSV(*table, writeOptions, outfile.get());
   if (!status.ok()) {
      throw std::runtime_error("copy failed");
   }
}
void RelationHelper::copyToFromTableParquet(runtime::VarLen32 tableName, runtime::VarLen32 fileName, runtime::VarLen32 compression) {
   std::shared_ptr<arrow::io::FileOutputStream> outfile;
   auto table = getArrowTableFromName(tableName.str());

   // Convert fixed_size_binary columns to string for export
   std::vector<int> fixedSizeBinaryColumnIndices;
   for (int i = 0; i < table->num_columns(); i++) {
      if (table->field(i)->type()->id() == arrow::Type::FIXED_SIZE_BINARY) {
         fixedSizeBinaryColumnIndices.push_back(i);
      }
   }
   for (int idx : fixedSizeBinaryColumnIndices) {
      auto column = table->column(idx);
      auto stringBuilder = std::make_unique<arrow::StringBuilder>(arrow::default_memory_pool());

      auto chunks = column->chunks();
      for (const auto& chunk : chunks) {
         const auto fsbArray = std::static_pointer_cast<arrow::FixedSizeBinaryArray>(chunk);
         for (int64_t i = 0; i < fsbArray->length(); i++) {
            if (fsbArray->IsNull(i)) {
               if (!stringBuilder->AppendNull().ok()) {
                  throw std::runtime_error("failed to append null string data");
               }
            } else {
               std::string str = fsbArray->GetString(i);
               // Trim at first null byte
               size_t nullPos = str.find('\0');
               if (nullPos != std::string::npos) {
                  str.resize(nullPos);
               }
               if (!stringBuilder->Append(str).ok()) {
                  throw std::runtime_error("failed to append string data");
               }
            }
         }
      }
      const auto stringArray = stringBuilder->Finish().ValueOrDie();
      const auto newColumnChunked = std::make_shared<arrow::ChunkedArray>(stringArray);
      auto field = arrow::field(table->field(idx)->name(), arrow::utf8());
      table = table->SetColumn(idx, field, newColumnChunked).ValueOrDie();
   }

   auto openResult = arrow::io::FileOutputStream::Open(fileName.str());
   if (!openResult.ok()) {
      throw std::runtime_error("Error opening file" + openResult.status().ToString());
   }
   outfile = openResult.ValueOrDie();
   auto compressionStr = compression.str();
   auto compressionType = arrow::Compression::SNAPPY;
   if (compressionStr == "UNCOMPRESSED") {
      compressionType = arrow::Compression::UNCOMPRESSED;
   } else if (compressionStr == "GZIP") {
      compressionType = arrow::Compression::GZIP;
   } else if (compressionStr == "BROTLI") {
      compressionType = arrow::Compression::BROTLI;
   } else if (compressionStr == "ZSTD") {
      compressionType = arrow::Compression::ZSTD;
   } else if (compressionStr == "LZ4") {
      compressionType = arrow::Compression::LZ4;
   } else if (compressionStr == "LZ4_FRAME") {
      compressionType = arrow::Compression::LZ4_FRAME;
   } else if (compressionStr == "LZO") {
      compressionType = arrow::Compression::LZO;
   } else if (compressionStr == "BZ2") {
      compressionType = arrow::Compression::BZ2;
   } else if (compressionStr == "LZ4_HADOOP") {
      compressionType = arrow::Compression::LZ4_HADOOP;
   } else if (compressionStr == "SNAPPY") {
   } else {
      throw std::runtime_error("Unsupported compression type");
   }

   std::shared_ptr<parquet::WriterProperties> props =
      parquet::WriterProperties::Builder().compression(compressionType)->build();

   std::shared_ptr<parquet::ArrowWriterProperties> arrowProps =
      parquet::ArrowWriterProperties::Builder().store_schema()->build();

   auto status = parquet::arrow::WriteTable(*table.get(), arrow::default_memory_pool(), outfile,
                                            /*chunk_size=*/64 * 1024, props, arrowProps);
   if (!status.ok()) {
      throw std::runtime_error("copy failed: " + status.ToString());
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
   std::string dataSourceRaw = description.str();
   auto dataSource = lingodb::utility::deserializeFromHexString<ExternalDatasourceProperty>(dataSourceRaw);

   std::string indexName = dataSource.index;
   auto& session = context->getSession();
   auto catalog = session.getCatalog();
   if (auto relation = catalog->getTypedEntry<catalog::LingoDBHashIndexEntry>(indexName)) {
      auto* hashIndex = static_cast<LingoDBHashIndex*>(&relation.value()->getIndex());
      std::vector<std::string> cols;
      for (auto& m : dataSource.mapping) {
         cols.push_back(m.identifier);
      }

      auto* access = new HashIndexAccess(*hashIndex, cols);
      context->registerState({access, [&](void* ptr) { delete static_cast<HashIndexAccess*>(ptr); }});
      return access;
   } else {
      throw std::runtime_error("Table " + description.str() + " not found");
   }

   throw std::runtime_error("index unsupported for now");
}

std::shared_ptr<arrow::Table> RelationHelper::getArrowTableFromName(std::string tableName) {
   auto* context = getCurrentExecutionContext();
   auto& session = context->getSession();
   auto catalog = session.getCatalog();
   if (auto relation = catalog->getTypedEntry<lingodb::catalog::TableCatalogEntry>(tableName)) {
      std::shared_ptr<arrow::io::FileOutputStream> outfile;
      // Ensure the table is loaded into memory and export from the in-memory representation
      relation.value()->ensureFullyLoaded();
      auto* lingoTable = dynamic_cast<lingodb::runtime::LingoDBTable*>(&relation.value()->getTableStorage());
      if (!lingoTable) {
         throw std::runtime_error("unsupported table storage");
      }

      lingoTable->ensureLoaded();
      std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
      for (const auto& chunk : *lingoTable->getTableChunks()) {
         batches.push_back(chunk.data());
      }
      if (batches.empty()) {
         throw std::runtime_error("no data to export");
      }
      auto table = arrow::Table::FromRecordBatches(batches).ValueOrDie();

      if (!table) {
         // empty table -> create empty Arrow table with schema from storage
         auto schema = relation.value()->getSample().getSampleData() ? relation.value()->getSample().getSampleData()->schema() : nullptr;
         if (!schema) {
            throw std::runtime_error("no data to export");
         }
         table = arrow::Table::FromRecordBatches({}).ValueOrDie();
         return table;
      }
      return table;
   } else {
      throw std::runtime_error("no such table found " + tableName);
   }
}
} // end namespace lingodb::runtime
