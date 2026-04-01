#include "lingodb/catalog/Catalog.h"

#include "lingodb/catalog/Defs.h"
#include "lingodb/catalog/FunctionCatalogEntry.h"
#include "lingodb/catalog/IndexCatalogEntry.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/catalog/Types.h"
#include "lingodb/utility/Serialization.h"

#include <arrow/io/api.h>
#include <arrow/type.h>

#include <filesystem>
#include <parquet/arrow/reader.h>

namespace {
using lingodb::catalog::Column;
using lingodb::catalog::CreateTableDef;
using lingodb::catalog::DateTypeInfo;
using lingodb::catalog::LogicalTypeId;
using lingodb::catalog::TimestampTypeInfo;
using lingodb::catalog::Type;

Type mapArrowTypeToCatalogType(const std::shared_ptr<arrow::DataType>& type) {
   auto id = type->id();
   switch (id) {
      case arrow::Type::BOOL:
         return Type::boolean();
      case arrow::Type::INT8:
         return Type::int8();
      case arrow::Type::INT16:
         return Type::int16();
      case arrow::Type::INT32:
         return Type::int32();
      case arrow::Type::INT64:
         return Type::int64();
      case arrow::Type::FLOAT:
         return Type::f32();
      case arrow::Type::DOUBLE:
         return Type::f64();
      case arrow::Type::DECIMAL128: {
         auto decimalType = std::static_pointer_cast<arrow::Decimal128Type>(type);
         return Type::decimal(decimalType->precision(), decimalType->scale());
      }
      case arrow::Type::DATE32:
         return Type(LogicalTypeId::DATE, std::make_shared<DateTypeInfo>(DateTypeInfo::DateUnit::DAY));
      case arrow::Type::DATE64:
         return Type(LogicalTypeId::DATE, std::make_shared<DateTypeInfo>(DateTypeInfo::DateUnit::MILLIS));
      case arrow::Type::TIMESTAMP: {
         auto timestampType = std::static_pointer_cast<arrow::TimestampType>(type);
         TimestampTypeInfo::TimestampUnit unit;
         switch (timestampType->unit()) {
            case arrow::TimeUnit::SECOND:
               unit = TimestampTypeInfo::TimestampUnit::SECONDS;
               break;
            case arrow::TimeUnit::MILLI:
               unit = TimestampTypeInfo::TimestampUnit::MILLIS;
               break;
            case arrow::TimeUnit::MICRO:
               unit = TimestampTypeInfo::TimestampUnit::MICROS;
               break;
            case arrow::TimeUnit::NANO:
               unit = TimestampTypeInfo::TimestampUnit::NANOS;
               break;
         }
         std::optional<std::string> timezone = timestampType->timezone().empty() ? std::nullopt : std::optional<std::string>(timestampType->timezone());
         return Type(LogicalTypeId::TIMESTAMP, std::make_shared<TimestampTypeInfo>(timezone, unit));
      }
      case arrow::Type::INTERVAL_MONTHS:
         return Type::intervalMonths();
      case arrow::Type::INTERVAL_DAY_TIME:
         return Type::intervalDaytime();
      case arrow::Type::FIXED_SIZE_BINARY: {
         auto fixedType = std::static_pointer_cast<arrow::FixedSizeBinaryType>(type);
         if (fixedType->byte_width() <= 4) {
            return Type::charType(fixedType->byte_width());
         }
         return Type::stringType();
      }
      case arrow::Type::STRING:
      case arrow::Type::LARGE_STRING:
      case arrow::Type::BINARY:
      case arrow::Type::LARGE_BINARY:
         return Type::stringType();
      default:
         throw std::runtime_error("Catalog parquet bootstrap: unsupported arrow type " + type->ToString());
   }
}

std::vector<std::filesystem::path> findParquetFiles(const std::string& dbDir) {
   std::vector<std::filesystem::path> parquetFiles;
   for (const auto& entry : std::filesystem::directory_iterator(dbDir)) {
      if (!entry.is_regular_file()) {
         continue;
      }
      auto ext = entry.path().extension().string();
      if (ext == ".parquet") {
         parquetFiles.push_back(entry.path());
      }
   }
   std::sort(parquetFiles.begin(), parquetFiles.end());
   return parquetFiles;
}
} // namespace

namespace lingodb::catalog {
Catalog Catalog::deserialize(lingodb::utility::Deserializer& deSerializer) {
   Catalog res;
   auto version = deSerializer.readProperty<size_t>(0);
   if (version != binaryVersion) {
      throw std::runtime_error("Catalog: version mismatch");
   }
   res.entries = deSerializer.readProperty<std::unordered_map<std::string, std::shared_ptr<CatalogEntry>>>(1);
   return res;
}
void Catalog::serialize(lingodb::utility::Serializer& serializer) const {
   serializer.writeProperty(0, binaryVersion);
   serializer.writeProperty(1, entries);
}

void CatalogEntry::serialize(lingodb::utility::Serializer& serializer) const {
   serializer.writeProperty(1, entryType);
   serializeEntry(serializer);
}
std::shared_ptr<CatalogEntry> CatalogEntry::deserialize(lingodb::utility::Deserializer& deserializer) {
   auto entryType = deserializer.readProperty<CatalogEntryType>(1);
   switch (entryType) {
      case CatalogEntryType::INVALID_ENTRY:
         return nullptr;
      case CatalogEntryType::LINGODB_TABLE_ENTRY:
         return LingoDBTableCatalogEntry::deserialize(deserializer);
      case CatalogEntryType::LINGODB_HASH_INDEX_ENTRY:
         return LingoDBHashIndexEntry::deserialize(deserializer);
      case CatalogEntryType::C_FUNCTION_ENTRY:
         return FunctionCatalogEntry::deserialize(deserializer);
      default:
         throw std::runtime_error("deserialize: unknown catalog entry type");
   }
}

void Catalog::insertEntry(std::shared_ptr<CatalogEntry> entry, bool replace) {
   if (entries.contains(entry->getName())) {
      if (replace) {
         entries.erase(entry->getName());
      } else {
         throw std::runtime_error("catalog entry already exists");
      }
   }
   entry->setCatalog(this);
   entry->setDBDir(dbDir);
   entry->setShouldPersist(shouldPersist);
   entries.insert({entry->getName(), std::move(entry)});
}

void Catalog::persist() {
   if (shouldPersist) {
      if (!std::filesystem::exists(dbDir)) {
         throw std::runtime_error("Catalog: dbDir does not exist");
      }
      for (auto& entry : entries) {
         entry.second->flush();
      }
      lingodb::utility::FileByteWriter reader(dbDir + "/db.lingodb");
      lingodb::utility::Serializer serializer(reader);
      serializer.writeProperty(0, *this);
   }
}
std::shared_ptr<Catalog> Catalog::create(std::string dbDir, bool eagerLoading) {
   if (!std::filesystem::exists(dbDir)) {
      std::filesystem::create_directories(dbDir);
   }
   auto parquetFiles = findParquetFiles(dbDir);
   if (!parquetFiles.empty()) {
      auto res = std::make_shared<Catalog>();
      res->dbDir = dbDir;
      for (const auto& parquetFile : parquetFiles) {
         auto inputFile = arrow::io::ReadableFile::Open(parquetFile.string()).ValueOrDie();
         auto parquetFileReader = parquet::ParquetFileReader::Open(inputFile);
         std::unique_ptr<parquet::arrow::FileReader> parquetArrowReader;
         auto status = parquet::arrow::FileReader::Make(arrow::default_memory_pool(), std::move(parquetFileReader), &parquetArrowReader);
         if (!status.ok()) {
            throw std::runtime_error("Catalog parquet bootstrap: failed to open parquet file " + parquetFile.string() + ": " + status.ToString());
         }

         std::shared_ptr<arrow::Schema> parquetSchema;
         status = parquetArrowReader->GetSchema(&parquetSchema);
         if (!status.ok()) {
            throw std::runtime_error("Catalog parquet bootstrap: failed to read schema for " + parquetFile.string() + ": " + status.ToString());
         }

         std::vector<Column> columns;
         columns.reserve(parquetSchema->num_fields());
         for (const auto& field : parquetSchema->fields()) {
            columns.emplace_back(field->name(), mapArrowTypeToCatalogType(field->type()), field->nullable());
         }

         CreateTableDef def{.name = parquetFile.stem().string(), .columns = std::move(columns), .primaryKey = {}};
         auto tableEntry = LingoDBTableCatalogEntry::createFromCreateTable(def, true);
         res->insertEntry(tableEntry);
      }

      return res;
   }

   if (!std::filesystem::exists(dbDir + "/db.lingodb")) {
      auto res = std::make_shared<Catalog>();
      res->dbDir = dbDir;
      return res;
   } else {
      lingodb::utility::FileByteReader reader(dbDir + "/db.lingodb");
      lingodb::utility::Deserializer deserializer(reader);
      auto res = std::make_shared<Catalog>(deserializer.readProperty<Catalog>(0));
      res->dbDir = dbDir;
      for (auto& entry : res->entries) {
         entry.second->setDBDir(dbDir);
         entry.second->setCatalog(&*res);
      }
      if (eagerLoading) {
         for (auto& entry : res->entries) {
            entry.second->ensureFullyLoaded();
         }
      }
      return res;
   }
}
std::shared_ptr<Catalog> Catalog::createEmpty() {
   return std::make_shared<Catalog>();
}

} // namespace lingodb::catalog