#include "lingodb/catalog/Catalog.h"

#include "lingodb/catalog/FunctionCatalogEntry.h"
#include "lingodb/catalog/IndexCatalogEntry.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/catalog/Types.h"
#include "lingodb/utility/Serialization.h"

#include <filesystem>

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
      case CatalogEntryType::PYTHON_FUNCTION_ENTRY:
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