#ifndef LINGODB_CATALOG_CATALOG_H
#define LINGODB_CATALOG_CATALOG_H
#include <memory>
#include <optional>
#include <unordered_map>
namespace lingodb::utility {
class Serializer;
class Deserializer;
} // namespace lingodb::utility
namespace lingodb::catalog {

class CatalogEntry {
   public:
   enum class CatalogEntryType : uint8_t {
      INVALID_ENTRY = 0,
      TABLE_ENTRY = 1,
   };

   protected:
   CatalogEntryType entryType;
   CatalogEntry(CatalogEntryType entryType) : entryType(entryType) {}

   public:
   CatalogEntryType getEntryType() { return entryType; }
   virtual std::string getName() = 0;
   void serialize(lingodb::utility::Serializer& serializer) const;
   static std::unique_ptr<CatalogEntry> deserialize(lingodb::utility::Deserializer& deSerializer);
   virtual void serializeEntry(lingodb::utility::Serializer& serializer) const = 0;
   virtual ~CatalogEntry() = default;
};

class Catalog {
   public:
   void serialize(lingodb::utility::Serializer& serializer) const;
   static std::unique_ptr<Catalog> deserialize(lingodb::utility::Deserializer& deSerializer);

   std::optional<CatalogEntry*> getEntry(std::string name) {
      if (entries.contains(name)) {
         return entries.at(name).get();
      } else {
         return std::nullopt;
      }
   }

   void insertEntry(std::unique_ptr<CatalogEntry> entry) {
      entries.insert({entry->getName(), std::move(entry)});
   }

   private:
   std::unordered_map<std::string, std::unique_ptr<CatalogEntry>> entries;
};

} // namespace lingodb::catalog

#endif //LINGODB_CATALOG_CATALOG_H
