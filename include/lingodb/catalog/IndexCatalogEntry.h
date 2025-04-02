#ifndef LINGODB_CATALOG_INDEXCATALOGENTRY_H
#define LINGODB_CATALOG_INDEXCATALOGENTRY_H
namespace lingodb::runtime {
class Index;
class LingoDBHashIndex;
} // namespace lingodb::runtime
namespace lingodb::catalog {
class IndexCatalogEntry : public CatalogEntry {
   protected:
   std::string name;
   std::string tableName;
   std::vector<std::string> indexedColumns;

   public:
   static constexpr std::array<CatalogEntryType, 1> entryTypes = {CatalogEntryType::LINGODB_HASH_INDEX_ENTRY};
   IndexCatalogEntry(CatalogEntryType entryType, std::string name, std::string tableName, std::vector<std::string> indexedColumns) : CatalogEntry(entryType), name(name), tableName(tableName), indexedColumns(indexedColumns) {}
   std::string getName() override { return name; }
   std::string getTableName() const { return tableName; }
   std::vector<std::string> getIndexedColumns() const { return indexedColumns; }
   virtual lingodb::runtime::Index& getIndex() = 0;
};
class LingoDBHashIndexEntry : public IndexCatalogEntry {
   std::unique_ptr<lingodb::runtime::LingoDBHashIndex> impl;

   public:
   LingoDBHashIndexEntry(std::string name, std::string tableName, std::vector<std::string> indexedColumns, std::unique_ptr<lingodb::runtime::LingoDBHashIndex> impl);
   static constexpr std::array<CatalogEntryType, 1> entryTypes = {CatalogEntryType::LINGODB_HASH_INDEX_ENTRY};
   void serializeEntry(lingodb::utility::Serializer& serializer) const override;
   static std::shared_ptr<LingoDBHashIndexEntry> deserialize(lingodb::utility::Deserializer& deserializer);
   void setCatalog(Catalog* catalog) override;
   lingodb::runtime::Index& getIndex() override;
   virtual void setShouldPersist(bool shouldPersist) override;
   virtual void setDBDir(std::string dbDir) override;
   static std::shared_ptr<LingoDBHashIndexEntry> createForPrimaryKey(std::string table, std::vector<std::string> primaryKey);
   virtual void flush() override;
   void ensureFullyLoaded() override;
};

} // namespace lingodb::catalog

#endif //LINGODB_CATALOG_INDEXCATALOGENTRY_H
