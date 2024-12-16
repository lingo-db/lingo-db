#include "lingodb/runtime/Catalog.h"

#include "json.h"
#include <filesystem>
#include <fstream>

namespace lingodb::runtime {
class EmptyCatalog : public Catalog {
   std::shared_ptr<Relation> findRelation(std::string name) override {
      return {};
   }

   public:
   void addTable(std::string tableName, std::shared_ptr<TableMetaData> mD) override {
      throw std::runtime_error("can not add table");
   }
   void setPersist(bool value) override {
      if (value) {
         throw std::runtime_error("can not persist");
      }
   }
};

std::shared_ptr<Relation> LocalCatalog::findRelation(std::string name) {
   if (relations.contains(name)) {
      return relations.at(name);
   }
   return nested->findRelation(name);
}
std::shared_ptr<LocalCatalog> LocalCatalog::create(std::shared_ptr<Catalog> nested) {
   return std::shared_ptr<LocalCatalog>(new LocalCatalog(nested));
}
void LocalCatalog::addTable(std::string tableName, std::shared_ptr<TableMetaData> mD) {
   if (relations.contains(tableName)) {
      throw std::runtime_error("can not insert: table does already exist");
   }
   relations[tableName] = Relation::createLocalRelation(tableName, mD);
}
void LocalCatalog::setPersist(bool value) {
   if (value) {
      throw std::runtime_error("can not persist");
   }
}
std::shared_ptr<Catalog> Catalog::createEmpty() {
   return std::make_shared<EmptyCatalog>();
}
std::shared_ptr<DBCatalog> DBCatalog::create(std::shared_ptr<Catalog> nested, std::string dbDir,bool eagerLoading) {
   auto* catalog = new DBCatalog(nested);
   catalog->dbDirectory = dbDir;
   for (const auto& p : std::filesystem::directory_iterator(dbDir)) {
      auto path = p.path();
      if (path.extension().string() == ".json" && path.stem().string().ends_with(".metadata")) {
         auto tableName = path.stem().string().substr(0, path.stem().string().size() - std::string(".metadata").size());
         std::ifstream t(path);
         auto json = std::string((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
         catalog->relations.insert({tableName, Relation::loadRelation(dbDir, tableName, json,eagerLoading)});
      }
   }
   return std::shared_ptr<DBCatalog>(catalog);
}
std::shared_ptr<Relation> DBCatalog::findRelation(std::string name) {
   if (relations.contains(name)) {
      return relations.at(name);
   }
   return nested->findRelation(name);
}
void DBCatalog::addTable(std::string tableName, std::shared_ptr<TableMetaData> mD) {
   if (relations.contains(tableName)) {
      throw std::runtime_error("can not insert: table does already exist");
   }
   relations[tableName] = Relation::createDBRelation(dbDirectory, tableName, mD);
   relations[tableName]->setPersist(persist);
}
void DBCatalog::setPersist(bool value) {
   persist = value;
   for (auto rel : relations) {
      rel.second->setPersist(value);
   }
}
} // end namespace lingodb::runtime