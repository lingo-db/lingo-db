#ifndef LINGODB_RUNTIME_RELATION_H
#define LINGODB_RUNTIME_RELATION_H

#include "Index.h"
#include "metadata.h"
namespace lingodb::runtime {
struct ExternalHashIndexMapping;
class Relation {
   protected:
   std::string name;
   bool persist = false;

   public:
   const std::string& getName() const {
      return name;
   }
   virtual void setPersist(bool persist) {
      Relation::persist = persist;
   }
   virtual std::shared_ptr<TableMetaData> getMetaData() = 0;
   virtual std::shared_ptr<arrow::RecordBatch> getSample() = 0;
   virtual std::shared_ptr<arrow::Table> getTable() = 0;
   virtual std::shared_ptr<arrow::Schema> getArrowSchema() = 0;
   virtual const std::vector<std::shared_ptr<arrow::RecordBatch>>& getRecordBatches() = 0;
   virtual std::shared_ptr<Index> getIndex(const std::string name) = 0;
   static std::shared_ptr<Relation> loadRelation(std::string dbDir, std::string name, std::string json,bool eagerLoading);
   static std::shared_ptr<Relation> createLocalRelation(std::string name, std::shared_ptr<TableMetaData>);
   static std::shared_ptr<Relation> createDBRelation(std::string dbDir, std::string name, std::shared_ptr<TableMetaData>);

   //really load data
   virtual void loadData() = 0;
   virtual void append(std::shared_ptr<arrow::Table> toAppend) = 0;

   virtual ~Relation(){};
};
} // end namespace lingodb::runtime

#endif //LINGODB_RUNTIME_RELATION_H
