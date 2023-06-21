#ifndef RUNTIME_RELATION_H
#define RUNTIME_RELATION_H

#include "metadata.h"
namespace runtime {
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
   virtual std::shared_ptr<arrow::Schema> getArrowSchema() = 0;
   virtual const std::vector<std::shared_ptr<arrow::RecordBatch>>& getRecordBatches() = 0;
   virtual ExternalHashIndexMapping* getHashIndex(const std::vector<std::string>& mapping) = 0;
   static std::shared_ptr<Relation> loadRelation(std::string dbDir, std::string name, std::string json);
   static std::shared_ptr<Relation> createLocalRelation(std::string name, std::shared_ptr<TableMetaData>);
   static std::shared_ptr<Relation> createDBRelation(std::string dbDir, std::string name, std::shared_ptr<TableMetaData>);

   //really load data
   virtual void loadData() = 0;
   virtual void buildIndex() = 0;
   virtual void append(std::shared_ptr<arrow::Table> toAppend) = 0;

   virtual ~Relation(){};
};
} // end namespace runtime

#endif //RUNTIME_RELATION_H
