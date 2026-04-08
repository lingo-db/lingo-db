//
// Created by michael on 3/17/25.
//

#ifndef LINGODB_CATALOG_METADATA_H
#define LINGODB_CATALOG_METADATA_H
#include "lingodb/utility/Sketches.h"

#include <arrow/type_fwd.h>

#include <optional>
namespace lingodb::utility {
class Serializer;
class Deserializer;
} //end namespace lingodb::utility
namespace lingodb::catalog {
class Sample {
   std::shared_ptr<arrow::RecordBatch> sampleData;

   public:
   Sample(std::shared_ptr<arrow::RecordBatch> sampleData) : sampleData(sampleData) {}
   Sample(std::shared_ptr<arrow::Schema> sampleData);
   std::shared_ptr<arrow::RecordBatch> getSampleData() const { return sampleData; }
   void serialize(utility::Serializer& serializer) const;
   static Sample deserialize(utility::Deserializer& deserializer);
   operator bool() {
      return !!sampleData;
   }
};
class ColumnStatistics {
   std::optional<utility::HyperLogLogSketch> hllSketch;

   public:
   ColumnStatistics() = default;
   ColumnStatistics(std::optional<utility::HyperLogLogSketch> hllSketch) : hllSketch(hllSketch) {}
   std::optional<size_t> getNumDistinctValues() {
      if (hllSketch.has_value()) {
         return hllSketch.value().estimate();
      }
      return std::nullopt;
   }
   void merge(std::shared_ptr<arrow::Array> newSegment);
   void serialize(utility::Serializer& serializer) const;
   static ColumnStatistics deserialize(utility::Deserializer& deserializer);
};
class TableMetaDataProvider {
   public:
   virtual const Sample& getSample() const = 0;
   virtual std::vector<std::string> getColumnNames() const = 0;
   virtual const ColumnStatistics& getColumnStatistics(std::string_view column) const = 0;
   virtual size_t getNumRows() const = 0;
   virtual std::vector<std::string> getPrimaryKey() const = 0;
   virtual std::vector<std::pair<std::string, std::vector<std::string>>> getIndices() const = 0;
   void serialize(utility::Serializer& serializer) const;
   static std::shared_ptr<TableMetaDataProvider> deserialize(utility::Deserializer& deserializer);
   virtual ~TableMetaDataProvider() = default;
};
} //end namespace lingodb::catalog

#endif //LINGODB_CATALOG_METADATA_H
