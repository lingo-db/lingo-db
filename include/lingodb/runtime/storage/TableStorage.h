#ifndef LINGODB_RUNTIME_STORAGE_TABLESTORAGE_H
#define LINGODB_RUNTIME_STORAGE_TABLESTORAGE_H

#include <functional>
#include <memory>
#include <variant>

#include "lingodb/runtime/ArrowView.h"
#include "lingodb/scheduler/Task.h"
#include "lingodb/utility/Serialization.h"

#include <arrow/type_fwd.h>
namespace lingodb::runtime {
enum class FilterOp : uint8_t {
   EQ,
   NEQ,
   LT,
   LTE,
   GT,
   GTE,
   NOTNULL,
   IN

};
struct FilterDescription {
   std::string columnName;
   size_t columnId;
   FilterOp op;
   std::variant<std::string, int64_t, double> value;
   std::variant<std::vector<std::string>, std::vector<int64_t>, std::vector<double>> values;
   bool operator==(const FilterDescription& other) const noexcept {
      return columnName == other.columnName &&
         columnId == other.columnId &&
         op == other.op &&
         value == other.value && values == other.values;
   }
   void serialize(utility::Serializer& serializer) const {
      serializer.writeProperty(0, columnName);
      serializer.writeProperty(1, columnId);
      serializer.writeProperty(2, op);
      serializer.writeProperty(3, value.index());
      std::visit([&](auto const& v) {
         serializer.writeProperty(4, v);
      },
                 value);
      serializer.writeProperty(5, values.index());
      std::visit([&](auto const& v) {
         serializer.writeProperty(6, v);
      },
                 values);
   }

   static FilterDescription deserialize(lingodb::utility::Deserializer& deserializer) {
      FilterDescription desc{};
      desc.columnName = deserializer.readProperty<std::string>(0);
      desc.columnId = deserializer.readProperty<size_t>(1);
      desc.op = deserializer.readProperty<FilterOp>(2);
      switch (deserializer.readProperty<size_t>(3)) {
         case 0:
            desc.value = deserializer.readProperty<std::string>(4);
            break;
         case 1:
            desc.value = deserializer.readProperty<int64_t>(4);
            break;
         default:
            desc.value = deserializer.readProperty<double>(4);
      }
      switch (deserializer.readProperty<size_t>(5)) {
         case 0:
            desc.values = deserializer.readProperty<std::vector<std::string>>(6);
            break;
         case 1:
            desc.values = deserializer.readProperty<std::vector<int64_t>>(6);
            break;
         default:
            desc.values = deserializer.readProperty<std::vector<double>>(6);
      }
      return desc;
   }
};
struct ScanConfig {
   bool parallel;
   std::vector<std::string> columns;
   std::vector<FilterDescription> filters;
   std::function<void(lingodb::runtime::BatchView*)> cb;
};
class TableStorage {
   public:
   virtual std::shared_ptr<arrow::DataType> getColumnStorageType(std::string_view columnName) const = 0;
   virtual std::unique_ptr<scheduler::Task> createScanTask(const ScanConfig& scanConfig) = 0;
   virtual void append(const std::vector<std::shared_ptr<arrow::RecordBatch>>& toAppend) = 0;
   virtual size_t nextRowId() = 0;
   virtual void append(const std::shared_ptr<arrow::Table>& toAppend) = 0;
   virtual ~TableStorage() = default;
};
} // namespace lingodb::runtime
namespace std {

template <>
struct hash<lingodb::runtime::FilterDescription> {
   size_t operator()(const lingodb::runtime::FilterDescription& f) const noexcept {
      size_t h = 0;

      // helper lambda to combine hashes
      auto combine = [&h](auto const& v) {
         std::hash<std::decay_t<decltype(v)>> hasher;
         h ^= hasher(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
      };

      combine(f.columnName);
      combine(f.columnId);
      combine(static_cast<std::underlying_type_t<lingodb::runtime::FilterOp>>(f.op));

      std::visit([&](auto const& val) {
         combine(val);
      },
                 f.value);
      std::visit([&](auto const& val) {
         for (const auto& v : val) {
            combine(v);
         }
      },
                 f.values);
      return h;
   }
};

} // namespace std

#endif //LINGODB_RUNTIME_STORAGE_TABLESTORAGE_H
