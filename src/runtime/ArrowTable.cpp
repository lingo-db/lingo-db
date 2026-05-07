#include "lingodb/runtime/ArrowTable.h"

#include "lingodb/catalog/Types.h"
#include "lingodb/utility/Serialization.h"
#include "lingodb/utility/Tracer.h"

#include <arrow/table.h>
#include <lingodb/runtime/ExecutionContext.h>

#include <sstream>
using namespace lingodb::runtime;
namespace {
lingodb::utility::Tracer::Event tableMerge("ArrowTable", "merge");
} // end namespace

std::shared_ptr<arrow::DataType> lingodb::runtime::arrowPhysicalTypeFor(lingodb::catalog::Type t) {
   using TypeId = lingodb::catalog::LogicalTypeId;
   switch (t.getTypeId()) {
      case TypeId::BOOLEAN:
         return arrow::boolean();
      case TypeId::INT:
         switch (t.getInfo<lingodb::catalog::IntTypeInfo>()->getBitWidth()) {
            case 8: return arrow::int8();
            case 16: return arrow::int16();
            case 32: return arrow::int32();
            case 64: return arrow::int64();
            default: throw std::runtime_error("unsupported int bit width");
         }
      case TypeId::FLOAT:
         return arrow::float32();
      case TypeId::DOUBLE:
         return arrow::float64();
      case TypeId::DECIMAL:
         return arrow::decimal128(t.getInfo<lingodb::catalog::DecimalTypeInfo>()->getPrecision(),
                                  t.getInfo<lingodb::catalog::DecimalTypeInfo>()->getScale());
      case TypeId::DATE: {
         auto dateUnit = t.getInfo<lingodb::catalog::DateTypeInfo>()->getUnit();
         switch (dateUnit) {
            case lingodb::catalog::DateTypeInfo::DateUnit::DAY: return arrow::date32();
            case lingodb::catalog::DateTypeInfo::DateUnit::MILLIS: return arrow::date64();
         }
         throw std::runtime_error("unsupported date unit");
      }
      case TypeId::TIMESTAMP: {
         arrow::TimeUnit::type timeUnit;
         switch (t.getInfo<lingodb::catalog::TimestampTypeInfo>()->getUnit()) {
            case lingodb::catalog::TimestampTypeInfo::TimestampUnit::NANOS: timeUnit = arrow::TimeUnit::NANO; break;
            case lingodb::catalog::TimestampTypeInfo::TimestampUnit::MICROS: timeUnit = arrow::TimeUnit::MICRO; break;
            case lingodb::catalog::TimestampTypeInfo::TimestampUnit::MILLIS: timeUnit = arrow::TimeUnit::MILLI; break;
            case lingodb::catalog::TimestampTypeInfo::TimestampUnit::SECONDS: timeUnit = arrow::TimeUnit::SECOND; break;
            default: throw std::runtime_error("unsupported timestamp unit");
         }
         return arrow::timestamp(timeUnit);
      }
      case TypeId::INTERVAL: {
         switch (t.getInfo<lingodb::catalog::IntervalTypeInfo>()->getUnit()) {
            case lingodb::catalog::IntervalTypeInfo::IntervalUnit::DAYTIME: return arrow::day_time_interval();
            case lingodb::catalog::IntervalTypeInfo::IntervalUnit::MONTH: return arrow::month_interval();
         }
         throw std::runtime_error("unsupported interval unit");
      }
      case TypeId::CHAR: {
         if (t.getInfo<lingodb::catalog::CharTypeInfo>()->getLength() == 1) {
            return arrow::fixed_size_binary(4);
         }
         return arrow::utf8();
      }
      case TypeId::STRING:
         return arrow::utf8();
      default:
         throw std::runtime_error("arrowPhysicalTypeFor: unsupported type");
   }
}

ArrowTable* ArrowTable::createEmpty() {
   auto table = arrow::Table::MakeEmpty(std::make_shared<arrow::Schema>(std::vector<std::shared_ptr<arrow::Field>>{})).ValueOrDie();
   auto* t = new ArrowTable(table);
   getCurrentExecutionContext()->registerState({t, [](void* ptr) { delete reinterpret_cast<ArrowTable*>(ptr); }});
   return t;
}
ArrowTable* ArrowTable::addColumn(VarLen32 name, ArrowColumn* column) {
   auto fields = this->table->schema()->fields();
   fields.push_back(arrow::field(name, column->getColumn()->type()));
   auto schema = std::make_shared<arrow::Schema>(fields);
   auto arrays = this->table->columns();
   arrays.push_back(column->getColumn());
   auto* t = new ArrowTable(arrow::Table::Make(schema, arrays));
   getCurrentExecutionContext()->registerState({t, [](void* ptr) { delete reinterpret_cast<ArrowTable*>(ptr); }});
   return t;
}
ArrowTable* ArrowTable::merge(ThreadLocal* threadLocal) {
   utility::Tracer::Trace trace(tableMerge);
   std::vector<std::shared_ptr<arrow::Table>> tables;
   for (auto* ptr : threadLocal->getThreadLocalValues<ArrowTable>()) {
      if (!ptr) continue;
      auto* current = ptr;
      tables.push_back(current->get());
   }
   auto concatenated = arrow::ConcatenateTables(tables).ValueOrDie();
   trace.stop();
   auto* t = new ArrowTable(concatenated);
   getCurrentExecutionContext()->registerState({t, [](void* ptr) { delete reinterpret_cast<ArrowTable*>(ptr); }});
   return t;
}
ArrowColumn* ArrowTable::getColumn(VarLen32 name) {
   auto* c = new ArrowColumn(get()->GetColumnByName(name.str()));
   getCurrentExecutionContext()->registerState({c, [](void* ptr) { delete reinterpret_cast<ArrowColumn*>(ptr); }});
   return c;
}

ArrowTable* ArrowTable::verifySchema(ArrowTable* table, VarLen32 descriptor) {
   if (!table) {
      throw std::runtime_error("verifySchema: null arrow table (did the UDF return None?)");
   }
   auto expected = lingodb::utility::deserializeFromHexString<std::vector<std::pair<std::string, lingodb::catalog::Type>>>(descriptor.str());
   const auto& schema = table->get()->schema();
   if (static_cast<size_t>(schema->num_fields()) != expected.size()) {
      std::ostringstream actual;
      for (int i = 0; i < schema->num_fields(); ++i) {
         if (i) actual << ", ";
         actual << schema->field(i)->name() << ":" << schema->field(i)->type()->ToString();
      }
      std::ostringstream want;
      for (size_t i = 0; i < expected.size(); ++i) {
         if (i) want << ", ";
         want << expected[i].first << ":" << arrowPhysicalTypeFor(expected[i].second)->ToString();
      }
      throw std::runtime_error(
         "tabular UDF returned a table with " + std::to_string(schema->num_fields()) +
         " column(s); declared schema has " + std::to_string(expected.size()) +
         " column(s).\n  declared: " + want.str() +
         "\n  actual:   " + actual.str());
   }
   for (size_t i = 0; i < expected.size(); ++i) {
      const auto& [expectedName, expectedLogical] = expected[i];
      auto field = schema->field(i);
      auto expectedArrow = arrowPhysicalTypeFor(expectedLogical);
      if (field->name() != expectedName) {
         throw std::runtime_error(
            "tabular UDF returned column #" + std::to_string(i + 1) +
            " named '" + field->name() + "'; declared schema expected '" + expectedName + "'");
      }
      if (!field->type()->Equals(*expectedArrow)) {
         throw std::runtime_error(
            "tabular UDF column '" + expectedName + "' has Arrow type '" + field->type()->ToString() +
            "'; declared schema expected '" + expectedArrow->ToString() +
            "' (hint: pass `schema=` to pa.Table.from_pandas to pin types — pyarrow's default for object/string columns is large_string, not string)");
      }
   }
   return table;
}
