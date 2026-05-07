#ifndef LINGODB_CATALOG_FUNCTIONCATALOGENTRY_H
#define LINGODB_CATALOG_FUNCTIONCATALOGENTRY_H

#include "Catalog.h"
#include "Column.h"

#include <mlir/IR/BuiltinOps.h.inc>
#include <vector>
namespace lingodb::catalog {
// Scalar UDFs (C or Python) — one row in, one value out. Tabular UDFs live in
// TableFunctionCatalogEntry; the two never share a base because the surface
// (return type vs. return-schema, scalar-call vs. table-function-call) and the
// places they're consumed (expression analysis vs. FROM-clause analysis) are
// disjoint.
class FunctionCatalogEntry : public CatalogEntry {
   protected:
   std::string name;
   std::string code;
   Type returnType;
   std::vector<Type> argumentTypes;

   public:
   static constexpr std::array<CatalogEntryType, 2> entryTypes = {CatalogEntryType::C_FUNCTION_ENTRY, CatalogEntryType::PYTHON_FUNCTION_ENTRY};
   struct UDFHandle {
      void* handle;
      void* addrPtr;
   };
   static std::unordered_map<std::string, UDFHandle>& getUdfFunctions() {
      static std::unordered_map<std::string, UDFHandle> udfFunctions;
      return udfFunctions;
   }
   FunctionCatalogEntry(CatalogEntryType entryType, std::string name, std::string code, Type returnType, std::vector<Type> argumentTypes)
      : CatalogEntry(entryType), name(std::move(name)), code(std::move(code)), returnType(std::move(returnType)), argumentTypes(std::move(argumentTypes)) {}
   std::string getName() override { return name; }
   [[nodiscard]] std::string getCode() const { return code; }
   [[nodiscard]] Type getReturnType() const { return returnType; }
   [[nodiscard]] std::vector<Type> getArgumentTypes() const { return argumentTypes; }
   void serializeEntry(lingodb::utility::Serializer& serializer) const override;

   static std::shared_ptr<FunctionCatalogEntry> deserialize(lingodb::utility::Deserializer& deserializer);
};

class CFunctionCatalogEntry : public FunctionCatalogEntry {
   public:
   CFunctionCatalogEntry(std::string name, std::string code, Type returnType, std::vector<Type> argumentTypes)
      : FunctionCatalogEntry(CatalogEntryType::C_FUNCTION_ENTRY, name, code, returnType, argumentTypes) {}

   static std::shared_ptr<FunctionCatalogEntry> deserialize(lingodb::utility::Deserializer& deserializer);
};

class PythonFunctionCatalogEntry : public FunctionCatalogEntry {
   public:
   PythonFunctionCatalogEntry(std::string name, std::string code, Type returnType, std::vector<Type> argumentTypes)
      : FunctionCatalogEntry(CatalogEntryType::PYTHON_FUNCTION_ENTRY, name, code, returnType, argumentTypes) {}

   static std::shared_ptr<FunctionCatalogEntry> deserialize(lingodb::utility::Deserializer& deserializer);
};

// One declared input table in a tabular UDF: the parameter name and its
// per-column (name, type) schema. A signature can list one or more.
struct TableFunctionInput {
   std::string name;
   std::vector<std::pair<std::string, Type>> columns;

   void serialize(lingodb::utility::Serializer& serializer) const;
   static TableFunctionInput deserialize(lingodb::utility::Deserializer& deserializer);
};

// Tabular UDFs (currently Python-only) — one or more input tables in, one
// output table out. No `returnType` here; the input/output schemas live in
// inputTables / returnColumns.
class TableFunctionCatalogEntry : public CatalogEntry {
   std::string name;
   std::string language;
   std::string code;
   std::vector<TableFunctionInput> inputTables;
   std::vector<Type> argumentTypes; // declared scalar arguments (after the input tables)
   std::vector<std::pair<std::string, Type>> returnColumns;

   public:
   static constexpr std::array<CatalogEntryType, 1> entryTypes = {CatalogEntryType::TABLE_FUNCTION_ENTRY};

   TableFunctionCatalogEntry(std::string name, std::string language, std::string code,
                             std::vector<TableFunctionInput> inputTables,
                             std::vector<Type> argumentTypes,
                             std::vector<std::pair<std::string, Type>> returnColumns)
      : CatalogEntry(CatalogEntryType::TABLE_FUNCTION_ENTRY),
        name(std::move(name)), language(std::move(language)), code(std::move(code)),
        inputTables(std::move(inputTables)),
        argumentTypes(std::move(argumentTypes)), returnColumns(std::move(returnColumns)) {}

   std::string getName() override { return name; }
   [[nodiscard]] const std::string& getLanguage() const { return language; }
   [[nodiscard]] const std::string& getCode() const { return code; }
   [[nodiscard]] const std::vector<TableFunctionInput>& getInputTables() const { return inputTables; }
   [[nodiscard]] const std::vector<Type>& getArgumentTypes() const { return argumentTypes; }
   [[nodiscard]] const std::vector<std::pair<std::string, Type>>& getReturnColumns() const { return returnColumns; }

   void serializeEntry(lingodb::utility::Serializer& serializer) const override;
   static std::shared_ptr<TableFunctionCatalogEntry> deserialize(lingodb::utility::Deserializer& deserializer);
};

void visitUDFFunctions(const std::function<void(std::string, void*)>& fn);
} // namespace lingodb::catalog
#endif
