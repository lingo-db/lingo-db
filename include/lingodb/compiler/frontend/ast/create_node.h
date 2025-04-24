#pragma once
#include "ast_node.h"
#include "constraint.h"
#include "parsed_expression.h"

#include <memory>
#include <vector>
namespace lingodb::ast {
enum class CatalogType : uint8_t {
   INVALID = 0,
   TABLE_ENTRY = 1,
   SCHEMA_ENTRY = 2,
   VIEW_ENTRY = 3,
   INDEX_ENTRY = 4,
   PREPARED_STATEMENT = 5,
   SEQUENCE_ENTRY = 6,
   COLLATION_ENTRY = 7,
   TYPE_ENTRY = 8,
   DATABASE_ENTRY = 9,

   // functions
   TABLE_FUNCTION_ENTRY = 25,
   SCALAR_FUNCTION_ENTRY = 26,
   AGGREGATE_FUNCTION_ENTRY = 27,
   PRAGMA_FUNCTION_ENTRY = 28,
   COPY_FUNCTION_ENTRY = 29,
   MACRO_ENTRY = 30,
   TABLE_MACRO_ENTRY = 31,

   // version info
   DELETED_ENTRY = 51,
   RENAMED_ENTRY = 52,

   // secrets
   SECRET_ENTRY = 71,
   SECRET_TYPE_ENTRY = 72,
   SECRET_FUNCTION_ENTRY = 73,

   // dependency info
   DEPENDENCY_ENTRY = 100

};
enum class TableElementType : uint8_t {
   INVALID = 0,
   COLUMN = 1,
   CONSTRAINT = 2,
};
class TableElement;
class CreateInfo;
class CreateNode : public AstNode {
   public:
   explicit  CreateNode(std::shared_ptr<CreateInfo>  createInfo);
   std::shared_ptr<CreateInfo> createInfo;


   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};
class CreateInfo {
   public:
   CreateInfo(CatalogType type, std::string catalog, std::string schema, bool temporary)
         : type(type), catalog(std::move(catalog)), schema(std::move(schema)), temporary(temporary) {}


   //! The to-be-created catalog type
   CatalogType type;
   //! The catalog name of the entry
   std::string catalog;
   //! The schema name of the entry
   std::string schema;
   /* //! What to do on create conflict
    OnCreateConflict on_conflict;*/
   //! Whether or not the entry is temporary
   bool temporary;
   /* //! Whether or not the entry is an internal entry
   bool internal;
   //! The SQL string of the CREATE statement
   string sql;
   //! The inherent dependencies of the created entry
   LogicalDependencyList dependencies;
   //! User provided comment
   Value comment;
   //! Key-value tags with additional metadata
   unordered_map<string, string> tags;*/
};

class CreateTableInfo : public CreateInfo {
   public:
   CreateTableInfo(std::string catalog, std::string schema, bool temporary)
         : CreateInfo(CatalogType::TABLE_ENTRY, std::move(catalog), std::move(schema), temporary) {}
   //! The table name
   std::string tableName;

   std::vector<std::shared_ptr<TableElement>> tableElements;
};

class TableElement {
   public:
   TableElement(TableElementType type) : type(type) {}
   TableElementType type;


   location loc;
};

class ColumnElement : public TableElement {
   public:
   ColumnElement(std::string name, LogicalTypeWithMods typeMods)
         : TableElement(TableElementType::COLUMN), name(std::move(name)), logicalTypeWithMods(typeMods) {}

   LogicalTypeWithMods logicalTypeWithMods;
   std::string name;
   std::vector<std::shared_ptr<Constraint>> constraints; //Optional constraints for the column
};

class TableConstraintElement : public TableElement {
   public:
   TableConstraintElement(std::shared_ptr<Constraint> constraint)
         : TableElement(TableElementType::CONSTRAINT), constraint(std::move(constraint)) {}

   std::shared_ptr<Constraint> constraint;





};
}