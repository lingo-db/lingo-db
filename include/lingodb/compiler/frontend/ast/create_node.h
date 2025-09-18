#pragma once
#include "ast_node.h"
#include "constraint.h"
#include "parsed_expression.h"
#include "lingodb/catalog/Catalog.h"

#include <memory>
#include <vector>
namespace lingodb::ast {
enum class TableElementType : uint8_t {
   INVALID = 0,
   COLUMN = 1,
   CONSTRAINT = 2,
};
class TableElement;
class CreateInfo;
class CreateNode : public AstNode {
   public:
   explicit CreateNode(std::shared_ptr<CreateInfo> createInfo) : AstNode(NodeType::CREATE_NODE), createInfo(createInfo) {};
   std::shared_ptr<CreateInfo> createInfo;
};
class CreateInfo {
   public:
   CreateInfo(catalog::CatalogEntry::CatalogEntryType type, std::string catalog, std::string schema, bool temporary)
      : type(type), catalog(std::move(catalog)), schema(std::move(schema)), temporary(temporary) {}

   //! The to-be-created catalog type
   catalog::CatalogEntry::CatalogEntryType type;
   //! The catalog name of the entry
   std::string catalog;
   //! The schema name of the entry
   std::string schema;
   /* //! What to do on create conflict
    OnCreateConflict on_conflict;*/
   //! Whether or not the entry is temporary
   bool temporary;
};

class CreateTableInfo : public CreateInfo {
   public:
   CreateTableInfo(std::string catalog, std::string schema, bool temporary)
      : CreateInfo(catalog::CatalogEntry::CatalogEntryType::LINGODB_TABLE_ENTRY, std::move(catalog), std::move(schema), temporary) {}
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