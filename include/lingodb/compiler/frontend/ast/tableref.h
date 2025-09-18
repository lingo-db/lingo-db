#pragma once
#include "ast_node.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "table_producer.h"

#include <cstdint>
#include <string>
#include <variant>
namespace lingodb::ast {
class QueryNode;
class ParsedExpression;
class ColumnRefExpression;
enum class TableReferenceType : uint8_t;

class TableRef : public TableProducer {
   public:
   explicit TableRef(TableReferenceType type) : TableProducer(NodeType::TABLE_REF), type(type) {
   }
   TableReferenceType type;
   std::string alias;

   virtual std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) = 0;
};

enum class TableReferenceType : uint8_t {
   INVALID = 0, // invalid table reference type
   BASE_TABLE = 1, // base table reference
   SUBQUERY = 2, // output of a subquery
   JOIN = 3, // output of join
   TABLE_FUNCTION = 5, // table producing function
   EXPRESSION_LIST = 6, // expression list
   CTE = 7, // Recursive CTE
   EMPTY_FROM = 8, // placeholder for empty FROM
   PIVOT = 9, // pivot statement
   SHOW_REF = 10, // SHOW statement
   COLUMN_DATA = 11, // column data collection
   DELIM_GET = 12, // Delim get ref
   BOUND_EXPRESSION_LIST = 13,
   CROSS_PRODUCT = 14, //crossproducts: FROM x,y,..
};
class TableDescription {
   public:
   TableDescription(std::string database, std::string schema, std::string table) : database(database), schema(schema), table(table) {};

   std::string database;
   std::string schema;
   std::string table;
   bool readonly = true;
};
class BaseTableRef : public TableRef {
   public:
   static constexpr TableReferenceType TYPE = TableReferenceType::BASE_TABLE;
   BaseTableRef(TableDescription tableDescription);

   //! The catalog name.
   std::string catalogName;
   //! The schema name.
   std::string schemaName;
   //! The table name.
   std::string tableName;
   //! The timestamp/version at which to read this table entry (if any)

   /*
    * Semantic
   */
   std::shared_ptr<catalog::TableCatalogEntry> catalogEntry = nullptr;
   std::string scopeName;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};

enum class JoinCondType : uint8_t {
   REGULAR, // Explicit conditions
   NATURAL, // Implied conditions
   CROSS, // No condition
   POSITIONAL, // Positional condition
   ASOF, // AsOf conditions
   DEPENDENT, // Dependent join conditions
};
enum class JoinType : uint8_t {
   INVALID = 0, // invalid join type
   LEFT = 1, // left
   RIGHT = 2, // right
   INNER = 3, // inner
   OUTER = 4, // outer
   SEMI = 5, // LEFT SEMI join returns left side row ONLY if it has a join partner, no duplicates.
   ANTI = 6, // LEFT ANTI join returns left side row ONLY if it has NO join partner, no duplicates
   MARK = 7, // MARK join returns marker indicating whether or not there is a join partner (true), there is no join
   // partner (false)
   SINGLE = 8, // SINGLE join is like LEFT OUTER JOIN, BUT returns at most one join partner per entry on the LEFT side
   // (and NULL if no partner is found)
   RIGHT_SEMI = 9, // RIGHT SEMI join is created by the optimizer when the children of a semi join need to be switched
   // so that the build side can be the smaller table
   RIGHT_ANTI = 10, // RIGHT ANTI join is created by the optimizer when the children of an anti join need to be
   // switched so that the build side can be the smaller table
   FULL = 12,
};
using jointCondOrUsingCols = std::variant<std::shared_ptr<ParsedExpression>, std::vector<std::shared_ptr<ColumnRefExpression>>>;
class JoinRef : public TableRef {
   static constexpr TableReferenceType TYPE = TableReferenceType::JOIN;

   public:
   JoinRef(JoinType type, JoinCondType refType);

   //! The left hand side of the join
   //! QueryNode as variant is needed for pipe syntax. Example: FROM Test |> join ok on id1=id2
   std::shared_ptr<TableProducer> left;
   //! The right hand side of the join
   std::shared_ptr<TableProducer> right;

   std::vector<std::shared_ptr<ast::TableProducer>> rights;
   //! The joint condition or a vector of ColumnRefExpression if USING
   jointCondOrUsingCols condition;
   //! The join type
   JoinType type;
   //! Join condition type
   JoinCondType refType;


   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};

class CrossProductRef : public TableRef {
   static constexpr TableReferenceType TYPE = TableReferenceType::CROSS_PRODUCT;
   public:
   CrossProductRef();
   std::vector<std::shared_ptr<TableProducer>> tables;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};

class SubqueryRef : public TableRef {
   static constexpr TableReferenceType TYPE = TableReferenceType::SUBQUERY;

   public:
   SubqueryRef(std::shared_ptr<TableProducer> subSelectNode);

   //! The subquery
   std::shared_ptr<TableProducer> subSelectNode;
   std::vector<std::string> columnNames;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};

/**
 * Naming consistent with that of DuckDb.
 * Represents an expression list generated by a VALUES statement.
 */
class ExpressionListRef : public TableRef {
   public:
   static constexpr TableReferenceType TYPE = TableReferenceType::EXPRESSION_LIST;
   ExpressionListRef(std::vector<std::vector<std::shared_ptr<ParsedExpression>>> values);

   //! The expressions in the list
   std::vector<std::vector<std::shared_ptr<ParsedExpression>>> values;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};

} // namespace lingodb::ast