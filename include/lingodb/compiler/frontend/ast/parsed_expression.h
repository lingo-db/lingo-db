#pragma once
#include "constant_value.h"
#include "lingodb/catalog/Column.h"
#include "lingodb/catalog/Types.h"
#include "ast_node.h"
#include "table_producer.h"

#include <string>
#include <variant>
#include <vector>
namespace lingodb::ast {
class OrderByModifier;
enum class ExpressionType : uint8_t;
enum class ExpressionClass : uint8_t;
enum LogicalType : uint8_t {
   INVALID = 0,
   BOOLEAN = 1,
   CHAR = 2,
   STRING = 3,
   DATE = 4,
   INTERVAL = 5,
   DAYS = 6,
   YEARS = 7,
   SMALLINT = 8,
   INT = 9,
   BIGINT = 10,
   DECIMAL= 11,
   TIMESTAMP = 12,
   FLOAT4 = 13,
   FLOAT8 = 14,
   //TODO other
};
class LogicalTypeWithMods {
   public:
   LogicalTypeWithMods() : LogicalTypeWithMods(LogicalType::INVALID) {}
   LogicalTypeWithMods(LogicalType logicalType)
                : logicalType(logicalType), typeModifiers({}) {}
        LogicalTypeWithMods(LogicalType logicalType, std::vector<std::shared_ptr<Value>> typeModifiers)
                : logicalType(logicalType), typeModifiers(std::move(typeModifiers)) {}
   LogicalType logicalType;
   std::vector<std::shared_ptr<Value>> typeModifiers;
};

class BaseExpression : public AstNode {
   public:
   BaseExpression(ExpressionType type, ExpressionClass exprClass) : AstNode(NodeType::EXPRESSION), type(type), exprClass(exprClass) {}

   ExpressionClass exprClass;
   ExpressionType type;
   //! The alias of the expression,
   std::string alias;
};
//Extracted from PostgresSQL
//===--------------------------------------------------------------------===//
// Predicate Expression Operation Types
//===--------------------------------------------------------------------===//
enum class ExpressionType : uint8_t {
   INVALID = 0,

   // explicitly cast left as right (right is integer in ValueType enum)
   OPERATOR_CAST = 12,
   // logical not operator
   OPERATOR_NOT = 13,
   // is null operator
   OPERATOR_IS_NULL = 14,
   // is not null operator
   OPERATOR_IS_NOT_NULL = 15,
   // unpack operator
   OPERATOR_UNPACK = 16,
   OPERATOR_MINUS = 17,
   OPERATOR_PLUS = 18,
   OPERATOR_TIMES = 19,
   OPERATOR_DIVIDE = 20,
   OPERATOR_MOD = 21,

   // -----------------------------
   // Comparison Operators
   // -----------------------------
   // equal operator between left and right
   COMPARE_EQUAL = 25,
   // compare initial boundary
   COMPARE_BOUNDARY_START = COMPARE_EQUAL,
   // inequal operator between left and right
   COMPARE_NOTEQUAL = 26,
   // less than operator between left and right
   COMPARE_LESSTHAN = 27,
   // greater than operator between left and right
   COMPARE_GREATERTHAN = 28,
   // less than equal operator between left and right
   COMPARE_LESSTHANOREQUALTO = 29,
   // greater than equal operator between left and right
   COMPARE_GREATERTHANOREQUALTO = 30,
   // IN operator [left IN (right1, right2, ...)]
   COMPARE_IN = 35,
   // NOT IN operator [left NOT IN (right1, right2, ...)]
   COMPARE_NOT_IN = 36,
   // IS DISTINCT FROM operator
   COMPARE_DISTINCT_FROM = 37,
   // LIKE
   COMPARE_LIKE = 38,
   // NOT LIKE
   COMPARE_NOT_LIKE = 39,

   COMPARE_BETWEEN = 40,
   COMPARE_NOT_BETWEEN = 41,
   // IS NOT DISTINCT FROM operator
   COMPARE_NOT_DISTINCT_FROM = 42,
   // compare final boundary
   COMPARE_BOUNDARY_END = COMPARE_NOT_DISTINCT_FROM,

   // -----------------------------
   // Conjunction Operators
   // -----------------------------
   CONJUNCTION_AND = 50,
   CONJUNCTION_OR = 51,

   // -----------------------------
   // Values
   // -----------------------------
   VALUE_CONSTANT = 75,
   VALUE_PARAMETER = 76,
   VALUE_TUPLE = 77,
   VALUE_TUPLE_ADDRESS = 78,
   VALUE_NULL = 79,
   VALUE_VECTOR = 80,
   VALUE_SCALAR = 81,
   VALUE_DEFAULT = 82,

   // -----------------------------
   // Aggregates
   // -----------------------------
   AGGREGATE = 100,
   BOUND_AGGREGATE = 101,
   GROUPING_FUNCTION = 102,

   // -----------------------------
   // Window Functions
   // -----------------------------
   WINDOW_AGGREGATE = 110,

   WINDOW_RANK = 120,
   WINDOW_RANK_DENSE = 121,
   WINDOW_NTILE = 122,
   WINDOW_PERCENT_RANK = 123,
   WINDOW_CUME_DIST = 124,
   WINDOW_ROW_NUMBER = 125,

   WINDOW_FIRST_VALUE = 130,
   WINDOW_LAST_VALUE = 131,
   WINDOW_LEAD = 132,
   WINDOW_LAG = 133,
   WINDOW_NTH_VALUE = 134,

   // -----------------------------
   // Functions
   // -----------------------------
   FUNCTION = 140,
   BOUND_FUNCTION = 141,

   // -----------------------------
   // Operators
   // -----------------------------
   CASE_EXPR = 150,
   OPERATOR_NULLIF = 151,
   OPERATOR_COALESCE = 152,
   ARRAY_EXTRACT = 153,
   ARRAY_SLICE = 154,
   STRUCT_EXTRACT = 155,
   ARRAY_CONSTRUCTOR = 156,
   ARROW = 157,
   OPERATOR_TRY = 158,

   // -----------------------------
   // Subquery IN/EXISTS
   // -----------------------------
   SUBQUERY = 175,

   // -----------------------------
   // Parser
   // -----------------------------
   STAR = 200,
   TABLE_STAR = 201,
   PLACEHOLDER = 202,
   COLUMN_REF = 203,
   FUNCTION_REF = 204,
   TABLE_REF = 205,
   LAMBDA_REF = 206,

   // -----------------------------
   // Miscellaneous
   // -----------------------------
   CAST = 225,
   BOUND_REF = 227,
   BOUND_COLUMN_REF = 228,
   BOUND_UNNEST = 229,
   COLLATE = 230,
   LAMBDA = 231,
   POSITIONAL_REFERENCE = 232,
   BOUND_LAMBDA_REF = 233,
   BOUND_EXPANDED = 234,
   TARGETS = 240,
   BOUND_TARGETS = 241,
};

enum class ExpressionClass : uint8_t {
   INVALID = 0,
   //===--------------------------------------------------------------------===//
   // Parsed Expressions
   //===--------------------------------------------------------------------===//
   AGGREGATE = 1,
   CASE = 2,
   CAST = 3,
   COLUMN_REF = 4,
   COMPARISON = 5,
   CONJUNCTION = 6,
   CONSTANT = 7,
   DEFAULT = 8,
   FUNCTION = 9,
   OPERATOR = 10,
   STAR = 11,
   SUBQUERY = 13,
   WINDOW = 14,
   PARAMETER = 15,
   COLLATE = 16,
   LAMBDA = 17,
   POSITIONAL_REFERENCE = 18,
   BETWEEN = 19,
   LAMBDA_REF = 20,
   //===--------------------------------------------------------------------===//
   // Bound Expressions
   //===--------------------------------------------------------------------===//
   BOUND_AGGREGATE = 25,
   BOUND_CASE = 26,
   BOUND_CAST = 27,
   BOUND_COLUMN_REF = 28,
   BOUND_COMPARISON = 29,
   BOUND_CONJUNCTION = 30,
   BOUND_CONSTANT = 31,
   BOUND_DEFAULT = 32,
   BOUND_FUNCTION = 33,
   BOUND_OPERATOR = 34,
   BOUND_PARAMETER = 35,
   BOUND_REF = 36,
   BOUND_SUBQUERY = 37,
   BOUND_WINDOW = 38,
   BOUND_BETWEEN = 39,
   BOUND_UNNEST = 40,
   BOUND_LAMBDA = 41,
   BOUND_LAMBDA_REF = 42,
   BOUND_STAR = 43,
   //===--------------------------------------------------------------------===//
   // Miscellaneous
   //===--------------------------------------------------------------------===//
   BOUND_EXPRESSION = 50,
   BOUND_EXPANDED = 51,
   TARGETS = 52,
   BOUND_TARGETS = 53,

};

class ParsedExpression : public BaseExpression {
   public:
   ParsedExpression(ExpressionType type, ExpressionClass expression_class) : BaseExpression(type, expression_class) {
   }

   virtual std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) = 0;

   virtual size_t hash();
   virtual bool operator==(ParsedExpression& other);
};

struct ParsedExprPtrHash {
   size_t operator()(const std::shared_ptr<ParsedExpression>& expr) const {
      return expr ? expr->hash() : 0;
   }
};
struct ParsedExprPtrEqual {
   bool operator()(const std::shared_ptr<ParsedExpression>& lhs,
                  const std::shared_ptr<ParsedExpression>& rhs) const {
      if (lhs == rhs) return true;  // Same pointer or both null
      if (!lhs || !rhs) return false;  // One is null, other isn't
      return *lhs == *rhs;  // Compare actual expressions
   }
};



class ColumnRefExpression : public ParsedExpression {
   public:
   static constexpr ExpressionClass TYPE = ExpressionClass::COLUMN_REF;

   //! Specify both the column and table name
   ColumnRefExpression(std::string column_name, std::string table_name);
   //! Specify both the column and table alias
   //TODO ColumnRefExpression(std::string column_name, const BindingAlias &alias);
   //! Only specify the column name, the table name will be derived later
   explicit ColumnRefExpression(std::string column_name);
   //! Specify a set of names
   explicit ColumnRefExpression(std::vector<std::string> column_names);

   //! The stack of names in order of which they appear (column_names[0].column_names[1].column_names[2]....)
   std::vector<std::string> column_names;

   size_t hash() override;
   bool operator==(ParsedExpression& other) override;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};

//! ComparisonExpression represents a boolean comparison (e.g. =, >=, <>). Always returns a boolean
//! and has two children.
class ComparisonExpression : public ParsedExpression {
   public:
   static constexpr const ExpressionClass TYPE = ExpressionClass::COMPARISON;

   explicit ComparisonExpression(ExpressionType type);
   ComparisonExpression(ExpressionType type, std::shared_ptr<ParsedExpression> left, std::shared_ptr<ParsedExpression> right);
   ComparisonExpression(ExpressionType type, std::shared_ptr<ParsedExpression> left, std::vector<std::shared_ptr<ParsedExpression>> rightChildren);

   std::shared_ptr<ParsedExpression> left;
   ///Multiple right children, e.g. for IN expressions IN (1, 2, 3)
   std::vector<std::shared_ptr<ParsedExpression>> rightChildren;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;

   private:
   std::string typeToAscii(ExpressionType type) const;
};

class ConjunctionExpression : public ParsedExpression {
   public:
   static constexpr const ExpressionClass TYPE = ExpressionClass::CONJUNCTION;
   explicit ConjunctionExpression(ExpressionType type);
   ConjunctionExpression(ExpressionType type, std::shared_ptr<lingodb::ast::ParsedExpression> left, std::shared_ptr<lingodb::ast::ParsedExpression> right);
   ConjunctionExpression(ExpressionType type, std::vector<std::shared_ptr<ParsedExpression>> children);

   std::vector<std::shared_ptr<ParsedExpression>> children;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;

   private:
   std::string typeToAscii(ExpressionType type) const;
};

class ConstantExpression : public ParsedExpression {
   public:
   static constexpr ExpressionClass TYPE = ExpressionClass::CONSTANT;
   ConstantExpression();

   std::shared_ptr<Value> value;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;

   size_t hash() override;
   bool operator==(ParsedExpression& other) override;
};

class FunctionExpression : public ParsedExpression {
   public:
   static constexpr ExpressionClass TYPE = ExpressionClass::FUNCTION;
   //TODO Finish constructor
   FunctionExpression(std::string catalog, std::string schema, std::string functionName, bool isOperator, bool distinct, bool exportState);

   //! Catalog of the function
   std::string catalog;
   //! Schema of the function
   std::string schema;
   //! Function name
   std::string functionName;
   //! Whether or not the function is an operator, only used for rendering
   bool isOperator;
   //! List of arguments to the function
   std::vector<std::shared_ptr<ParsedExpression>> arguments;
   //! Whether or not the aggregate function is distinct, only used for aggregates
   bool distinct;
   //! Expression representing a filter, only used for aggregates
   std::shared_ptr<ParsedExpression> filter;
   //! Modifier representing an ORDER BY, only used for aggregates
   std::shared_ptr<ParsedExpression> orderBy;
   //! whether this function should export its state or not
   bool exportState;

   bool star = false;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;

   size_t hash() override;
   bool operator==(ParsedExpression& other) override;
};
static std::vector<std::string> aggregationFunctions{
   "min",
   "max",
   "avg",
   "sum", "count", "stddev_samp"};

class StarExpression : public ParsedExpression {
   public:
   static constexpr const ExpressionClass TYPE = ExpressionClass::STAR;

   explicit StarExpression(std::string relationName);
   //! The relation name in case of tbl.*, or empty if this is a normal *
   std::string relationName;

   //TODO add missing variables

   //! The expression to select the columns (regular expression or list)
   std::shared_ptr<ParsedExpression> expr;

   //! Whether or not this is a COLUMNS expression
   bool columnsExpr = false;

   /*
    * Semantic
    */
   //Columns and their scope
   std::vector<std::pair<std::string, catalog::Column>> columns{};

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};

//List of targets
//Used for the select_list
//Select ...,...,...
class TargetsExpression : public ParsedExpression {
   public:
   static constexpr ExpressionClass TYPE = ExpressionClass::TARGETS;
   TargetsExpression();

   std::vector<std::shared_ptr<ParsedExpression>> targets{};

   std::optional<std::vector<std::shared_ptr<ParsedExpression>>> distinctExpressions = std::nullopt;

   /*
    * Semantic
    */
   //TODO make std::vector<mlir::Attribute> names; instead of std::string vector
   std::vector<std::string> names{};
   //TODO only reference to column

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};

class OperatorExpression : public ParsedExpression {
   public:
   static constexpr const ExpressionClass TYPE = ExpressionClass::OPERATOR;
   OperatorExpression(ExpressionType type, std::shared_ptr<ParsedExpression> left);
   OperatorExpression(ExpressionType type, std::shared_ptr<ParsedExpression> left, std::shared_ptr<ParsedExpression> right);
   std::vector<std::shared_ptr<ParsedExpression>> children;
   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};

class CastExpression : public ParsedExpression {
   public:
   static constexpr const ExpressionClass TYPE = ExpressionClass::CAST;
   CastExpression(LogicalTypeWithMods logicalTypeWithMods, std::shared_ptr<ParsedExpression> child);
   std::optional<LogicalTypeWithMods> logicalTypeWithMods;
   //TODO better
   std::optional<LogicalType> optInterval;
   std::shared_ptr<ParsedExpression> child;
   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};

enum class WindowBoundary : uint8_t {
   INVALID = 0,
   UNBOUNDED_PRECEDING = 1,
   UNBOUNDED_FOLLOWING = 2,
   CURRENT_ROW_RANGE = 3,
   CURRENT_ROW_ROWS = 4,
   EXPR_PRECEDING_ROWS = 5,
   EXPR_FOLLOWING_ROWS = 6,
   EXPR_PRECEDING_RANGE = 7,
   EXPR_FOLLOWING_RANGE = 8,
   CURRENT_ROW_GROUPS = 9,
   EXPR_PRECEDING_GROUPS = 10,
   EXPR_FOLLOWING_GROUPS = 11
};
class WindowExpression : public ParsedExpression {
   public:
   static constexpr const ExpressionClass TYPE = ExpressionClass::WINDOW;
   WindowExpression(ExpressionType type, std::string catalogName, std::string schemaName, std::string functionName);

   //TODO

   std::string functionName;

   std::vector<std::shared_ptr<ParsedExpression>> children;

   std::vector<std::shared_ptr<ParsedExpression>> partitions;

   std::vector<std::shared_ptr<OrderByModifier>> orders;

   /// The window boundaries
   WindowBoundary start = WindowBoundary::INVALID;
   WindowBoundary end = WindowBoundary::INVALID;
};

class BetweenExpression : public ParsedExpression {
   public:
   static constexpr const ExpressionClass TYPE = ExpressionClass::BETWEEN;

   BetweenExpression(ExpressionType type, std::shared_ptr<ParsedExpression> input, std::shared_ptr<ParsedExpression> lower, std::shared_ptr<ParsedExpression> upper);

   std::shared_ptr<ParsedExpression> input;
   std::shared_ptr<ParsedExpression> lower;
   std::shared_ptr<ParsedExpression> upper;
   bool asymmetric = false; // If true, the lower and upper bounds are not symmetric (e.g., BETWEEN x AND y vs. BETWEEN y AND x)

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};

   enum class SubqueryType : uint8_t {
   INVALID = 0,
   SCALAR = 1,     // Regular scalar subquery
   EXISTS = 2,     // EXISTS (SELECT...)
   NOT_EXISTS = 3, // NOT EXISTS(SELECT...)
   ANY = 4,        // x = ANY(SELECT...) OR x IN (SELECT...)
   NOT_ANY = 5,   // x != ANY(SELECT...) OR x NOT IN (SELECT...)
};

class SubqueryExpression : public ParsedExpression {
   public:
   static constexpr const ExpressionClass TYPE = ExpressionClass::SUBQUERY;

   SubqueryExpression(SubqueryType subQueryType, std::shared_ptr<TableProducer> subquery);

   //! The type of the subquery, e.g. scalar, exists, not exists, any
   SubqueryType subQueryType;
   //! The subquery expression
   std::shared_ptr<TableProducer> subquery;

   std::shared_ptr<ParsedExpression> testExpr; // For EXISTS/NOT EXISTS/ANY/NOT ANY, the expression to test against

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};

class CaseExpression : public ParsedExpression {
   public:
   struct CaseCheck {
      std::shared_ptr<ParsedExpression> whenExpr;
      std::shared_ptr<ParsedExpression> thenExpr;
   };

   static constexpr const ExpressionClass TYPE = ExpressionClass::CASE;

   CaseExpression(std::vector<CaseCheck> caseChecks, std::shared_ptr<ParsedExpression> elseExpr);

   std::vector<CaseCheck> caseChecks;
   std::shared_ptr<ParsedExpression> elseExpr;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;



};

} // namespace lingodb::ast
