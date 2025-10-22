#ifndef LINGODB_COMPILER_FRONTEND_AST_BOUND_BOUND_EXPRESSION_H
#define LINGODB_COMPILER_FRONTEND_AST_BOUND_BOUND_EXPRESSION_H

#include "lingodb/catalog/FunctionCatalogEntry.h"
#include "lingodb/compiler/frontend/ast/ast_node.h"
#include "lingodb/compiler/frontend/ast/parsed_expression.h"
#include "lingodb/compiler/frontend/column_semantic.h"
#include "lingodb/compiler/frontend/sql_scope.h"

#include <mlir/Dialect/MLProgram/Transforms/Passes.h.inc>
#include <unordered_set>
namespace lingodb::ast {
class BoundOrderByModifier;
enum class BindingType : uint8_t {
   TABLE = 1,
   FUNCTION = 2,
};
class BoundExpression : public AstNode {
   public:
   BoundExpression(ExpressionClass exprClass, ExpressionType type, std::string alias) : AstNode(NodeType::BOUND_EXPRESSION), exprClass(exprClass), type(type), alias(alias) {}
   BoundExpression(ExpressionClass exprClass, ExpressionType type, catalog::Type resultType, std::string alias) : AstNode(NodeType::BOUND_EXPRESSION), exprClass(exprClass), type(type), alias(alias), resultType(NullableType(resultType)) {}
   BoundExpression(ExpressionClass exprClass, ExpressionType type, NullableType resultType, std::string alias) : AstNode(NodeType::BOUND_EXPRESSION), exprClass(exprClass), type(type), alias(alias), resultType(resultType) {}

   ExpressionClass exprClass;
   ExpressionType type;
   //! The alias of the expression
   std::string alias;

   std::optional<NullableType> resultType = std::nullopt;

   //If this expression is a column reference or (SELECT 2*d from t), it can be used to find the named result
   std::optional<std::shared_ptr<ColumnReference>> columnReference;
};

class BoundColumnRefExpression : public BoundExpression {
   public:
   static constexpr ExpressionClass cType = ExpressionClass::BOUND_COLUMN_REF;

   //! Specify both the column and table name
   BoundColumnRefExpression(NullableType resultType, std::shared_ptr<ColumnReference> columnReference, std::string alias) : BoundExpression(cType, ExpressionType::BOUND_COLUMN_REF, resultType, alias) {
      this->columnReference = columnReference;
   }
};

class BoundComparisonExpression : public BoundExpression {
   public:
   static constexpr const ExpressionClass cType = ExpressionClass::BOUND_COMPARISON;

   BoundComparisonExpression(ExpressionType type, std::string alias, bool resultTypeNullable, std::shared_ptr<BoundExpression> left, std::vector<std::shared_ptr<BoundExpression>> rightChildren) : BoundExpression(cType, type, NullableType(catalog::Type::boolean(), resultTypeNullable), alias), left(std::move(left)), rightChildren(std::move(rightChildren)) {}

   std::shared_ptr<BoundExpression> left;
   std::vector<std::shared_ptr<BoundExpression>> rightChildren;
};

class BoundConjunctionExpression : public BoundExpression {
   public:
   static constexpr const ExpressionClass cType = ExpressionClass::BOUND_CONJUNCTION;
   BoundConjunctionExpression(ExpressionType type, std::string alias, std::vector<std::shared_ptr<BoundExpression>> children) : BoundExpression(cType, type, catalog::Type::boolean(), alias), children(std::move(children)) {}

   std::vector<std::shared_ptr<BoundExpression>> children;
};

class BoundConstantExpression : public BoundExpression {
   public:
   static constexpr ExpressionClass cType = ExpressionClass::BOUND_CONSTANT;
   BoundConstantExpression(NullableType resultType, std::shared_ptr<Value> value, std::string alias) : BoundExpression(cType, ExpressionType::VALUE_CONSTANT, resultType, alias), value(std::move(value)) {}

   std::shared_ptr<Value> value;
};

class BoundFunctionExpression : public BoundExpression {
   public:
   static constexpr const ExpressionClass cType = ExpressionClass::BOUND_FUNCTION;
   BoundFunctionExpression(ExpressionType type, NullableType resultType, std::string functionName, std::string scope, std::string alias, bool distinct, std::vector<std::shared_ptr<BoundExpression>> arguments) : BoundExpression(cType, type, resultType, alias), functionName(functionName), scope(scope), distinct(distinct), arguments(arguments) {}

   std::string functionName;
   std::string scope;
   bool distinct;
   //Is set if function is a UDF
   std::optional<std::shared_ptr<lingodb::catalog::FunctionCatalogEntry>> udfFunction;

   std::vector<std::shared_ptr<BoundExpression>> arguments;
};

class BoundStarExpression : public BoundExpression {
   public:
   static constexpr const ExpressionClass cType = ExpressionClass::BOUND_STAR;
   explicit BoundStarExpression(std::string relationName, std::unordered_set<std::pair<std::shared_ptr<ast::ColumnReference>, size_t>, ast::ColumnRefHash, ast::ColumnRefEq> columnReferences) : BoundExpression(cType, ExpressionType::STAR, ""), relationName(relationName), columnReferences(std::move(columnReferences)) {}

   std::string relationName;
   std::unordered_set<std::pair<std::shared_ptr<ast::ColumnReference>, size_t>, ast::ColumnRefHash, ast::ColumnRefEq> columnReferences{};
};

class BoundOperatorExpression : public BoundExpression {
   public:
   static constexpr const ExpressionClass cType = ExpressionClass::BOUND_OPERATOR;
   BoundOperatorExpression(ExpressionType type, NullableType resultType, std::string alias, std::vector<std::shared_ptr<BoundExpression>> children) : BoundExpression(cType, type, resultType, alias), children(children) {}

   std::vector<std::shared_ptr<BoundExpression>> children;
};

class BoundCastExpression : public BoundExpression {
   public:
   static constexpr const ExpressionClass cType = ExpressionClass::BOUND_CAST;
   BoundCastExpression(NullableType resultType, std::string alias, std::shared_ptr<BoundExpression> child, std::optional<LogicalTypeWithMods> logicalTypeWithMods, std::string stringRepr) : BoundExpression(cType, ExpressionType::CAST, resultType, alias), logicalTypeWithMods(logicalTypeWithMods), stringRepr(stringRepr), child(std::move(child)) {}
   std::optional<LogicalTypeWithMods> logicalTypeWithMods;

   std::string stringRepr;
   std::shared_ptr<BoundExpression> child;
};
struct BoundWindowFrame {
   WindowMode windowMode = WindowMode::INVALID;

   size_t start = std::numeric_limits<int64_t>::min();
   size_t end = std::numeric_limits<int64_t>::max();

   location loc;
};
class BoundWindowExpression : public BoundExpression {
   public:
   static constexpr const ExpressionClass cType = ExpressionClass::BOUND_WINDOW;
   BoundWindowExpression(ExpressionType type, std::string alias, NullableType resultType, std::shared_ptr<BoundFunctionExpression> function, std::vector<std::shared_ptr<BoundExpression>> partitions, std::optional<std::shared_ptr<BoundOrderByModifier>> order, std::shared_ptr<BoundWindowFrame> windowFrame) : BoundExpression(cType, type, resultType, alias), function(function), partitions(partitions), order(order), windowFrame(windowFrame) {}

   std::shared_ptr<BoundFunctionExpression> function;
   std::vector<std::shared_ptr<BoundExpression>> partitions;
   std::optional<std::shared_ptr<BoundOrderByModifier>> order;
   std::shared_ptr<BoundWindowFrame> windowFrame;
};

class BoundBetweenExpression : public BoundExpression {
   public:
   static constexpr const ExpressionClass cType = ExpressionClass::BOUND_BETWEEN;

   BoundBetweenExpression(ExpressionType type, catalog::Type resultType, std::string alias, std::shared_ptr<BoundExpression> input, std::shared_ptr<BoundExpression> lower, std::shared_ptr<BoundExpression> upper) : BoundExpression(cType, type, resultType, alias), input(std::move(input)), lower(std::move(lower)), upper(std::move(upper)) {}

   std::shared_ptr<BoundExpression> input;
   std::shared_ptr<BoundExpression> lower;
   std::shared_ptr<BoundExpression> upper;
   bool asymmetric = false; // If true, the lower and upper bounds are not symmetric (e.g., BETWEEN x AND y vs. BETWEEN y AND x)
};

class BoundSubqueryExpression : public BoundExpression {
   public:
   static constexpr const ExpressionClass cType = ExpressionClass::BOUND_SUBQUERY;

   BoundSubqueryExpression(SubqueryType subqueryType, NullableType resultType, std::string alias, std::shared_ptr<ColumnReference> columnReferenceForSubquery, std::shared_ptr<analyzer::SQLScope> sqlScope, std::shared_ptr<TableProducer> subquery, std::shared_ptr<BoundExpression> testExpr) : BoundExpression(cType, ExpressionType::SUBQUERY, resultType, alias), subqueryType(subqueryType), subquery(std::move(subquery)), columnReferenceForSubquery(columnReferenceForSubquery), sqlScope(sqlScope), testExpr(testExpr) {}

   SubqueryType subqueryType = SubqueryType::INVALID;
   /// The subquery expression
   std::shared_ptr<TableProducer> subquery;
   std::shared_ptr<ColumnReference> columnReferenceForSubquery;
   std::shared_ptr<analyzer::SQLScope> sqlScope;
   /// Expression to test against. The left side expression of an IN expression.
   std::shared_ptr<BoundExpression> testExpr;
   /// e.g. (x LIKE some(...))
   std::optional<ExpressionType> comparisonType = std::nullopt;
};

class BoundCaseExpression : public BoundExpression {
   public:
   struct BoundCaseCheck {
      std::shared_ptr<BoundExpression> whenExpr;
      std::shared_ptr<BoundExpression> thenExpr;
   };
   static constexpr const ExpressionClass cType = ExpressionClass::BOUND_CASE;

   BoundCaseExpression(NullableType resultType, std::string alias, std::optional<std::shared_ptr<BoundExpression>> caseExpr, std::vector<BoundCaseCheck> caseChecks, std::shared_ptr<BoundExpression> elseExpr) : BoundExpression(cType, ExpressionType::CASE_EXPR, resultType, alias), caseExpr(caseExpr), caseChecks(std::move(caseChecks)), elseExpr(std::move(elseExpr)) {}

   std::optional<std::shared_ptr<BoundExpression>> caseExpr; //CASE expr ...
   std::vector<BoundCaseCheck> caseChecks; //CASE ... WHEN caseCheck
   std::shared_ptr<BoundExpression> elseExpr; // CASE ... WHEN ...  ELSE elseExpr
};
/**
 * Used for SET pipe operator:
 * SET <column> = <expression>, ... : Same rows, with updated values for modified columns.
 */
class BoundSetColumnExpression : public BoundExpression {
   public:
   static constexpr const ExpressionClass cType = ExpressionClass::BOUND_SET;
   BoundSetColumnExpression(std::string mapName, std::vector<std::shared_ptr<BoundExpression>> sets) : BoundExpression(cType, ExpressionType::SET, ""), mapName(mapName), sets(std::move(sets)) {}

   std::string mapName;
   std::vector<std::shared_ptr<BoundExpression>> sets;
};
} // namespace lingodb::ast
#endif
