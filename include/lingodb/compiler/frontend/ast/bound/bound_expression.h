#pragma once
#include "lingodb/compiler/frontend/ast/ast_node.h"
#include "lingodb/compiler/frontend/ast/parsed_expression.h"
#include "lingodb/compiler/frontend/column_semantic.h"
#include "lingodb/compiler/frontend/sql_scope.h"

#include <mlir/Dialect/MLProgram/Transforms/Passes.h.inc>
#include <mlir/IR/Types.h>
namespace lingodb::ast {
class BoundOrderByModifier;
enum class BindingType : uint8_t {
   TABLE = 1,
   FUNCTION = 2,
};
class BoundExpression : public AstNode {
   public:
   BoundExpression(ExpressionClass exprClass, ExpressionType type, std::string alias) : AstNode(NodeType::BOUND_EXPRESSION), exprClass(exprClass), type(type), alias(alias) {}
   BoundExpression(ExpressionClass exprClass, ExpressionType type, catalog::Type resultType, std::string alias) : AstNode(NodeType::BOUND_EXPRESSION), exprClass(exprClass), type(type), resultType(NullableType(resultType)), alias(alias) {}
   BoundExpression(ExpressionClass exprClass, ExpressionType type, NullableType resultType, std::string alias) : AstNode(NodeType::BOUND_EXPRESSION), exprClass(exprClass), type(type), resultType(resultType), alias(alias) {}

   ExpressionClass exprClass;
   ExpressionType type;
   //! The alias of the expression
   std::string alias;

   std::optional<NullableType> resultType = std::nullopt;

   //If this expression is a column reference or (SELECT 2*d from t), it can be used to find the named result
   std::optional<std::shared_ptr<NamedResult>> namedResult;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override {
      return "";
   };
};

class BoundColumnRefExpression : public BoundExpression {
   public:
   static constexpr ExpressionClass TYPE = ExpressionClass::BOUND_COLUMN_REF;

   //! Specify both the column and table name
   BoundColumnRefExpression(NullableType resultType, std::shared_ptr<NamedResult> namedResult, std::string alias);
};

class BoundComparisonExpression : public BoundExpression {
   public:
   static constexpr const ExpressionClass TYPE = ExpressionClass::BOUND_COMPARISON;

   BoundComparisonExpression(ExpressionType type, std::string alias, bool resultTypeNullable, std::shared_ptr<BoundExpression> left, std::vector<std::shared_ptr<BoundExpression>> rightChildren);

   std::shared_ptr<BoundExpression> left;
   std::vector<std::shared_ptr<BoundExpression>> rightChildren;
};

class BoundConjunctionExpression : public BoundExpression {
   public:
   static constexpr const ExpressionClass TYPE = ExpressionClass::BOUND_CONJUNCTION;
   BoundConjunctionExpression(ExpressionType type, std::string alias, std::shared_ptr<BoundExpression> left, std::shared_ptr<BoundExpression> right);
   BoundConjunctionExpression(ExpressionType type, std::string alias, std::vector<std::shared_ptr<BoundExpression>> children);

   std::vector<std::shared_ptr<BoundExpression>> children;
};

class BoundConstantExpression : public BoundExpression {
   public:
   static constexpr ExpressionClass TYPE = ExpressionClass::BOUND_CONSTANT;
   BoundConstantExpression(NullableType resultType, std::shared_ptr<Value> value, std::string alias);

   std::shared_ptr<Value> value;
};

class BoundTargetsExpression : public BoundExpression {
   public:
   static constexpr ExpressionClass TYPE = ExpressionClass::BOUND_TARGETS;
   BoundTargetsExpression(std::string alias, bool distinct, std::vector<std::shared_ptr<NamedResult>> targetColumns);

   bool distinct = false;
   std::vector<std::shared_ptr<NamedResult>> targetColumns;
};

class BoundFunctionExpression : public BoundExpression {
   public:
   static constexpr const ExpressionClass TYPE = ExpressionClass::BOUND_FUNCTION;
   BoundFunctionExpression(ExpressionType type, NullableType resultType, std::string functionName, std::string scope, std::string alias, bool distinct, std::vector<std::shared_ptr<BoundExpression>> arguments);

   std::string functionName;
   std::string scope;
   bool distinct;


   std::vector<std::shared_ptr<BoundExpression>> arguments;
};

class BoundStarExpression : public BoundExpression {
   public:
   static constexpr const ExpressionClass TYPE = ExpressionClass::BOUND_STAR;
   explicit BoundStarExpression(std::string relationName, std::vector<std::shared_ptr<NamedResult>> namedResults);

   std::string relationName;
   std::vector<std::shared_ptr<NamedResult>> namedResults{};
};

class BoundOperatorExpression : public BoundExpression {
   public:
   static constexpr const ExpressionClass TYPE = ExpressionClass::BOUND_OPERATOR;
   BoundOperatorExpression(ExpressionType type, NullableType resultType, std::string alias, std::vector<std::shared_ptr<BoundExpression>> children);
   BoundOperatorExpression(ExpressionType type, NullableType resultType, std::string alias, std::shared_ptr<BoundExpression> left, std::shared_ptr<BoundExpression> right);

   std::vector<std::shared_ptr<BoundExpression>> children;
};

class BoundCastExpression : public BoundExpression {
   public:
   static constexpr const ExpressionClass TYPE = ExpressionClass::BOUND_CAST;
   BoundCastExpression(NullableType resultType, std::string alias, std::shared_ptr<BoundExpression> child, std::optional<LogicalTypeWithMods> logicalTypeWithMods, std::string stringRepr);
   std::optional<LogicalTypeWithMods> logicalTypeWithMods;

   std::string stringRepr;
   std::shared_ptr<BoundExpression> child;
};
struct BoundWindowBoundary {
   WindowMode windowMode = WindowMode::INVALID;

   size_t start = std::numeric_limits<int64_t>::min();
   size_t end = std::numeric_limits<int64_t>::max();

   location loc;
};
class BoundWindowExpression : public BoundExpression {
   public:
   static constexpr const ExpressionClass TYPE = ExpressionClass::BOUND_WINDOW;
   BoundWindowExpression(ExpressionType type, std::string alias, NullableType resultType, std::shared_ptr<BoundFunctionExpression> function, std::vector<std::shared_ptr<BoundExpression>> partitions, std::optional<std::shared_ptr<BoundOrderByModifier>> order, std::shared_ptr<BoundWindowBoundary> windowBoundary );

   std::shared_ptr<BoundFunctionExpression> function;
   std::vector<std::shared_ptr<BoundExpression>> partitions;
   std::optional<std::shared_ptr<BoundOrderByModifier>> order;
   std::shared_ptr<BoundWindowBoundary> windowBoundary;
};

class BoundBetweenExpression : public BoundExpression {
   public:
   static constexpr const ExpressionClass TYPE = ExpressionClass::BOUND_BETWEEN;

   BoundBetweenExpression(ExpressionType type, catalog::Type resultType, std::string alias, std::shared_ptr<BoundExpression> input, std::shared_ptr<BoundExpression> lower, std::shared_ptr<BoundExpression> upper);

   std::shared_ptr<BoundExpression> input;
   std::shared_ptr<BoundExpression> lower;
   std::shared_ptr<BoundExpression> upper;
   bool asymmetric = false; // If true, the lower and upper bounds are not symmetric (e.g., BETWEEN x AND y vs. BETWEEN y AND x)
};

class BoundSubqueryExpression : public BoundExpression {
   public:
   static constexpr const ExpressionClass TYPE = ExpressionClass::BOUND_SUBQUERY;

   BoundSubqueryExpression(SubqueryType subqueryType, NullableType resultType, std::string alias, std::shared_ptr<NamedResult> namedResultForSubquery, std::shared_ptr<analyzer::SQLScope> sqlScope, std::shared_ptr<TableProducer> subquery, std::shared_ptr<BoundExpression> testExpr);

   SubqueryType subqueryType = SubqueryType::INVALID;
   /// The subquery expression
   std::shared_ptr<TableProducer> subquery;
   std::shared_ptr<NamedResult> namedResultForSubquery;
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
   static constexpr const ExpressionClass TYPE = ExpressionClass::BOUND_CASE;

   BoundCaseExpression(NullableType resultType, std::string alias, std::optional<std::shared_ptr<BoundExpression>> caseExpr, std::vector<BoundCaseCheck> caseChecks, std::shared_ptr<BoundExpression> elseExpr);

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
   static constexpr const ExpressionClass TYPE = ExpressionClass::BOUND_SET;
   BoundSetColumnExpression( std::string mapName, std::vector<std::shared_ptr<BoundExpression>> sets);

   std::string mapName;
   std::vector<std::shared_ptr<BoundExpression>> sets;
};
} // namespace lingodb::ast