#pragma once
#include "lingodb/compiler/frontend/sql_scope.h"
#include "lingodb/compiler/frontend/ast/ast_node.h"
#include "lingodb/compiler/frontend/column_semantic.h"
#include "lingodb/compiler/frontend/ast/parsed_expression.h"

#include <mlir/Dialect/MLProgram/Transforms/Passes.h.inc>
#include <mlir/IR/Types.h>
namespace lingodb::ast {
enum class BindingType : uint8_t {
   TABLE = 1,
   FUNCTION = 2,
   //TODO other
};
class BoundExpression : public AstNode {
   public:
   //TODO make enums sense
   BoundExpression(ExpressionClass exprClass, ExpressionType type, std::string alias) : AstNode(NodeType::BOUND_EXPRESSION), exprClass(exprClass), type(type), alias(alias) {}
   BoundExpression(ExpressionClass exprClass, ExpressionType type, catalog::Type resultType, std::string alias) : AstNode(NodeType::BOUND_EXPRESSION), exprClass(exprClass), type(type), resultType(catalog::NullableType(resultType)), alias(alias) {}
   BoundExpression(ExpressionClass exprClass, ExpressionType type, catalog::NullableType resultType, std::string alias) : AstNode(NodeType::BOUND_EXPRESSION), exprClass(exprClass), type(type), resultType(resultType), alias(alias) {}

   ExpressionClass exprClass;
   ExpressionType type;
   //! The alias of the expression
   std::string alias;

   std::optional<catalog::NullableType> resultType = std::nullopt;




   //If this expression is a column reference or (SELECT 2*d from t), it can be used to find the named result
   std::optional<std::shared_ptr<NamedResult>> namedResult;
};

class BoundColumnRefExpression : public BoundExpression {
   public:
   static constexpr ExpressionClass TYPE = ExpressionClass::BOUND_COLUMN_REF;

   //! Specify both the column and table name
   BoundColumnRefExpression(std::string scope, catalog::NullableType resultType, std::shared_ptr<NamedResult> namedResult, std::string alias);

   //TODO semenatic
   std::string scope;
   std::shared_ptr<NamedResult> namedResult;

   //TODO type etc

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};

class BoundComparisonExpression : public BoundExpression {
   public:
   static constexpr const ExpressionClass TYPE = ExpressionClass::BOUND_COMPARISON;


   BoundComparisonExpression(ExpressionType type, std::string alias, std::shared_ptr<BoundExpression> left, std::vector<std::shared_ptr<BoundExpression>> rightChildren);

   std::shared_ptr<BoundExpression> left;
   std::vector<std::shared_ptr<BoundExpression>> rightChildren;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};

class BoundConjunctionExpression : public BoundExpression {
   public:
   static constexpr const ExpressionClass TYPE = ExpressionClass::BOUND_CONJUNCTION;
   BoundConjunctionExpression(ExpressionType type, std::string alias, std::shared_ptr<BoundExpression> left, std::shared_ptr<BoundExpression> right);
   BoundConjunctionExpression(ExpressionType type, std::string alias, std::vector<std::shared_ptr<BoundExpression>> children);

   std::vector<std::shared_ptr<BoundExpression>> children;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;

   private:
   std::string typeToAscii(ExpressionType type) const;
};

class BoundConstantExpression : public BoundExpression {
   public:
   static constexpr ExpressionClass TYPE = ExpressionClass::BOUND_CONSTANT;
   BoundConstantExpression(catalog::NullableType resultType, std::shared_ptr<Value> value, std::string alias);

   std::shared_ptr<Value> value;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};

//TODO other
class BoundTargetsExpression : public BoundExpression {
   public:
   static constexpr ExpressionClass TYPE = ExpressionClass::BOUND_TARGETS;
   //BoundTargetsExpression(std::vector<std::shared_ptr<BoundExpression>> targets, std::vector<std::pair<std::string, catalog::Column>> targetColumns);
   BoundTargetsExpression(std::string alias, std::vector<std::shared_ptr<BoundExpression>> targets, std::vector<std::shared_ptr<NamedResult>> targetColumns);

   std::vector<std::shared_ptr<BoundExpression>> targets;
   //std::vector<std::pair<std::string, catalog::Column>> targetColumns;
   std::vector<std::shared_ptr<NamedResult>> targetColumns;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};

class BoundFunctionExpression : public BoundExpression {
   public:
   static constexpr const ExpressionClass TYPE = ExpressionClass::BOUND_FUNCTION;
   BoundFunctionExpression(ExpressionType type, catalog::NullableType resultType, std::string functionName, std::string scope, std::string aliasOrUniqueIdentifier, bool distinct, std::vector<std::shared_ptr<BoundExpression>> arguments, std::shared_ptr<FunctionInfo> functionInfo);

   std::string functionName;
   std::string scope;
   bool distinct;
   //TODO!!!!!!!
   std::string aliasOrUniqueIdentifier;
   std::vector<std::shared_ptr<BoundExpression>> arguments;
   std::shared_ptr<FunctionInfo> functionInfo;


   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};

class BoundStarExpression : public BoundExpression {
   public:
   static constexpr const ExpressionClass TYPE = ExpressionClass::BOUND_STAR;
   explicit BoundStarExpression(std::string relationName, std::vector<std::pair<std::string, std::shared_ptr<NamedResult>>> namedResults);

   std::string relationName;
   std::vector<std::pair<std::string, std::shared_ptr<NamedResult>>> namedResults{};

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};

class BoundOperatorExpression : public BoundExpression {
   public:
   static constexpr const ExpressionClass TYPE = ExpressionClass::BOUND_OPERATOR;
   BoundOperatorExpression(ExpressionType type, catalog::NullableType resultType, std::string alias, std::vector<std::shared_ptr<BoundExpression>> children);
   BoundOperatorExpression(ExpressionType type, catalog::NullableType resultType, std::string alias, std::shared_ptr<BoundExpression> left, std::shared_ptr<BoundExpression> right);

   std::vector<std::shared_ptr<BoundExpression>> children;
   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};

class BoundCastExpression : public BoundExpression {
   public:
   static constexpr const ExpressionClass TYPE = ExpressionClass::BOUND_CAST;
   BoundCastExpression(catalog::Type resultType, std::string alias, std::shared_ptr<BoundExpression> child, std::optional<LogicalType> logicalType, std::string stringRepr);
   std::optional<LogicalType> logicalType;
   //TODO better
   std::string stringRepr;
   std::shared_ptr<BoundExpression> child;
   //LogicalType logicalType;
   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};

class BoundBetweenExpression : public BoundExpression {
   public:
   static constexpr const ExpressionClass TYPE = ExpressionClass::BOUND_BETWEEN;

   BoundBetweenExpression(ExpressionType type, catalog::Type resultType, std::string alias, std::shared_ptr<BoundExpression> input, std::shared_ptr<BoundExpression> lower, std::shared_ptr<BoundExpression> upper);


   std::shared_ptr<BoundExpression> input;
   std::shared_ptr<BoundExpression> lower;
   std::shared_ptr<BoundExpression> upper;
   bool asymmetric = false; // If true, the lower and upper bounds are not symmetric (e.g., BETWEEN x AND y vs. BETWEEN y AND x)

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};

class BoundSubqueryExpression : public BoundExpression {
   public:
   static constexpr const ExpressionClass TYPE = ExpressionClass::BOUND_SUBQUERY;

   BoundSubqueryExpression(SubqueryType subqueryType, catalog::NullableType resultType, std::string alias, std::shared_ptr<NamedResult> namedResult, std::shared_ptr<analyzer::SQLScope> sqlScope, std::shared_ptr<TableProducer> subquery,  std::shared_ptr<BoundExpression> testExpr);

   SubqueryType subqueryType = SubqueryType::INVALID;
   //! The subquery expression
   std::shared_ptr<TableProducer> subquery;
   std::shared_ptr<analyzer::SQLScope> sqlScope;
   //! Expression to test against. The left side expression of an IN expression.
   std::shared_ptr<BoundExpression> testExpr;


   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};

class BoundCaseExpression : public BoundExpression {
   public:
   struct BoundCaseCheck {
      std::shared_ptr<BoundExpression> whenExpr;
      std::shared_ptr<BoundExpression> thenExpr;
   };
   static constexpr const ExpressionClass TYPE = ExpressionClass::BOUND_CASE;

   BoundCaseExpression( catalog::NullableType resultType, std::string alias, std::vector<BoundCaseCheck> caseChecks, std::shared_ptr<BoundExpression> caseExpr);

   std::vector<BoundCaseCheck> caseChecks;
   std::shared_ptr<BoundExpression> elseExpr;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};
} // namespace lingodb::ast