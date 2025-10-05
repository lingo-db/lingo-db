#ifndef LINGODB_COMPILER_FRONTEND_SQL_ANALYZER_H
#define LINGODB_COMPILER_FRONTEND_SQL_ANALYZER_H

#include "ast/bound/bound_insert_node.h"
#include "lingodb/compiler/frontend/ast/bound/bound_expression.h"
#include "lingodb/compiler/frontend/ast/bound/bound_resultmodifier.h"
#include "lingodb/compiler/frontend/ast/query_node.h"
#include "lingodb/compiler/frontend/driver.h"
#include "lingodb/compiler/frontend/frontend_error.h"
#include "sql_context.h"
#define DEBUG false

#include <boost/context/detail/disable_overload.hpp>
#include <boost/context/stack_context.hpp>
#include <functional>
#include <sys/resource.h>
namespace lingodb::analyzer {
using ResolverScope = llvm::ScopedHashTable<std::string, std::shared_ptr<ast::ColumnReference>, StringInfo>::ScopeTy;
#define error(message, loc)              \
   {                                     \
      std::ostringstream s{};            \
      s << message;                      \
      throw FrontendError(s.str(), loc); \
   }
class StackGuard {
   public:
   StackGuard() = default;
   virtual ~StackGuard() = default;
   /**
    * Resets the stack guard
    * Exact behavior depends on implementation (normal stack or fiber)
    */
   virtual void reset() = 0;
   /**
    * @return true if stack limit is exceeded
    */
   virtual bool newStackNeeded() = 0;
};
/**
 * StackGuard for normal stack usage (no fiber)
 */
class StackGuardNormal : public StackGuard {
   public:
   StackGuardNormal();
   ~StackGuardNormal() override = default;
   void reset() override;
   bool newStackNeeded() override;

   private:
   void* startFameAddress;
   size_t limit;
};

/**
 * StackGuard for fiber stack usage (see https://www.boost.org/doc/libs/1_89_0/libs/context/doc/html/index.html)
 */
class StackGuardFiber : public StackGuard {
   public:
   StackGuardFiber(boost::context::stack_context& stackContext);
   ~StackGuardFiber() override = default;
   void reset() override;
   bool newStackNeeded() override;

   private:
   size_t limit;
   void* startFameAddress;
   boost::context::stack_context& stackContext;
};

class SQLCanonicalizer {
   public:
   /**
    * Transforms a query tree into a canonical.
    * Converts a nested SELECT statement into a sequence of pipe operations.
    *
    * Key transformations:
    * - SELECT node is converted into a pipeline of operations:
    *   FROM -> WHERE -> EXTEND -> AGGREGATE -> EXTEND -> SELECT -> WHERE -> MODIFIERS
    *   Functions inside SELECT are moved to the corresponding EXTEND/AGGREGATE PIPE
    * - Handles SET operations (UNION, etc.) by canonicalizing both branches
    * - Processes CTEs by canonicalizing both the CTE query and its child
    * - Maintains proper scoping for subqueries and nested operations
    *
    * @param rootNode Root of the query tree to canonicalize
    * @param context Transformation context for managing scopes and intermediate nodes
    * @return Transformed query tree in canonical form
    */
   std::shared_ptr<ast::TableProducer> canonicalize(std::shared_ptr<ast::TableProducer> rootNode, std::shared_ptr<ASTTransformContext> context);
   std::shared_ptr<ast::ParsedExpression> canonicalizeWindowExpression(std::shared_ptr<ast::WindowExpression> windowExpr, std::shared_ptr<ast::ExtendNode> extendNode, int& i, std::shared_ptr<ASTTransformContext> context);
   std::shared_ptr<ast::ParsedExpression> canonicalizeFunctionExpression(std::shared_ptr<ast::ParsedExpression> rootNode, std::shared_ptr<ast::FunctionExpression> functionExpr, bool extend, std::shared_ptr<ast::ExtendNode> extendNode, int& i, std::shared_ptr<ASTTransformContext> context);
   /**
    * Canonicalizes a parsed expression by transforming it into a canonical form.
    * This method recursively processes different types of expressions and applies specific
    * canonicalization rules based on the expression class.
    *
    * Key transformations include:
    * - Canonicalizes all child expressions recursively
    * - Functions: Special handling for aggregate and non-aggregate functions:
    *   - For aggregates: Creates a unique alias and adds to aggregation node, returns ColumnReference
    *   - For non-aggregate functions:
    *     - If extend=true: Adds to extension node if not already present, returns ColumnReference
    *     - If extend=false: Returns function unchanged (needed for SELECT list and GROUP BY)
    *
    * @param rootNode The expression tree to canonicalize
    * @param context The transformation context containing scoping and other metadata
    * @param extend Controls whether non-aggregate functions should be added to extension node
    * @return The canonicalized expression
    */
   std::shared_ptr<ast::ParsedExpression> canonicalizeParsedExpression(std::shared_ptr<ast::ParsedExpression> rootNode, std::shared_ptr<ASTTransformContext> context, bool extend, std::shared_ptr<ast::ExtendNode> extendNode);

   private:
   /**
    * Helper template method that combines canonicalization with static type casting.
    * Canonicalizes the node first, then casts it to the desired type.
    *
    * @tparam T Target type to cast the canonicalized node to
    * @param rootNode Node to be canonicalized
    * @param context Transformation context
    * @return Canonicalized node statically cast to type T
    */
   template <class T>
   std::shared_ptr<T> canonicalizeCast(std::shared_ptr<ast::TableProducer> rootNode, std::shared_ptr<ASTTransformContext> context);

   Driver drv{};
   std::shared_ptr<StackGuard> stackGuard = std::make_shared<StackGuardNormal>();
};

class SQLQueryAnalyzer {
   public:
   SQLQueryAnalyzer(catalog::Catalog* catalog);
   std::shared_ptr<SQLContext> context = std::make_shared<SQLContext>();
   std::shared_ptr<StackGuard> stackGuard = std::make_shared<StackGuardNormal>();

   std::shared_ptr<ast::AstNode> canonicalizeAndAnalyze(std::shared_ptr<ast::AstNode> rootNode, std::shared_ptr<SQLContext> context);

   private:
   std::shared_ptr<ast::TableProducer> analyzeTableProducer(std::shared_ptr<ast::TableProducer> rootNode, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope);
   std::shared_ptr<ast::CreateNode> analyzeCreateNode(std::shared_ptr<ast::CreateNode> createNode);
   std::shared_ptr<ast::BoundInsertNode> analyzeInsertNode(std::shared_ptr<ast::InsertNode> insertNode, std::shared_ptr<SQLContext> context, SQLContext::ResolverScope& resolverScope);
   std::shared_ptr<ast::SetNode> analyzeSetNode(std::shared_ptr<ast::SetNode> setNode);
   /**
    * Analyzes a single PIPE operator
    *
    * Behavior by operator type includes:
    * - binding columns and intermediate results
    * - handling types, check for correct types, finding common types
    * @param pipeOperator The parsed PIPE operator to analyze
    * @param context      Analysis context providing scope and attribute mapping utilities
    * @param resolverScope Resolver scope used for symbol resolution and replacement
    * @return A bound TableProducer representing the analyzed operator
    * @throws FrontendError on semantic or type errors (e.g., non-boolean WHERE)
    */
   std::shared_ptr<ast::TableProducer> analyzePipeOperator(std::shared_ptr<ast::PipeOperator> pipeOperator, std::shared_ptr<SQLContext>& context, ResolverScope& resolverScope);
   /**
   * Analyzes a TableRef
   * @param tableRef The table reference to analyze
   * @param context      Analysis context providing scope and attribute mapping utilities
   * @param resolverScope Resolver scope used for symbol resolution and replacement
   * @return A bound TableProducer representing the analyzed table reference
   * @throws FrontendError on semantic or type errors
   */
   std::shared_ptr<ast::TableProducer> analyzeTableRef(std::shared_ptr<ast::TableRef> tableRef, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope);
   std::shared_ptr<ast::TableProducer> analyzeExpressionListRef(std::shared_ptr<ast::ExpressionListRef> expressionListRef, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope);
   std::shared_ptr<ast::TableProducer> analyzeJoinRef(std::shared_ptr<ast::JoinRef> join, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope);
   std::shared_ptr<ast::TableProducer> analyzeBaseTableRef(std::shared_ptr<ast::BaseTableRef> baseTableRef, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope);
   std::shared_ptr<ast::TableProducer> analyzeInnerJoin(std::shared_ptr<ast::JoinRef> join, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope);
   std::shared_ptr<ast::TableProducer> analyzeLeftOuterJoin(std::shared_ptr<ast::JoinRef> join, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope);
   std::shared_ptr<ast::TableProducer> analyzeFullOuterJoin(std::shared_ptr<ast::JoinRef> join, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope);
   /**
  * Analyzes a result modifier like order by or limit
  * @param resultModifier The result modifier to analyze
  * @param context      Analysis context providing scope and attribute mapping utilities
  * @return A bound result modifier representing the analyzed result modifier
  * @throws FrontendError on semantic or type errors
  */
   std::shared_ptr<ast::BoundResultModifier> analyzeResultModifier(std::shared_ptr<ast::ResultModifier> resultModifier, std::shared_ptr<SQLContext> context);

   /**
    * Analyzes expressions
    *
    * Behavior by operator type includes:
    * - binding columns and intermediate results
    * - handling types, check for correct types, finding common types
    * @param rootNode The expression to analyze
    * @param context      Analysis context providing scope and attribute mapping utilities
    * @param resolverScope Resolver scope used for symbol resolution and replacement
    * @return A bound expression representing the analyzed operator
    * @throws FrontendError on semantic or type errors (e.g., non-boolean conjunction)
    */
   std::shared_ptr<ast::BoundExpression> analyzeExpression(std::shared_ptr<ast::ParsedExpression> rootNode, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope);
   std::shared_ptr<ast::BoundExpression> analyzeOperatorExpression(std::shared_ptr<ast::OperatorExpression> operatorExpr, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope);
   std::shared_ptr<ast::BoundExpression> analyzeWindowExpression(std::shared_ptr<ast::WindowExpression> windowExpr, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope);
   std::shared_ptr<ast::BoundExpression> analyzeCastExpression(std::shared_ptr<ast::CastExpression> castExpr, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope);
   std::shared_ptr<ast::BoundExpression> analyzeFunctionExpression(std::shared_ptr<ast::FunctionExpression> function, std::shared_ptr<SQLContext> context, ResolverScope& resolverScope);
   std::shared_ptr<ast::BoundColumnRefExpression> analyzeColumnRefExpression(std::shared_ptr<ast::ColumnRefExpression> columnRef, std::shared_ptr<SQLContext> context);

   ast::ExpressionType stringToExpressionType(const std::string& parserStr) {
      std::string str = parserStr;
      std::transform(str.begin(), str.end(), str.begin(), ::toupper);
      return llvm::StringSwitch<ast::ExpressionType>(str)
         .Case("||", ast::ExpressionType::OPERATOR_CONCAT)
         .Default(ast::ExpressionType::OPERATOR_UNKNOWN);
   }

   catalog::Catalog* catalog;
   Driver drv{};
   SQLCanonicalizer sqlCanonicalizer{};
};

struct SQLTypeUtils {
   /**
    * !Not yet fully implemented <br>
    * Determines the common type between two nullable types
    * Used for type resolution in operations involving two different types (e.g., comparisons, arithmetic operations).
    *
    * Handles the following type conversions:
    * - Identical types (returns the same type, with DECIMAL getting special handling)
    * - DATE with STRING/INTERVAL
    * - DECIMAL with INT
    * - STRING with CHAR
    * - INT with CHAR
    *
    * @param nullableType1 First type to compare
    * @param nullableType2 Second type to compare
    * @return A NullableType representing the common type that can hold both values
    * @throws std::runtime_error if no common type exists between the input types
    */
   static NullableType getCommonType(NullableType nullableType1, NullableType nullableType2);
   static NullableType getHigherDecimalType(NullableType left, NullableType right);

   static NullableType getCommonBaseType(std::vector<NullableType> types, ast::ExpressionType operationType);
   static NullableType getCommonBaseType(std::vector<NullableType> types);
   static NullableType getCommonTypeAfterOperation(NullableType type1, NullableType type2, ast::ExpressionType operationType);

   static std::vector<NullableType> toCommonTypes(std::vector<NullableType> types);
   static std::vector<NullableType> toCommonNumber(std::vector<NullableType> types);

   static NullableType typemodsToCatalogType(catalog::LogicalTypeId logicalTypeId, std::vector<std::shared_ptr<ast::Value>>& typeModifiers);

   [[nodiscard]]
   static std::pair<unsigned long, unsigned long> getAdaptedDecimalPAndSAfterMulDiv(unsigned long p, unsigned long s);
};
} // namespace lingodb::analyzer
#endif
