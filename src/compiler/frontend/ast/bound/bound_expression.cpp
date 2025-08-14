#include "lingodb/compiler/frontend/ast/bound/bound_expression.h"
namespace lingodb::ast {
/*
 * BoundColumnRefExpression
*/
BoundColumnRefExpression::BoundColumnRefExpression(catalog::NullableType resultType, std::shared_ptr<NamedResult> namedResult, std::string alias) : BoundExpression(TYPE, ExpressionType::BOUND_COLUMN_REF, resultType, alias) {
   this->namedResult = namedResult;
}
std::string BoundColumnRefExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}

/*
 * BoundComparisonExpression
*/

BoundComparisonExpression::BoundComparisonExpression(ExpressionType type, std::string alias, bool resultTypeNullable, std::shared_ptr<BoundExpression> left, std::vector<std::shared_ptr<BoundExpression>> rightChildren) : BoundExpression(TYPE, type, catalog::NullableType(catalog::Type::boolean(), resultTypeNullable), alias), left(std::move(left)), rightChildren(std::move(rightChildren)) {
}
std::string BoundComparisonExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}

/*
 * BoundConjunctionExpression
*/
BoundConjunctionExpression::BoundConjunctionExpression(ExpressionType type, std::string alias, std::shared_ptr<BoundExpression> left, std::shared_ptr<BoundExpression> right) : BoundExpression(TYPE, type, catalog::Type::boolean(), alias), children(std::vector{left, right}) {
   if (type != ExpressionType::CONJUNCTION_AND && type != ExpressionType::CONJUNCTION_OR) {
      throw std::runtime_error("Invalid type for BoundConjunctionExpression");
   }
}
BoundConjunctionExpression::BoundConjunctionExpression(ExpressionType type, std::string alias, std::vector<std::shared_ptr<BoundExpression>> children) : BoundExpression(TYPE, type, catalog::Type::boolean(), alias), children(std::move(children)) {
}
std::string BoundConjunctionExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}

/*
 * BoundConstantExpression
*/
BoundConstantExpression::BoundConstantExpression(catalog::NullableType resultType, std::shared_ptr<Value> value, std::string alias) : BoundExpression(TYPE, ExpressionType::VALUE_CONSTANT, resultType, alias), value(std::move(value)) {
}
std::string BoundConstantExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}

/*
 * BoundTargetsExpression
*/
BoundTargetsExpression::BoundTargetsExpression(std::string alias, std::vector<std::shared_ptr<BoundExpression>> targets, std::optional<std::vector<std::shared_ptr<BoundExpression>>> distinctExpressions, std::vector<std::shared_ptr<NamedResult>> targetColumns) : BoundExpression(TYPE, ExpressionType::BOUND_TARGETS, alias), targets(std::move(targets)), distinctExpressions(std::move(distinctExpressions)), targetColumns(std::move(targetColumns)) {
   for (const auto& target : this->targets) {
      if (target->type == ExpressionType::AGGREGATE && target->exprClass == ExpressionClass::FUNCTION) {
         //TODO handle aggregation
      }
   }
}
std::string BoundTargetsExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}

/*
 * BoundFunctionExpression
*/
BoundFunctionExpression::BoundFunctionExpression(ExpressionType type, catalog::NullableType resultType, std::string functionName, std::string scope, std::string alias, bool distinct, std::vector<std::shared_ptr<BoundExpression>> arguments) : BoundExpression(TYPE, type, resultType, alias), functionName(functionName), scope(scope), distinct(distinct), arguments(arguments) {
}
std::string BoundFunctionExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}

/*
 * BoundStarExpression
*/
BoundStarExpression::BoundStarExpression(std::string relationName, std::vector<std::pair<std::string, std::shared_ptr<NamedResult>>> namedResults) : BoundExpression(TYPE, ExpressionType::STAR, ""), relationName(relationName), namedResults(std::move(namedResults)) {
}
std::string BoundStarExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}

/*
 * BoundComparisonExpression
*/
BoundOperatorExpression::BoundOperatorExpression(ExpressionType type, catalog::NullableType resultType, std::string alias, std::vector<std::shared_ptr<BoundExpression>> children) : BoundExpression(TYPE, type, resultType, alias), children(children) {
}
BoundOperatorExpression::BoundOperatorExpression(ExpressionType type, catalog::NullableType resultType, std::string alias, std::shared_ptr<BoundExpression> left, std::shared_ptr<BoundExpression> right) : BoundExpression(TYPE, type, resultType, alias), children({left, right}) {
}
std::string BoundOperatorExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}

/*
 * BoundCastExpression
*/
BoundCastExpression::BoundCastExpression(catalog::NullableType resultType, std::string alias, std::shared_ptr<BoundExpression> child, std::optional<LogicalTypeWithMods> logicalTypeWithMods, std::string stringRepr) : BoundExpression(TYPE, ExpressionType::CAST, resultType, alias), child(std::move(child)), logicalTypeWithMods(logicalTypeWithMods), stringRepr(stringRepr) {
}
std::string BoundCastExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}

/*
 * BoundWindoExpression
*/
BoundWindowExpression::BoundWindowExpression(ExpressionType type, std::string alias, catalog::NullableType resultType, std::shared_ptr<BoundFunctionExpression> function, std::vector<std::shared_ptr<BoundExpression>> partitions, std::optional<std::shared_ptr<BoundOrderByModifier>> order, std::shared_ptr<BoundWindowBoundary> windowBoundary) : BoundExpression(TYPE, type, resultType, alias), function(function), partitions(partitions), order(order), windowBoundary(windowBoundary){
}
std::string BoundWindowExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}

/*
 * BoundBetweenExpression
 */
BoundBetweenExpression::BoundBetweenExpression(ExpressionType type, catalog::Type resultType, std::string alias, std::shared_ptr<BoundExpression> input, std::shared_ptr<BoundExpression> lower, std::shared_ptr<BoundExpression> upper) : BoundExpression(TYPE, type, resultType, alias), input(std::move(input)), lower(std::move(lower)), upper(std::move(upper)) {
}
std::string BoundBetweenExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}

/*
 * BoundSubqueryExpression
 */
BoundSubqueryExpression::BoundSubqueryExpression(SubqueryType subqueryType, catalog::NullableType resultType, std::string alias, std::shared_ptr<NamedResult> namedResultForSubquery, std::shared_ptr<analyzer::SQLScope> sqlScope, std::shared_ptr<TableProducer> subquery, std::shared_ptr<BoundExpression> testExpr) : BoundExpression(TYPE, ExpressionType::SUBQUERY, resultType, alias), subqueryType(subqueryType), sqlScope(sqlScope), namedResultForSubquery(namedResultForSubquery), subquery(std::move(subquery)), testExpr(testExpr) {
}
std::string BoundSubqueryExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}

BoundCaseExpression::BoundCaseExpression(catalog::NullableType resultType, std::string alias, std::optional<std::shared_ptr<ast::BoundExpression>> caseExpr, std::vector<BoundCaseCheck> caseChecks, std::shared_ptr<BoundExpression> elseExpr) : BoundExpression(TYPE, ExpressionType::CASE_EXPR, resultType, alias), caseExpr(caseExpr),  caseChecks(std::move(caseChecks)), elseExpr(std::move(elseExpr)) {
}
std::string BoundCaseExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}
} // namespace lingodb::ast