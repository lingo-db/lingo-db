#include "lingodb/compiler/frontend/ast/bound/bound_expression.h"
namespace lingodb::ast {
/*
 * BoundColumnRefExpression
*/
BoundColumnRefExpression::BoundColumnRefExpression(NullableType resultType, std::shared_ptr<NamedResult> namedResult, std::string alias) : BoundExpression(TYPE, ExpressionType::BOUND_COLUMN_REF, resultType, alias) {
   this->namedResult = namedResult;
}

/*
 * BoundComparisonExpression
*/

BoundComparisonExpression::BoundComparisonExpression(ExpressionType type, std::string alias, bool resultTypeNullable, std::shared_ptr<BoundExpression> left, std::vector<std::shared_ptr<BoundExpression>> rightChildren) : BoundExpression(TYPE, type, NullableType(catalog::Type::boolean(), resultTypeNullable), alias), left(std::move(left)), rightChildren(std::move(rightChildren)) {
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

/*
 * BoundConstantExpression
*/
BoundConstantExpression::BoundConstantExpression(NullableType resultType, std::shared_ptr<Value> value, std::string alias) : BoundExpression(TYPE, ExpressionType::VALUE_CONSTANT, resultType, alias), value(std::move(value)) {
}

/*
 * BoundTargetsExpression
*/
BoundTargetsExpression::BoundTargetsExpression(std::string alias, bool distinct, std::vector<std::shared_ptr<NamedResult>> targetColumns) : BoundExpression(TYPE, ExpressionType::BOUND_TARGETS, alias), distinct(std::move(distinct)), targetColumns(std::move(targetColumns)) {
}

/*
 * BoundFunctionExpression
*/
BoundFunctionExpression::BoundFunctionExpression(ExpressionType type, NullableType resultType, std::string functionName, std::string scope, std::string alias, bool distinct, std::vector<std::shared_ptr<BoundExpression>> arguments) : BoundExpression(TYPE, type, resultType, alias), functionName(functionName), scope(scope), distinct(distinct), arguments(arguments) {
}

/*
 * BoundStarExpression
*/
BoundStarExpression::BoundStarExpression(std::string relationName, std::vector<std::shared_ptr<NamedResult>> namedResults) : BoundExpression(TYPE, ExpressionType::STAR, ""), relationName(relationName), namedResults(std::move(namedResults)) {
}

/*
 * BoundComparisonExpression
*/
BoundOperatorExpression::BoundOperatorExpression(ExpressionType type, NullableType resultType, std::string alias, std::vector<std::shared_ptr<BoundExpression>> children) : BoundExpression(TYPE, type, resultType, alias), children(children) {
}
BoundOperatorExpression::BoundOperatorExpression(ExpressionType type, NullableType resultType, std::string alias, std::shared_ptr<BoundExpression> left, std::shared_ptr<BoundExpression> right) : BoundExpression(TYPE, type, resultType, alias), children({left, right}) {
}

/*
 * BoundCastExpression
*/
BoundCastExpression::BoundCastExpression(NullableType resultType, std::string alias, std::shared_ptr<BoundExpression> child, std::optional<LogicalTypeWithMods> logicalTypeWithMods, std::string stringRepr) : BoundExpression(TYPE, ExpressionType::CAST, resultType, alias), child(std::move(child)), logicalTypeWithMods(logicalTypeWithMods), stringRepr(stringRepr) {
}

/*
 * BoundWindoExpression
*/
BoundWindowExpression::BoundWindowExpression(ExpressionType type, std::string alias, NullableType resultType, std::shared_ptr<BoundFunctionExpression> function, std::vector<std::shared_ptr<BoundExpression>> partitions, std::optional<std::shared_ptr<BoundOrderByModifier>> order, std::shared_ptr<BoundWindowBoundary> windowBoundary) : BoundExpression(TYPE, type, resultType, alias), function(function), partitions(partitions), order(order), windowBoundary(windowBoundary){
}

/*
 * BoundBetweenExpression
 */
BoundBetweenExpression::BoundBetweenExpression(ExpressionType type, catalog::Type resultType, std::string alias, std::shared_ptr<BoundExpression> input, std::shared_ptr<BoundExpression> lower, std::shared_ptr<BoundExpression> upper) : BoundExpression(TYPE, type, resultType, alias), input(std::move(input)), lower(std::move(lower)), upper(std::move(upper)) {
}

/*
 * BoundSubqueryExpression
 */
BoundSubqueryExpression::BoundSubqueryExpression(SubqueryType subqueryType, NullableType resultType, std::string alias, std::shared_ptr<NamedResult> namedResultForSubquery, std::shared_ptr<analyzer::SQLScope> sqlScope, std::shared_ptr<TableProducer> subquery, std::shared_ptr<BoundExpression> testExpr) : BoundExpression(TYPE, ExpressionType::SUBQUERY, resultType, alias), subqueryType(subqueryType), sqlScope(sqlScope), namedResultForSubquery(namedResultForSubquery), subquery(std::move(subquery)), testExpr(testExpr) {
}
/*
 * BoundCaseExpression
 */
BoundCaseExpression::BoundCaseExpression(NullableType resultType, std::string alias, std::optional<std::shared_ptr<ast::BoundExpression>> caseExpr, std::vector<BoundCaseCheck> caseChecks, std::shared_ptr<BoundExpression> elseExpr) : BoundExpression(TYPE, ExpressionType::CASE_EXPR, resultType, alias), caseExpr(caseExpr),  caseChecks(std::move(caseChecks)), elseExpr(std::move(elseExpr)) {
}

/*
 * BoundSetExpression
 */
BoundSetColumnExpression::BoundSetColumnExpression(std::string mapName, std::vector<std::shared_ptr<BoundExpression>> sets) : BoundExpression(TYPE, ExpressionType::SET, ""), mapName(mapName), sets(std::move(sets)){
}

} // namespace lingodb::ast