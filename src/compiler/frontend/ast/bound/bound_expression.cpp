#include "lingodb/compiler/frontend/ast/bound/bound_expression.h"
namespace lingodb::ast {
/*
 * BoundColumnRefExpression
*/
BoundColumnRefExpression::BoundColumnRefExpression(std::string scope, catalog::NullableType resultType, std::shared_ptr<NamedResult> namedResult, std::string alias) : BoundExpression(TYPE, ExpressionType::BOUND_COLUMN_REF, resultType, alias), scope(scope) {
   this->namedResult = namedResult;
}
std::string BoundColumnRefExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}

/*
 * BoundComparisonExpression
*/

BoundComparisonExpression::BoundComparisonExpression(ExpressionType type, std::string alias, std::shared_ptr<BoundExpression> left, std::vector<std::shared_ptr<BoundExpression>> rightChildren) : BoundExpression(TYPE, type, catalog::Type::boolean(), alias), left(std::move(left)), rightChildren(std::move(rightChildren)) {
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
BoundTargetsExpression::BoundTargetsExpression(std::string alias, std::vector<std::shared_ptr<BoundExpression>> targets, std::vector<std::shared_ptr<NamedResult>> targetColumns) : BoundExpression(TYPE, ExpressionType::BOUND_TARGETS, alias), targets(std::move(targets)), targetColumns(std::move(targetColumns)) {
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
BoundFunctionExpression::BoundFunctionExpression(ExpressionType type, catalog::NullableType resultType, std::string functionName, std::string scope, std::string aliasOrUniqueIdentifier, bool distinct, std::vector<std::shared_ptr<BoundExpression>> arguments, std::shared_ptr<FunctionInfo> functionInfo) : BoundExpression(TYPE, type, resultType, aliasOrUniqueIdentifier), functionName(functionName), scope(scope), aliasOrUniqueIdentifier(aliasOrUniqueIdentifier), distinct(distinct), arguments(arguments), functionInfo(functionInfo) {
   namedResult = functionInfo;
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
BoundCastExpression::BoundCastExpression(catalog::Type resultType, std::string alias, std::shared_ptr<BoundExpression> child, std::optional<LogicalTypeWithMods> logicalTypeWithMods, std::string stringRepr) : BoundExpression(TYPE, ExpressionType::CAST, resultType, alias), child(std::move(child)), logicalTypeWithMods(logicalTypeWithMods), stringRepr(stringRepr)
{
}
std::string BoundCastExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
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
BoundSubqueryExpression::BoundSubqueryExpression(SubqueryType subqueryType, catalog::NullableType resultType, std::string alias, std::shared_ptr<NamedResult> namedResult, std::shared_ptr<analyzer::SQLScope> sqlScope,  std::shared_ptr<TableProducer> subquery,  std::shared_ptr<BoundExpression> testExpr) : BoundExpression(TYPE, ExpressionType::SUBQUERY, resultType, alias), subqueryType(subqueryType), sqlScope(sqlScope), subquery(std::move(subquery)), testExpr(testExpr) {
   this->namedResult = namedResult;
}
std::string BoundSubqueryExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}

BoundCaseExpression::BoundCaseExpression(catalog::NullableType resultType, std::string alias, std::vector<BoundCaseCheck> caseChecks, std::shared_ptr<BoundExpression> elseExpr) : BoundExpression(TYPE, ExpressionType::CASE_EXPR, resultType, alias), caseChecks(std::move(caseChecks)), elseExpr(std::move(elseExpr))  {
}
std::string BoundCaseExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}
} // namespace lingodb::ast