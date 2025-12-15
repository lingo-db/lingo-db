#include "lingodb/compiler/frontend/ast/parsed_expression.h"

#include "lingodb/compiler/frontend/ast/result_modifier.h"

#include <cassert>
namespace lingodb::ast {

size_t ParsedExpression::hash() {
   size_t result = std::hash<uint8_t>{}(static_cast<uint8_t>(type));
   result = result * 31 + std::hash<uint8_t>{}(static_cast<uint8_t>(exprClass));
   return result;
}
bool ParsedExpression::operator==(ParsedExpression& other) {
   return type == other.type && exprClass == other.exprClass;
}

///ColumnRef
ColumnRefExpression::ColumnRefExpression(std::string columnName, std::string tableName)
   : ColumnRefExpression(tableName.empty() ? std::vector<std::string>{std::move(columnName)} : std::vector<std::string>{std::move(tableName), std::move(columnName)}) {
}
ColumnRefExpression::ColumnRefExpression(std::string columnName) : ColumnRefExpression(std::vector<std::string>{std::move(columnName)}) {
}

ColumnRefExpression::ColumnRefExpression(std::vector<std::string> columnNames) : ParsedExpression(ExpressionType::COLUMN_REF, cType), columnNames(columnNames) {
}

size_t ColumnRefExpression::hash() {
   size_t result = ParsedExpression::hash();
   for (const auto& name : columnNames) {
      result = result * 31 + std::hash<std::string>{}(name);
   }

   return result;
}
bool ColumnRefExpression::operator==(ParsedExpression& other) {
   if (!ParsedExpression::operator==(other)) return false;

   const auto& otherRef = static_cast<ColumnRefExpression&>(other);
   return columnNames == otherRef.columnNames;
}

/// ComparisonExpression
ComparisonExpression::ComparisonExpression(ExpressionType type, std::shared_ptr<ParsedExpression> left, std::shared_ptr<ParsedExpression> right) : ParsedExpression(type, cType), left(std::move(left)), rightChildren({std::move(right)}) {
}

ComparisonExpression::ComparisonExpression(ExpressionType type, std::shared_ptr<ParsedExpression> left, std::vector<std::shared_ptr<ParsedExpression>> rightChildren) : ParsedExpression(type, cType), left(std::move(left)), rightChildren(rightChildren) {
}
size_t ComparisonExpression::hash() {
   size_t result = ParsedExpression::hash();

   // Hash the left expression
   if (left) {
      result ^= left->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
   }

   // Hash all right children expressions
   for (const auto& child : rightChildren) {
      if (child) {
         result ^= child->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
      }
   }

   return result;
}

bool ComparisonExpression::operator==(ParsedExpression& other) {
   if (!ParsedExpression::operator==(other)) return false;

   auto& otherComp = static_cast<ComparisonExpression&>(other);

   // Compare left expressions (handle nulls and pointer-equality)
   if (!((left == otherComp.left) ||
         (left && otherComp.left && *left == *otherComp.left))) {
      return false;
   }

   // Compare number of right children
   if (rightChildren.size() != otherComp.rightChildren.size()) {
      return false;
   }

   // Compare each right child expression (handle nulls and pointer-equality)
   for (size_t i = 0; i < rightChildren.size(); ++i) {
      const auto& a = rightChildren[i];
      const auto& b = otherComp.rightChildren[i];
      if (!((a == b) || (a && b && *a == *b))) {
         return false;
      }
   }

   return true;
}

/// ConjunctionExpression
ConjunctionExpression::ConjunctionExpression(ExpressionType type, std::shared_ptr<lingodb::ast::ParsedExpression> left, std::shared_ptr<lingodb::ast::ParsedExpression> right) : ConjunctionExpression(type, std::vector{left, right}) {}
ConjunctionExpression::ConjunctionExpression(ExpressionType type, std::vector<std::shared_ptr<ParsedExpression>> children) : ParsedExpression(type, cType), children(std::move(children)) {
}

size_t ConjunctionExpression::hash() {
   size_t result = ParsedExpression::hash();
   // Hash all children expressions using built-in hash combine
   for (const auto& child : children) {
      result ^= child->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
   }
   return result;
}

bool ConjunctionExpression::operator==(ParsedExpression& other) {
   if (!ParsedExpression::operator==(other)) {
      return false;
   }
   auto& otherConj = static_cast<ConjunctionExpression&>(other);
   // Compare the number of children first
   if (children.size() != otherConj.children.size()) {
      return false;
   }
   // Compare each child expression
   for (size_t i = 0; i < children.size(); i++) {
      if ((*children[i] != *otherConj.children[i])) {
         return false;
      }
   }
   return true;
}

/// ConstantExpression
ConstantExpression::ConstantExpression() : ParsedExpression(ExpressionType::VALUE_CONSTANT, cType) {}

size_t ConstantExpression::hash() {
   size_t result = ParsedExpression::hash();
   if (value) {
      result = result * 31 + value->hash();
   }
   return result;
}

bool ConstantExpression::operator==(ParsedExpression& other) {
   if (!ParsedExpression::operator==(other)) return false;

   const auto& otherConst = static_cast<const ConstantExpression&>(other);

   // Handle cases where one value is null and the other isn't
   if ((value && !otherConst.value) || (!value && otherConst.value)) {
      return false;
   }

   // If both values are null, they're equal
   if (!value && !otherConst.value) {
      return true;
   }

   // Compare the actual values
   return *value == *otherConst.value;
}
/// FunctionExpression
FunctionExpression::FunctionExpression(std::string catalog, std::string schema, std::string functionName, bool isOperator, bool distinct, bool exportState) : ParsedExpression(ExpressionType::FUNCTION, cType), catalog(catalog), schema(schema), functionName(functionName), isOperator(isOperator), distinct(distinct), exportState(exportState) {
   auto found = std::find(aggregationFunctions.begin(), aggregationFunctions.end(), functionName);
   if (found != aggregationFunctions.end()) {
      this->type = ExpressionType::AGGREGATE;
   }
}

size_t FunctionExpression::hash() {
   size_t result = ParsedExpression::hash();
   // Hash basic function properties
   result = result * 31 + std::hash<std::string>{}(catalog);
   result = result * 31 + std::hash<std::string>{}(schema);
   result = result * 31 + std::hash<std::string>{}(functionName);
   result = result * 31 + std::hash<bool>{}(isOperator);
   result = result * 31 + std::hash<bool>{}(distinct);
   result = result * 31 + std::hash<bool>{}(exportState);
   result = result * 31 + std::hash<bool>{}(star);

   // Hash function arguments
   for (const auto& arg : arguments) {
      result = result * 31 + arg->hash();
   }

   // Hash optional components
   if (filter) {
      result = result * 31 + filter->hash();
   }
   if (orderBy) {
      result = result * 31 + orderBy->hash();
   }

   return result;
}
bool FunctionExpression::operator==(ParsedExpression& other) {
   if (!ParsedExpression::operator==(other)) return false;
   const auto& otherFunc = static_cast<const FunctionExpression&>(other);
   if (catalog != otherFunc.catalog ||
       schema != otherFunc.schema ||
       functionName != otherFunc.functionName ||
       isOperator != otherFunc.isOperator ||
       distinct != otherFunc.distinct ||
       exportState != otherFunc.exportState ||
       arguments.size() != otherFunc.arguments.size()) {
      return false;
   }

   // Compare function arguments
   for (size_t i = 0; i < arguments.size(); i++) {
      if (*arguments[i] != *(otherFunc.arguments[i])) {
         return false;
      }
   }

   // Compare optional filter and orderBy if present
   if ((filter && !otherFunc.filter) || (!filter && otherFunc.filter)) {
      return false;
   }
   if (filter && *filter != *otherFunc.filter) {
      return false;
   }

   if ((orderBy && !otherFunc.orderBy) || (!orderBy && otherFunc.orderBy)) {
      return false;
   }
   if (orderBy && *orderBy != *otherFunc.orderBy) {
      return false;
   }

   return star == otherFunc.star;
}

///StarExpression
StarExpression::StarExpression(std::string relationName)
   : ParsedExpression(ExpressionType::STAR, ExpressionClass::STAR), relationName(std::move(relationName)) {
}

size_t StarExpression::hash() {
   size_t result = ParsedExpression::hash();
   // Hash the relation name
   result ^= std::hash<std::string>{}(relationName) + 0x9e3779b9 + (result << 6) + (result >> 2);
   // Hash the expr if it exists
   if (expr) {
      result ^= expr->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
   }
   // Hash the columnsExpr boolean
   result ^= std::hash<bool>{}(columnsExpr) + 0x9e3779b9 + (result << 6) + (result >> 2);
   return result;
}

bool StarExpression::operator==(ParsedExpression& other) {
   if (!ParsedExpression::operator==(other)) {
      return false;
   }
   auto& otherStar = static_cast<StarExpression&>(other);

   // Compare relation name
   if (relationName != otherStar.relationName) {
      return false;
   }

   // Compare columnsExpr
   if (columnsExpr != otherStar.columnsExpr) {
      return false;
   }

   // Compare expr (handle null cases)
   if (expr == otherStar.expr) {
      return true; // Both null or same pointer
   }
   if (!expr || !otherStar.expr) {
      return false; // One is null, other isn't
   }
   return *expr == *otherStar.expr;
}

OperatorExpression::OperatorExpression(ExpressionType type, std::shared_ptr<ParsedExpression> left) : ParsedExpression(type, cType), children(std::vector{left}) {
}
OperatorExpression::OperatorExpression(ExpressionType type, std::shared_ptr<ParsedExpression> left, std::shared_ptr<ParsedExpression> right) : ParsedExpression(type, cType), children(std::vector{left, right}) {
}
OperatorExpression::OperatorExpression(std::string opString, std::shared_ptr<ParsedExpression> left, std::shared_ptr<ParsedExpression> right) : ParsedExpression(ExpressionType::OPERATOR_UNKNOWN, cType), children(std::vector{left, right}), opString(opString) {
}
size_t OperatorExpression::hash() {
   size_t result = ParsedExpression::hash();

   // Hash the operator string
   result ^= std::hash<std::string>{}(opString) + 0x9e3779b9 + (result << 6) + (result >> 2);

   // Hash all children expressions
   for (const auto& child : children) {
      result ^= child->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
   }

   return result;
}

bool OperatorExpression::operator==(ParsedExpression& other) {
   if (!ParsedExpression::operator==(other)) {
      return false;
   }

   auto& otherOp = static_cast<OperatorExpression&>(other);

   // Compare operator strings
   if (opString != otherOp.opString) {
      return false;
   }

   // Compare children
   if (children.size() != otherOp.children.size()) {
      return false;
   }

   for (size_t i = 0; i < children.size(); i++) {
      if ((*children[i] != *otherOp.children[i])) {
         return false;
      }
   }

   return true;
}

CastExpression::CastExpression(LogicalTypeWithMods logicalTypeWithMods, std::shared_ptr<ParsedExpression> child) : ParsedExpression(ExpressionType::CAST, cType), logicalTypeWithMods(logicalTypeWithMods), child(std::move(child)) {
}

size_t CastExpression::hash() {
   size_t result = ParsedExpression::hash();

   // Hash the logical type with mods if present
   if (logicalTypeWithMods) {
      // Hash the logical type
      result ^= std::hash<uint8_t>{}(static_cast<uint8_t>(logicalTypeWithMods->logicalTypeId)) +
         0x9e3779b9 + (result << 6) + (result >> 2);

      // Hash type modifiers
      for (const auto& mod : logicalTypeWithMods->typeModifiers) {
         if (mod) {
            result ^= mod->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
         }
      }
   }

   // Hash the child expression
   if (child) {
      result ^= child->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
   }

   return result;
}

bool CastExpression::operator==(ParsedExpression& other) {
   if (!ParsedExpression::operator==(other)) {
      return false;
   }

   auto& otherCast = static_cast<CastExpression&>(other);

   // Compare logical type with mods
   if (logicalTypeWithMods.has_value() != otherCast.logicalTypeWithMods.has_value()) {
      return false;
   }

   if (logicalTypeWithMods) {
      if (logicalTypeWithMods->logicalTypeId != otherCast.logicalTypeWithMods->logicalTypeId) {
         return false;
      }

      if (logicalTypeWithMods->typeModifiers.size() !=
          otherCast.logicalTypeWithMods->typeModifiers.size()) {
         return false;
      }

      for (size_t i = 0; i < logicalTypeWithMods->typeModifiers.size(); i++) {
         if ((*logicalTypeWithMods->typeModifiers[i] !=
              *otherCast.logicalTypeWithMods->typeModifiers[i])) {
            return false;
         }
      }
   }

   // Compare child expressions
   if (child == otherCast.child) {
      return true; // Both null or same pointer
   }
   if (!child || !otherCast.child) {
      return false; // One is null, other isn't
   }
   return *child == *otherCast.child;
}

WindowFrame::WindowFrame(WindowFrameType start) : start(start) {
}
WindowFrame::WindowFrame(WindowFrameType start, std::shared_ptr<ParsedExpression> startExpr) : start(start), startExpr(startExpr) {
}
WindowExpression::WindowExpression() : ParsedExpression(ExpressionType::WINDOW_INVALID, cType) {
}

BetweenExpression::BetweenExpression(ExpressionType type, std::shared_ptr<ParsedExpression> input, std::shared_ptr<ParsedExpression> lower, std::shared_ptr<ParsedExpression> upper) : ParsedExpression(type, cType), input(input), lower(lower), upper(upper) {
   assert(lower != nullptr && upper != nullptr && input != nullptr);
   assert(type == ExpressionType::COMPARE_BETWEEN || type == ExpressionType::COMPARE_NOT_BETWEEN);
}

size_t BetweenExpression::hash() {
   size_t result = ParsedExpression::hash();

   // Hash input expression
   if (input) {
      result ^= input->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
   }

   // Hash lower bound
   if (lower) {
      result ^= lower->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
   }

   // Hash upper bound
   if (upper) {
      result ^= upper->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
   }

   // Hash asymmetric flag
   result ^= std::hash<bool>{}(asymmetric) + 0x9e3779b9 + (result << 6) + (result >> 2);

   return result;
}

bool BetweenExpression::operator==(ParsedExpression& other) {
   if (!ParsedExpression::operator==(other)) {
      return false;
   }

   auto& otherBetween = static_cast<BetweenExpression&>(other);

   // Compare asymmetric flag
   if (asymmetric != otherBetween.asymmetric) {
      return false;
   }

   // Compare input expressions
   if (!((input == otherBetween.input) ||
         (input && otherBetween.input && *input == *otherBetween.input))) {
      return false;
   }

   // Compare lower bounds
   if (!((lower == otherBetween.lower) ||
         (lower && otherBetween.lower && *lower == *otherBetween.lower))) {
      return false;
   }

   // Compare upper bounds
   if (!((upper == otherBetween.upper) ||
         (upper && otherBetween.upper && *upper == *otherBetween.upper))) {
      return false;
   }

   return true;
}

SubqueryExpression::SubqueryExpression(SubqueryType subQueryType, std::shared_ptr<TableProducer> subquery) : ParsedExpression(ExpressionType::SUBQUERY, cType), subQueryType(subQueryType), subquery(subquery) {
}

size_t SubqueryExpression::hash() {
   size_t result = ParsedExpression::hash();

   // Hash subquery type
   result ^= std::hash<uint8_t>{}(static_cast<uint8_t>(subQueryType)) +
      0x9e3779b9 + (result << 6) + (result >> 2);

   // Hash subquery
   if (subquery) {
      //result ^= subquery->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
   }

   // Hash test expression
   if (testExpr) {
      result ^= testExpr->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
   }

   return result;
}

bool SubqueryExpression::operator==(ParsedExpression& other) {
   if (!ParsedExpression::operator==(other)) {
      return false;
   }

   auto& otherSubquery = static_cast<SubqueryExpression&>(other);

   // Compare subquery types
   if (subQueryType != otherSubquery.subQueryType) {
      return false;
   }

   // Compare subqueries
   /*if (!((subquery == otherSubquery.subquery) ||
         (subquery && otherSubquery.subquery && *subquery == *otherSubquery.subquery))) {
      return false;
         }*/

   // Compare test expressions
   if (!((testExpr == otherSubquery.testExpr) ||
         (testExpr && otherSubquery.testExpr && *testExpr == *otherSubquery.testExpr))) {
      return false;
   }

   return true;
}

CaseExpression::CaseExpression(std::optional<std::shared_ptr<ParsedExpression>> caseExpr, std::vector<CaseCheck> caseChecks, std::shared_ptr<ParsedExpression> elseExpr) : ParsedExpression(ExpressionType::CASE_EXPR, cType), caseChecks(std::move(caseChecks)), caseExpr(caseExpr), elseExpr(std::move(elseExpr)) {
}

size_t CaseExpression::hash() {
   size_t result = ParsedExpression::hash();

   // Hash caseExpr if present
   if (caseExpr && *caseExpr) {
      result ^= (*caseExpr)->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
   }

   // Hash all caseChecks (WHEN/THEN pairs)
   for (const auto& check : caseChecks) {
      if (check.whenExpr) {
         result ^= check.whenExpr->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
      }
      if (check.thenExpr) {
         result ^= check.thenExpr->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
      }
   }

   // Hash elseExpr if present
   if (elseExpr) {
      result ^= elseExpr->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
   }

   return result;
}

bool CaseExpression::operator==(ParsedExpression& other) {
   if (!ParsedExpression::operator==(other)) {
      return false;
   }
   auto& otherCase = static_cast<CaseExpression&>(other);

   // Compare caseExpr
   if (caseExpr.has_value() != otherCase.caseExpr.has_value()) {
      return false;
   }
   if (caseExpr && *caseExpr && otherCase.caseExpr && *otherCase.caseExpr) {
      if ((**caseExpr != **otherCase.caseExpr)) {
         return false;
      }
   }

   // Compare caseChecks
   if (caseChecks.size() != otherCase.caseChecks.size()) {
      return false;
   }
   for (size_t i = 0; i < caseChecks.size(); ++i) {
      const auto& a = caseChecks[i];
      const auto& b = otherCase.caseChecks[i];
      if (!((a.whenExpr == b.whenExpr) ||
            (a.whenExpr && b.whenExpr && *a.whenExpr == *b.whenExpr))) {
         return false;
      }
      if (!((a.thenExpr == b.thenExpr) ||
            (a.thenExpr && b.thenExpr && *a.thenExpr == *b.thenExpr))) {
         return false;
      }
   }

   // Compare elseExpr
   if (!((elseExpr == otherCase.elseExpr) ||
         (elseExpr && otherCase.elseExpr && *elseExpr == *otherCase.elseExpr))) {
      return false;
   }

   return true;
}

} // namespace lingodb::ast
