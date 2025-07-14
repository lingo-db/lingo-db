#include "lingodb/compiler/frontend/ast/parsed_expression.h"

#include <cassert>
#include <clang/AST/Type.h>
namespace lingodb::ast {

///ColumnRef
//TODO Find better solution for ColumnRefExpression than duckdb does with columnName and tableName
ColumnRefExpression::ColumnRefExpression(std::string columnName, std::string tableName)
   : ColumnRefExpression(tableName.empty() ? std::vector<std::string>{std::move(columnName)} : std::vector<std::string>{std::move(tableName), std::move(columnName)}) {
}
ColumnRefExpression::ColumnRefExpression(std::string columnName) : ColumnRefExpression(std::vector<std::string>{std::move(columnName)}) {
}

ColumnRefExpression::ColumnRefExpression(std::vector<std::string> columnNames) : ParsedExpression(ExpressionType::COLUMN_REF, TYPE), column_names(columnNames) {
   for (auto& columnName : columnNames) {
      assert(!columnName.empty());
   }
}

std::string ColumnRefExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot{};

   // Create node identifier
   std::string nodeId;
   nodeId.append("node");
   nodeId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this))));

   // Create the label with column names
   std::string label;
   label.append("ColumnRef\\n");

   // Add all column names with dots between them
   for (size_t i = 0; i < column_names.size(); ++i) {
      if (i > 0) {
         label.append(".");
      }
      label.append(column_names[i]);
   }

   // Create the node
   dot.append(nodeId);
   dot.append(" [label=\"");
   dot.append(label);
   if (!alias.empty()) {
      dot.append("\\nalias: ");
      dot.append(alias);
   }
   dot.append("\"];\n");
   return dot;
}

/// ComparisonExpression
ComparisonExpression::ComparisonExpression(ExpressionType type) : ParsedExpression(type, TYPE) {
}
ComparisonExpression::ComparisonExpression(ExpressionType type, std::shared_ptr<ParsedExpression> left, std::shared_ptr<ParsedExpression> right) : ParsedExpression(type, TYPE), left(std::move(left)), rightChildren({std::move(right)}) {
}

ComparisonExpression::ComparisonExpression(ExpressionType type, std::shared_ptr<ParsedExpression> left, std::vector<std::shared_ptr<ParsedExpression>> rightChildren) : ParsedExpression(type, TYPE), left(std::move(left)), rightChildren(rightChildren) {
}

std::string ComparisonExpression::typeToAscii(ExpressionType type) const {
   switch (type) {
      case ExpressionType::COMPARE_EQUAL: return "=";
      case ExpressionType::COMPARE_GREATERTHAN: return ">";
      case ExpressionType::COMPARE_LESSTHAN: return "<";
      case ExpressionType::COMPARE_GREATERTHANOREQUALTO: return ">=";
      case ExpressionType::COMPARE_LESSTHANOREQUALTO: return "<=";
      case ExpressionType::COMPARE_NOTEQUAL: return "<>";
      case ExpressionType::COMPARE_LIKE: return "LIKE";
      case ExpressionType::COMPARE_NOT_LIKE: return "NOT LIKE";
      default: return  "Unknown";
   }
}
std::string ComparisonExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot{};

   // Create node identifier for the comparison expression
   std::string nodeId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this)));

   // Create the node with operator label
   dot += nodeId + " [label=\"σ\\n" + typeToAscii(type) + "\"];\n";

   // Handle left operand
   if (left) {
      std::string leftId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(left.get())));
      dot += nodeId + " -> " + leftId + " [label=\"left\"];\n";
      dot += left->toDotGraph(depth + 1, idGen);
   }

   // Handle all right children
   for (size_t i = 0; i < rightChildren.size(); ++i) {
      if (rightChildren[i]) {
         std::string rightId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(rightChildren[i].get())));
         dot += nodeId + " -> " + rightId + " [label=\"right " + std::to_string(i + 1) + "\"];\n";
         dot += rightChildren[i]->toDotGraph(depth + 1, idGen);
      }
   }

   return dot;
}

/// ConjunctionExpression
ConjunctionExpression::ConjunctionExpression(ExpressionType type) : ParsedExpression(type, TYPE), children() {}
ConjunctionExpression::ConjunctionExpression(ExpressionType type, std::shared_ptr<lingodb::ast::ParsedExpression> left, std::shared_ptr<lingodb::ast::ParsedExpression> right) : ConjunctionExpression(type, std::vector{left, right}) {}
ConjunctionExpression::ConjunctionExpression(ExpressionType type, std::vector<std::shared_ptr<ParsedExpression>> children) : ParsedExpression(type, TYPE), children(std::move(children)) {
}

std::string ConjunctionExpression::typeToAscii(ExpressionType type) const {
   switch (type) {
      case ExpressionType::CONJUNCTION_AND: return "AND";
      case ExpressionType::CONJUNCTION_OR: return "OR";
      default: return "Not found";
   }
}

std::string ConjunctionExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot{};

   // Create node identifier for the conjunction
   std::string nodeId;
   nodeId.append("node");
   nodeId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this))));

   // Create the conjunction node
   dot.append(nodeId);
   dot.append(" [label=\"σ\n");
   dot.append(typeToAscii(type));
   dot.append("\"];\n");

   // Add all child expressions
   for (size_t i = 0; i < children.size(); ++i) {
      if (children[i]) {
         std::string childId;
         childId.append("node");
         childId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(children[i].get()))));

         // Create edge from conjunction to this child
         dot.append(nodeId);
         dot.append(" -> ");
         dot.append(childId);
         dot.append(" [label=\"child ");
         dot.append(std::to_string(i + 1));
         dot.append("\"];\n");

         // Add the child's graph representation
         dot.append(children[i]->toDotGraph(depth + 1, idGen));
      }
   }

   return dot;
}
/// ConstantExpression
ConstantExpression::ConstantExpression() : ParsedExpression(ExpressionType::VALUE_CONSTANT, TYPE) {}

std::string ConstantExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot{};

   // Create node identifier for the constant
   std::string nodeId;
   nodeId.append("node");
   nodeId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this))));

   // Create label with constant value
   std::string label;
   label.append("Constant\\n");

   label.append(value->toString());

   // Create the node
   dot.append(nodeId);
   dot.append(" [label=\"");
   dot.append(label);
   dot.append("\"];\n");

   return dot;
}
/// FunctionExpression
FunctionExpression::FunctionExpression(std::string catalog, std::string schema, std::string functionName, bool isOperator, bool distinct, bool exportState) : ParsedExpression(ExpressionType::FUNCTION, TYPE), catalog(catalog), schema(schema), functionName(functionName), isOperator(isOperator), distinct(distinct), exportState(exportState) {
   auto found = std::find(aggregationFunctions.begin(), aggregationFunctions.end(), functionName);
   if (found != aggregationFunctions.end()) {
      //! TODO Check if this make sense here
      this->type = ExpressionType::AGGREGATE;
   }
}

std::string FunctionExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot{};
   // Create node identifier for the function
   std::string nodeId;
   nodeId.append("node");
   nodeId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this))));

   // Create the function node with its label
   dot.append(nodeId);
   dot.append(" [label=\"Function\nname: ");
   dot.append(functionName);
   if (distinct) {
      dot.append("\\nDISTINCT");
   }
   if (type == ExpressionType::AGGREGATE) {
      dot.append("\nagg\n");
   }
   dot.append("\"];\n");

   // Add all function arguments
   for (size_t i = 0; i < arguments.size(); ++i) {
      if (arguments[i]) {
         std::string argId;
         argId.append("node");
         argId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(arguments[i].get()))));

         // Create edge from function to this argument
         dot.append(nodeId);
         dot.append(" -> ");
         dot.append(argId);
         dot.append(" [label=\"arg ");
         dot.append(std::to_string(i + 1));
         dot.append("\"];\n");

         // Add the argument's graph representation
         dot.append(arguments[i]->toDotGraph(depth + 1, idGen));
      }
   }

   // Add filter if present
   if (filter) {
      std::string filterId;
      filterId.append("node");
      filterId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(filter.get()))));

      dot.append(nodeId);
      dot.append(" -> ");
      dot.append(filterId);
      dot.append(" [label=\"filter\"];\n");
      dot.append(filter->toDotGraph(depth + 1, idGen));
   }

   // Add order by if present
   if (orderBy) {
      std::string orderId;
      orderId.append("node");
      orderId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(orderBy.get()))));

      dot.append(nodeId);
      dot.append(" -> ");
      dot.append(orderId);
      dot.append(" [label=\"order by\"];\n");
      dot.append(orderBy->toDotGraph(depth + 1, idGen));
   }

   return dot;
}
///StarExpression
StarExpression::StarExpression(std::string relationName)
   : ParsedExpression(ExpressionType::STAR, ExpressionClass::STAR), relationName(std::move(relationName)) {
}

std::string StarExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot{};

   // Create node identifier for the star expression
   std::string nodeId;
   nodeId.append("node");
   nodeId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this))));

   // Create the star node with its label
   dot.append(nodeId);
   dot.append(" [label=\"");
   if (!relationName.empty()) {
      dot.append(relationName);
      dot.append(".");
   }
   dot.append("*");
   if (columnsExpr) {
      dot.append("\\nCOLUMNS");
   }
   dot.append("\"];\n");

   // Add expression if present
   if (expr) {
      std::string exprId;
      exprId.append("node");
      exprId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(expr.get()))));

      dot.append(nodeId);
      dot.append(" -> ");
      dot.append(exprId);
      dot.append(" [label=\"expr\"];\n");
      dot.append(expr->toDotGraph(depth + 1, idGen));
   }

   return dot;
}
///TargetsExpression
TargetsExpression::TargetsExpression() : ParsedExpression(ExpressionType::TARGETS, TYPE) {
}

std::string TargetsExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot{};

   // Create node identifier for the targets list
   std::string nodeId;
   nodeId.append("node");
   nodeId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this))));

   // Create the targets list node
   dot.append(nodeId);
   dot.append(" [label=\"Targets List\"];\n");

   // Add all target expressions
   for (size_t i = 0; i < targets.size(); ++i) {
      if (targets[i]) {
         std::string targetId;
         targetId.append("node");
         targetId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(targets[i].get()))));

         // Create edge from targets list to this target
         dot.append(nodeId);
         dot.append(" -> ");
         dot.append(targetId);
         dot.append(" [label=\"target ");
         dot.append(std::to_string(i + 1));
         dot.append("\"];\n");

         // Add the target's graph representation
         dot.append(targets[i]->toDotGraph(depth + 1, idGen));
      }
   }

   return dot;
}

OperatorExpression::OperatorExpression(ExpressionType type, std::shared_ptr<ParsedExpression> left, std::shared_ptr<ParsedExpression> right) : ParsedExpression(type, TYPE), children(std::vector<std::shared_ptr<ParsedExpression>>{}) {
   children.push_back(std::move(left));
   children.push_back(std::move(right));
}
std::string OperatorExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot{};

   // Create node identifier for the operator expression
   std::string nodeId;
   nodeId.append("node");
   nodeId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this))));

   // Create the operator node with its label
   dot.append(nodeId);
   dot.append(" [label=\"Operator\\n");
   switch (type) {
      case ExpressionType::OPERATOR_PLUS:
         dot.append("+");
         break;
      case ExpressionType::OPERATOR_MINUS:
         dot.append("-");
         break;
      case ExpressionType::OPERATOR_TIMES:
         dot.append("*");
         break;
      case ExpressionType::OPERATOR_DIVIDE:
         dot.append("/");
         break;
      case ExpressionType::OPERATOR_MOD:
         dot.append("%");
         break;
      case ExpressionType::OPERATOR_NOT:
         dot.append("NOT");
         break;
      case ExpressionType::OPERATOR_IS_NULL:
         dot.append("IS NULL");
         break;
      case ExpressionType::OPERATOR_IS_NOT_NULL:
         dot.append("IS NOT NULL");
         break;
      default:
         dot.append("Unknown Operator");
         break;
   }
   dot.append("\"];\n");

   // Add all child expressions
   for (size_t i = 0; i < children.size(); ++i) {
      if (children[i]) {
         std::string childId;
         childId.append("node");
         childId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(children[i].get()))));

         // Create edge from operator to this child
         dot.append(nodeId);
         dot.append(" -> ");
         dot.append(childId);
         dot.append(" [label=\"child ");
         dot.append(std::to_string(i + 1));
         dot.append("\"];\n");

         // Add the child's graph representation
         dot.append(children[i]->toDotGraph(depth + 1, idGen));
      }
   }

   return dot;
}

CastExpression::CastExpression(LogicalTypeWithMods logicalTypeWithMods, std::shared_ptr<ParsedExpression> child) : ParsedExpression(ExpressionType::CAST, TYPE), logicalTypeWithMods(logicalTypeWithMods), child(std::move(child)) {
}
std::string CastExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot{};

   // Create node identifier for the cast expression
   std::string nodeId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this)));

   // Create the label with cast information
   std::string label = "Cast\\nType: ";
   if (logicalTypeWithMods.has_value()) {
      switch (logicalTypeWithMods.value().logicalType) {
         case LogicalType::DATE:
            label += "DATE";
            break;
            // TODO: Add other logical types as needed
         default:
            label += "Unknown";
            break;
      }
   }

   // Create the node
   dot += nodeId + " [label=\"" + label + "\"];\n";

   // Add the child expression if present
   if (child) {
      std::string childId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(child.get())));

      // Create edge from cast to its child
      dot += nodeId + " -> " + childId + " [label=\"child\"];\n";

      // Add the child's graph representation
      dot += child->toDotGraph(depth + 1, idGen);
   }

   return dot;
}

BetweenExpression::BetweenExpression(ExpressionType type, std::shared_ptr<ParsedExpression> input, std::shared_ptr<ParsedExpression> lower, std::shared_ptr<ParsedExpression> upper) : ParsedExpression(type, TYPE), input(input), lower(lower), upper(upper) {
   assert(lower != nullptr && upper != nullptr && input != nullptr);
   assert(type == ExpressionType::COMPARE_BETWEEN || type == ExpressionType::COMPARE_NOT_BETWEEN);
}

std::string BetweenExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot{};

   // Create node identifier for the between expression
   std::string nodeId;
   nodeId.append("node");
   nodeId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this))));

   // Create the node with operator label
   dot.append(nodeId);
   dot.append(" [label=\"");
   dot.append(type == ExpressionType::COMPARE_BETWEEN ? "BETWEEN" : "NOT BETWEEN");
   if (asymmetric) {
      dot.append("\\nASYMMETRIC");
   }
   dot.append("\"];\n");

   // Add input expression
   if (input) {
      std::string inputId;
      inputId.append("node");
      inputId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(input.get()))));

      dot.append(nodeId);
      dot.append(" -> ");
      dot.append(inputId);
      dot.append(" [label=\"input\"];\n");
      dot.append(input->toDotGraph(depth + 1, idGen));
   }

   // Add lower bound expression
   if (lower) {
      std::string lowerId;
      lowerId.append("node");
      lowerId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(lower.get()))));

      dot.append(nodeId);
      dot.append(" -> ");
      dot.append(lowerId);
      dot.append(" [label=\"lower\"];\n");
      dot.append(lower->toDotGraph(depth + 1, idGen));
   }

   // Add upper bound expression
   if (upper) {
      std::string upperId;
      upperId.append("node");
      upperId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(upper.get()))));

      dot.append(nodeId);
      dot.append(" -> ");
      dot.append(upperId);
      dot.append(" [label=\"upper\"];\n");
      dot.append(upper->toDotGraph(depth + 1, idGen));
   }

   return dot;
}

SubqueryExpression::SubqueryExpression(SubqueryType subQueryType, std::shared_ptr<TableProducer> subquery) : ParsedExpression(ExpressionType::SUBQUERY,TYPE), subQueryType(subQueryType),  subquery(subquery) {
}
std::string SubqueryExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot;

   // Create node identifier for the subquery expression
   std::string nodeId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this)));

   // Create the subquery node with its label
   dot += nodeId + " [label=\"Subquery\"];\n";

   // Add the subquery TableProducer if present
   if (subquery) {
      std::string subqueryId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(subquery.get())));
      dot += nodeId + " -> " + subqueryId + " [label=\"subquery\"];\n";
      dot += subquery->toDotGraph(depth + 1, idGen);
   }

   return dot;
}

CaseExpression::CaseExpression(std::vector<CaseCheck> caseChecks, std::shared_ptr<ParsedExpression> elseExpr) : ParsedExpression(ExpressionType::CASE_EXPR, TYPE), caseChecks(std::move(caseChecks)), elseExpr(std::move(elseExpr)) {
}

std::string CaseExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot;

   // Node for the CASE expression
   std::string nodeId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this)));
   dot += nodeId + " [label=\"CASE\"];\n";

   // Add all case checks (WHEN/THEN pairs)
   for (size_t i = 0; i < caseChecks.size(); ++i) {
      // WHEN
      if (caseChecks[i].whenExpr) {
         std::string whenId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(caseChecks[i].whenExpr.get())));
         dot += nodeId + " -> " + whenId + " [label=\"when " + std::to_string(i + 1) + "\"];\n";
         dot += caseChecks[i].whenExpr->toDotGraph(depth + 1, idGen);
      }
      // THEN
      if (caseChecks[i].thenExpr) {
         std::string thenId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(caseChecks[i].thenExpr.get())));
         dot += nodeId + " -> " + thenId + " [label=\"then " + std::to_string(i + 1) + "\"];\n";
         dot += caseChecks[i].thenExpr->toDotGraph(depth + 1, idGen);
      }
   }

   // Add ELSE expression if present
   if (elseExpr) {
      std::string elseId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(elseExpr.get())));
      dot += nodeId + " -> " + elseId + " [label=\"else\"];\n";
      dot += elseExpr->toDotGraph(depth + 1, idGen);
   }

   return dot;
}

} // namespace lingodb::ast
