#include "lingodb/compiler/frontend/ast/parsed_expression.h"

#include "lingodb/compiler/frontend/ast/result_modifier.h"

#include <cassert>
#include <clang/AST/Type.h>
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
//TODO Find better solution for ColumnRefExpression than duckdb does with columnName and tableName
ColumnRefExpression::ColumnRefExpression(std::string columnName, std::string tableName)
   : ColumnRefExpression(tableName.empty() ? std::vector<std::string>{std::move(columnName)} : std::vector<std::string>{std::move(tableName), std::move(columnName)}) {
}
ColumnRefExpression::ColumnRefExpression(std::string columnName) : ColumnRefExpression(std::vector<std::string>{std::move(columnName)}) {
}

ColumnRefExpression::ColumnRefExpression(std::vector<std::string> columnNames) : ParsedExpression(ExpressionType::COLUMN_REF, TYPE), columnNames(columnNames) {
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
   for (size_t i = 0; i < columnNames.size(); ++i) {
      if (i > 0) {
         label.append(".");
      }
      label.append(columnNames[i]);
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
      default: return "Unknown";
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
      if (!(*children[i] == *otherConj.children[i])) {
         return false;
      }
   }
   return true;
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
   label.append("\n" + alias);

   // Create the node
   dot.append(nodeId);
   dot.append(" [label=\"");
   dot.append(label);
   dot.append("\"];\n");

   return dot;
}

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
   dot.append("\\n alias: ");
   dot.append(alias);
   dot.append("\n");
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
size_t TargetsExpression::hash() {
   size_t result = ParsedExpression::hash();

   // Hash all target expressions
   for (const auto& target : targets) {
      result ^= target->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
   }

   return result;
}

bool TargetsExpression::operator==(ParsedExpression& other) {
   if (!ParsedExpression::operator==(other)) {
      return false;
   }

   auto& otherTargets = static_cast<TargetsExpression&>(other);

   // Compare targets
   if (targets.size() != otherTargets.targets.size()) {
      return false;
   }
   for (size_t i = 0; i < targets.size(); i++) {
      if (!(*targets[i] == *otherTargets.targets[i])) {
         return false;
      }
   }

   // Compare distinct expressions


   return otherTargets.distinct == distinct;
}

OperatorExpression::OperatorExpression(ExpressionType type, std::shared_ptr<ParsedExpression> left) : ParsedExpression(type, TYPE), children(std::vector{left}) {
}
OperatorExpression::OperatorExpression(ExpressionType type, std::shared_ptr<ParsedExpression> left, std::shared_ptr<ParsedExpression> right) : ParsedExpression(type, TYPE), children(std::vector{left, right}) {
}
OperatorExpression::OperatorExpression(std::string opString, std::shared_ptr<ParsedExpression> left, std::shared_ptr<ParsedExpression> right) : ParsedExpression(ExpressionType::OPERATOR_UNKNOWN, TYPE), opString(opString), children(std::vector{left, right}) {
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
   dot.append("\\n alias: ");
   dot.append(alias);
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
      if (!(*children[i] == *otherOp.children[i])) {
         return false;
      }
   }

   return true;
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
size_t CastExpression::hash() {
    size_t result = ParsedExpression::hash();

    // Hash the logical type with mods if present
    if (logicalTypeWithMods) {
        // Hash the logical type
        result ^= std::hash<uint8_t>{}(static_cast<uint8_t>(logicalTypeWithMods->logicalType)) +
                  0x9e3779b9 + (result << 6) + (result >> 2);

        // Hash type modifiers
        for (const auto& mod : logicalTypeWithMods->typeModifiers) {
            if (mod) {
                result ^= mod->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
            }
        }
    }

    // Hash the optional interval type if present
    if (optInterval) {
        result ^= std::hash<uint8_t>{}(static_cast<uint8_t>(*optInterval)) +
                  0x9e3779b9 + (result << 6) + (result >> 2);
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
        if (logicalTypeWithMods->logicalType != otherCast.logicalTypeWithMods->logicalType) {
            return false;
        }

        if (logicalTypeWithMods->typeModifiers.size() !=
            otherCast.logicalTypeWithMods->typeModifiers.size()) {
            return false;
        }

        for (size_t i = 0; i < logicalTypeWithMods->typeModifiers.size(); i++) {
            if (!(*logicalTypeWithMods->typeModifiers[i] ==
                  *otherCast.logicalTypeWithMods->typeModifiers[i])) {
                return false;
            }
        }
    }

    // Compare optional interval
    if (optInterval != otherCast.optInterval) {
        return false;
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

WindowBoundary::WindowBoundary(WindowBoundaryType start) : start(start) {
}
WindowBoundary::WindowBoundary(WindowBoundaryType start, std::shared_ptr<ParsedExpression> startExpr) : start(start), startExpr(startExpr) {
}
WindowExpression::WindowExpression() : ParsedExpression(ExpressionType::WINDOW_INVALID, TYPE) {
}
std::string WindowExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot{};

   // Create node identifier for the window expression
   std::string nodeId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this)));

   // Create the window node with its basic information
   dot += nodeId + " [label=\"Window";
   if (!alias.empty()) {
      dot += "\\nalias: " + alias;
   }

   // Add window function details if present
   if (functionExpression) {
      dot += "\\nFunction: " + functionExpression->functionName;
      if (distinct) {
         dot += "\\nDISTINCT";
      }
      if (ignoreNulls) {
         dot += "\\nIGNORE NULLS";
      }
   }
   dot += "\"];\n";

   // Add function expression if present
   if (functionExpression) {
      std::string funcId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(functionExpression.get())));
      dot += nodeId + " -> " + funcId + " [label=\"function\"];\n";
      dot += functionExpression->toDotGraph(depth + 1, idGen);
   }

   // Add partition expressions
   for (size_t i = 0; i < partitions.size(); i++) {
      if (partitions[i]) {
         std::string partId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(partitions[i].get())));
         dot += nodeId + " -> " + partId + " [label=\"partition " + std::to_string(i + 1) + "\"];\n";
         dot += partitions[i]->toDotGraph(depth + 1, idGen);
      }
   }

   // Add order by expressions
   if (order.has_value()) {
      std::string orderId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(order.value().get())));
      dot += nodeId + " -> " + orderId + " [label=\"order by\"];\n";
      dot += order.value()->toDotGraph(depth + 1, idGen);
   }

   // Add filter expression if present
   if (filter) {
      std::string filterId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(filter.get())));
      dot += nodeId + " -> " + filterId + " [label=\"filter\"];\n";
      dot += filter->toDotGraph(depth + 1, idGen);
   }

   // Add window boundary expressions if present
   if (startExpr) {
      std::string startId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(startExpr.get())));
      dot += nodeId + " -> " + startId + " [label=\"start\"];\n";
      dot += startExpr->toDotGraph(depth + 1, idGen);
   }

   if (endExpr) {
      std::string endId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(endExpr.get())));
      dot += nodeId + " -> " + endId + " [label=\"end\"];\n";
      dot += endExpr->toDotGraph(depth + 1, idGen);
   }

   // Add offset expression if present
   if (offsetExpr) {
      std::string offsetId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(offsetExpr.get())));
      dot += nodeId + " -> " + offsetId + " [label=\"offset\"];\n";
      dot += offsetExpr->toDotGraph(depth + 1, idGen);
   }

   // Add default expression if present
   if (defaultExpr) {
      std::string defaultId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(defaultExpr.get())));
      dot += nodeId + " -> " + defaultId + " [label=\"default\"];\n";
      dot += defaultExpr->toDotGraph(depth + 1, idGen);
   }

   return dot;
}
size_t WindowExpression::hash() {
    size_t result = ParsedExpression::hash();

    // Hash function expression
    if (functionExpression) {
        result ^= functionExpression->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
    }

    // Hash partition expressions
    for (const auto& partition : partitions) {
        if (partition) {
            result ^= partition->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
        }
    }

    // Hash order by modifier
    if (order && *order) {
       //TODO
       // result ^= (*order)->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
    }

    // Hash filter expression
    if (filter) {
        result ^= filter->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
    }

    // Hash boolean flags
    result ^= std::hash<bool>{}(ignoreNulls) + 0x9e3779b9 + (result << 6) + (result >> 2);
    result ^= std::hash<bool>{}(distinct) + 0x9e3779b9 + (result << 6) + (result >> 2);

    // Hash window boundary
    if (windowBoundary) {
        result ^= std::hash<uint8_t>{}(static_cast<uint8_t>(windowBoundary->windowMode)) +
                  0x9e3779b9 + (result << 6) + (result >> 2);
        result ^= std::hash<uint8_t>{}(static_cast<uint8_t>(windowBoundary->start)) +
                  0x9e3779b9 + (result << 6) + (result >> 2);
        result ^= std::hash<uint8_t>{}(static_cast<uint8_t>(windowBoundary->end)) +
                  0x9e3779b9 + (result << 6) + (result >> 2);
        if (windowBoundary->startExpr) {
            result ^= windowBoundary->startExpr->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
        }
        if (windowBoundary->endExpr) {
            result ^= windowBoundary->endExpr->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
        }
    }

    // Hash expressions
    if (startExpr) {
        result ^= startExpr->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
    }
    if (endExpr) {
        result ^= endExpr->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
    }
    if (offsetExpr) {
        result ^= offsetExpr->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
    }
    if (defaultExpr) {
        result ^= defaultExpr->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
    }

    // Hash argument orders
    if (argOrders) {
       //TODO
        //result ^= argOrders->hash() + 0x9e3779b9 + (result << 6) + (result >> 2);
    }

    return result;
}

bool WindowExpression::operator==(ParsedExpression& other) {
    if (!ParsedExpression::operator==(other)) {
        return false;
    }

    auto& otherWindow = static_cast<WindowExpression&>(other);

    // Compare function expressions
    if (!((functionExpression == otherWindow.functionExpression) ||
          (functionExpression && otherWindow.functionExpression &&
           *functionExpression == *otherWindow.functionExpression))) {
        return false;
    }

    // Compare partitions
    if (partitions.size() != otherWindow.partitions.size()) {
        return false;
    }
    for (size_t i = 0; i < partitions.size(); i++) {
        if (!((partitions[i] == otherWindow.partitions[i]) ||
              (partitions[i] && otherWindow.partitions[i] &&
               *partitions[i] == *otherWindow.partitions[i]))) {
            return false;
        }
    }

    // Compare order
    if (order.has_value() != otherWindow.order.has_value()) {
        return false;
    }
   //TODO
    if (order && *order && otherWindow.order && *otherWindow.order
        //&& !(**order == **otherWindow.order)
        ) {
        return false;
    }

    // Compare filter
    if (!((filter == otherWindow.filter) ||
          (filter && otherWindow.filter && *filter == *otherWindow.filter))) {
        return false;
    }

    // Compare boolean flags
    if (ignoreNulls != otherWindow.ignoreNulls || distinct != otherWindow.distinct) {
        return false;
    }

    // Compare window boundary
    if ((windowBoundary == nullptr) != (otherWindow.windowBoundary == nullptr)) {
        return false;
    }
    if (windowBoundary) {
        if (windowBoundary->windowMode != otherWindow.windowBoundary->windowMode ||
            windowBoundary->start != otherWindow.windowBoundary->start ||
            windowBoundary->end != otherWindow.windowBoundary->end) {
            return false;
        }
        if (!((windowBoundary->startExpr == otherWindow.windowBoundary->startExpr) ||
              (windowBoundary->startExpr && otherWindow.windowBoundary->startExpr &&
               *windowBoundary->startExpr == *otherWindow.windowBoundary->startExpr))) {
            return false;
        }
        if (!((windowBoundary->endExpr == otherWindow.windowBoundary->endExpr) ||
              (windowBoundary->endExpr && otherWindow.windowBoundary->endExpr &&
               *windowBoundary->endExpr == *otherWindow.windowBoundary->endExpr))) {
            return false;
        }
    }

    // Compare expressions
    if (!((startExpr == otherWindow.startExpr) ||
          (startExpr && otherWindow.startExpr && *startExpr == *otherWindow.startExpr))) {
        return false;
    }
    if (!((endExpr == otherWindow.endExpr) ||
          (endExpr && otherWindow.endExpr && *endExpr == *otherWindow.endExpr))) {
        return false;
    }
    if (!((offsetExpr == otherWindow.offsetExpr) ||
          (offsetExpr && otherWindow.offsetExpr && *offsetExpr == *otherWindow.offsetExpr))) {
        return false;
    }
    if (!((defaultExpr == otherWindow.defaultExpr) ||
          (defaultExpr && otherWindow.defaultExpr && *defaultExpr == *otherWindow.defaultExpr))) {
        return false;
    }

    // Compare argument orders
   //TODO
    /*if (!((argOrders == otherWindow.argOrders) ||
          (argOrders && otherWindow.argOrders && *argOrders == *otherWindow.argOrders))) {
        return false;
    }*/

    return true;
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


SubqueryExpression::SubqueryExpression(SubqueryType subQueryType, std::shared_ptr<TableProducer> subquery) : ParsedExpression(ExpressionType::SUBQUERY, TYPE), subQueryType(subQueryType), subquery(subquery) {
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


CaseExpression::CaseExpression(std::optional<std::shared_ptr<ParsedExpression>> caseExpr, std::vector<CaseCheck> caseChecks, std::shared_ptr<ParsedExpression> elseExpr) : ParsedExpression(ExpressionType::CASE_EXPR, TYPE), caseExpr(caseExpr), caseChecks(std::move(caseChecks)), elseExpr(std::move(elseExpr)) {
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
        if (!(**caseExpr == **otherCase.caseExpr)) {
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

SetColumnExpression::SetColumnExpression(std::vector<std::pair<std::shared_ptr<ColumnRefExpression>, std::shared_ptr<ParsedExpression>>> sets) : ParsedExpression(ExpressionType::SET, TYPE) , sets(std::move(sets)){

}
std::string SetColumnExpression::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "Not implemented";
}
// ... existing code ...
size_t SetColumnExpression::hash() {
   size_t result = ParsedExpression::hash();

   // Combine hash for each (column, expression) pair in order
   for (const auto& kv : sets) {
      const auto& col = kv.first;
      const auto& expr = kv.second;

      size_t pair_hash = 0;
      if (col) {
         pair_hash ^= col->hash() + 0x9e3779b9 + (pair_hash << 6) + (pair_hash >> 2);
      } else {
         // Distinguish null column positions
         pair_hash ^= 0x517cc1b7 + (pair_hash << 6) + (pair_hash >> 2);
      }

      if (expr) {
         pair_hash ^= expr->hash() + 0x9e3779b9 + (pair_hash << 6) + (pair_hash >> 2);
      } else {
         // Distinguish null expression positions
         pair_hash ^= 0x85ebca6b + (pair_hash << 6) + (pair_hash >> 2);
      }

      // Mix into the running result
      result ^= pair_hash + 0x9e3779b9 + (result << 6) + (result >> 2);
   }

   return result;
}
bool SetColumnExpression::operator==(ParsedExpression& other) {
   if (!ParsedExpression::operator==(other)) {
      return false;
   }

   auto* otherSet = dynamic_cast<SetColumnExpression*>(&other);
   if (!otherSet) {
      return false;
   }

   if (sets.size() != otherSet->sets.size()) {
      return false;
   }

   for (size_t i = 0; i < sets.size(); ++i) {
      const auto& [colA, exprA] = sets[i];
      const auto& [colB, exprB] = otherSet->sets[i];

      // Compare columns (allow nulls and pointer equality)
      if (!(colA == colB || (colA && colB && *colA == *colB))) {
         return false;
      }
      // Compare expressions (allow nulls and pointer equality)
      if (!(exprA == exprB || (exprA && exprB && *exprA == *exprB))) {
         return false;
      }
   }

   return true;
}


} // namespace lingodb::ast
