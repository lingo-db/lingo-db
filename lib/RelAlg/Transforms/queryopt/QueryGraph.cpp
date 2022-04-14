#include <iostream>

#define NDEBUG
//#include "arrow/compute/exec/expression.h"
#undef NDEBUG

#include "mlir-support/eval.h"
#include "mlir-support/parsing.h"
#include "mlir/Dialect/RelAlg/Transforms/queryopt/QueryGraph.h"

void mlir::relalg::QueryGraph::print(llvm::raw_ostream& out) {
   out << "QueryGraph:{\n";
   out << "Nodes: [\n";
   for (auto& n : nodes) {
      out << "{" << n.id << ",";
      n.op->print(out);
      out << ", selectivity=" << n.selectivity;
      out << ", predicates={";
      for (auto op : n.additionalPredicates) {
         op->print(out);
         out << ",";
      }

      out << "}}";
      out << "},\n";
   }
   out << "]\n";
   out << "Joins: [\n";
   for (auto& e : joins) {
      out << "{ v=";
      printReadable(e.left, out);
      out << ", u=";
      printReadable(e.right, out);
      out << ", op=\n";
      if (e.op) {
         e.op->print(out);
      }
      out << ", selectivity=" << e.selectivity;

      out << "},\n";
   }
   out << "]\n";
   out << "Selections: [\n";
   for (auto& e : selections) {
      out << "{ required=";
      printReadable(e.required, out);
      out << ", op=\n";
      if (e.op) {
         e.op->print(out);
      }
      out << ", selectivity=" << e.selectivity;
      out << "},\n";
   }
   out << "]\n";
   out << "}\n";
}

std::unique_ptr<support::eval::expr> buildEvalExpr(mlir::Value val, std::unordered_map<const mlir::relalg::Column*, std::string>& mapping) {
   auto* op = val.getDefiningOp();
   if (!op) return std::move(support::eval::createInvalid());
   if (auto constantOp = mlir::dyn_cast_or_null<mlir::db::ConstantOp>(op)) {
      std::variant<int64_t, double, std::string> parseArg;
      if (auto integerAttr = constantOp.value().dyn_cast_or_null<mlir::IntegerAttr>()) {
         parseArg = integerAttr.getInt();
      } else if (auto floatAttr = constantOp.value().dyn_cast_or_null<mlir::FloatAttr>()) {
         parseArg = floatAttr.getValueAsDouble();
      } else if (auto stringAttr = constantOp.value().dyn_cast_or_null<mlir::StringAttr>()) {
         parseArg = stringAttr.str();
      } else {
         return support::eval::createInvalid();
      }
      auto type = constantOp.getType();
      arrow::Type::type typeConstant = arrow::Type::type::NA;
      uint32_t param1 = 0, param2 = 0;
      if (isIntegerType(type, 1)) {
         typeConstant = arrow::Type::type::BOOL;
      } else if (auto intWidth = getIntegerWidth(type, false)) {
         switch (intWidth) {
            case 8: typeConstant = arrow::Type::type::INT8; break;
            case 16: typeConstant = arrow::Type::type::INT16; break;
            case 32: typeConstant = arrow::Type::type::INT32; break;
            case 64: typeConstant = arrow::Type::type::INT64; break;
         }
      } else if (auto uIntWidth = getIntegerWidth(type, true)) {
         switch (uIntWidth) {
            case 8: typeConstant = arrow::Type::type::UINT8; break;
            case 16: typeConstant = arrow::Type::type::UINT16; break;
            case 32: typeConstant = arrow::Type::type::UINT32; break;
            case 64: typeConstant = arrow::Type::type::UINT64; break;
         }
      } else if (auto decimalType = type.dyn_cast_or_null<mlir::db::DecimalType>()) {
         typeConstant = arrow::Type::type::DECIMAL128;
         param1 = decimalType.getP();
         param2 = decimalType.getS();
      } else if (auto floatType = type.dyn_cast_or_null<mlir::FloatType>()) {
         switch (floatType.getWidth()) {
            case 16: typeConstant = arrow::Type::type::HALF_FLOAT; break;
            case 32: typeConstant = arrow::Type::type::FLOAT; break;
            case 64: typeConstant = arrow::Type::type::DOUBLE; break;
         }
      } else if (auto stringType = type.dyn_cast_or_null<mlir::db::StringType>()) {
         typeConstant = arrow::Type::type::STRING;
      } else if (auto dateType = type.dyn_cast_or_null<mlir::db::DateType>()) {
         if (dateType.getUnit() == mlir::db::DateUnitAttr::day) {
            typeConstant = arrow::Type::type::DATE32;
         } else {
            typeConstant = arrow::Type::type::DATE64;
         }
      } else if (auto charType = type.dyn_cast_or_null<mlir::db::CharType>()) {
         typeConstant = arrow::Type::type::FIXED_SIZE_BINARY;
         param1 = charType.getBytes();
      } else if (auto intervalType = type.dyn_cast_or_null<mlir::db::IntervalType>()) {
         if (intervalType.getUnit() == mlir::db::IntervalUnitAttr::months) {
            typeConstant = arrow::Type::type::INTERVAL_MONTHS;
         } else {
            typeConstant = arrow::Type::type::INTERVAL_DAY_TIME;
         }
      } else if (auto timestampType = type.dyn_cast_or_null<mlir::db::TimestampType>()) {
         typeConstant = arrow::Type::type::TIMESTAMP;
         param1 = static_cast<uint32_t>(timestampType.getUnit());
      }
      assert(typeConstant != arrow::Type::type::NA);

      auto parseResult = support::parse(parseArg, typeConstant, param1);
      return support::eval::createLiteral(parseResult, std::make_tuple(typeConstant, param1, param2));
   } else if (auto attrRefOp = mlir::dyn_cast_or_null<mlir::relalg::GetColumnOp>(op)) {
      return support::eval::createAttrRef(mapping.at(&attrRefOp.attr().getColumn()));
   } else if (auto cmpOp = mlir::dyn_cast_or_null<mlir::relalg::CmpOpInterface>(op)) {
      auto left = cmpOp.getLeft();
      auto right = cmpOp.getRight();
      if (cmpOp.isEqualityPred()) {
         return support::eval::createEq(buildEvalExpr(left, mapping), buildEvalExpr(right, mapping));
      } else if (cmpOp.isLessPred(false)) {
         return support::eval::createLt(buildEvalExpr(left, mapping), buildEvalExpr(right, mapping));
      } else if (cmpOp.isGreaterPred(false)) {
         return support::eval::createGt(buildEvalExpr(left, mapping), buildEvalExpr(right, mapping));
      } else {
         return support::eval::createInvalid();
      }
   } else if (auto betweenOp = mlir::dyn_cast_or_null<mlir::db::BetweenOp>(op)) {
      std::vector<std::unique_ptr<support::eval::expr>> expressions;
      expressions.push_back(support::eval::createLt(buildEvalExpr(betweenOp.val(), mapping), buildEvalExpr(betweenOp.upper(), mapping)));
      expressions.push_back(support::eval::createGt(buildEvalExpr(betweenOp.val(), mapping), buildEvalExpr(betweenOp.lower(), mapping)));
      return support::eval::createAnd(expressions);
   } else if (auto oneOfOp = mlir::dyn_cast_or_null<mlir::db::OneOfOp>(op)) {
      std::vector<std::unique_ptr<support::eval::expr>> expressions;
      for (auto v : oneOfOp.vals()) {
         expressions.push_back(support::eval::createEq(buildEvalExpr(oneOfOp.val(), mapping), buildEvalExpr(v, mapping)));
      }
      return support::eval::createOr(expressions);
   } else if (auto andOp = mlir::dyn_cast_or_null<mlir::db::AndOp>(op)) {
      std::vector<std::unique_ptr<support::eval::expr>> expressions;
      for (auto v : andOp.vals()) {
         expressions.push_back(buildEvalExpr(v, mapping));
      }
      return support::eval::createAnd(expressions);
   } else if (auto orOp = mlir::dyn_cast_or_null<mlir::db::OrOp>(op)) {
      std::vector<std::unique_ptr<support::eval::expr>> expressions;
      for (auto v : orOp.vals()) {
         expressions.push_back(buildEvalExpr(v, mapping));
      }
      return support::eval::createOr(expressions);
   } else if (auto runtimeCall = mlir::dyn_cast_or_null<mlir::db::RuntimeCall>(op)) {
      if (runtimeCall.fn() == "ConstLike" || runtimeCall.fn() == "Like") {
         if (auto constantOp = mlir::dyn_cast_or_null<mlir::db::ConstantOp>(runtimeCall.args()[1].getDefiningOp())) {
            return support::eval::createLike(buildEvalExpr(runtimeCall.args()[0], mapping), constantOp.value().cast<mlir::StringAttr>().str());
         }
      }
      return support::eval::createInvalid();
   }
   return support::eval::createInvalid();
}

std::optional<double> estimateUsingSample(mlir::relalg::QueryGraph::Node& n) {
   if (!n.op) return {};
   if (n.additionalPredicates.empty()) return {};
   if (auto baseTableOp = mlir::dyn_cast_or_null<mlir::relalg::BaseTableOp>(n.op.getOperation())) {
      std::unordered_map<const mlir::relalg::Column*, std::string> mapping;
      for (auto c : baseTableOp.columns()) {
         mapping[&c.getValue().cast<mlir::relalg::ColumnDefAttr>().getColumn()] = c.getName().str();
      }
      auto meta = baseTableOp.meta().getMeta();
      auto sample = meta->getSample();
      if (!sample) return {};
      std::vector<std::unique_ptr<support::eval::expr>> expressions;
      for (auto pred : n.additionalPredicates) {
         if (auto selOp = mlir::dyn_cast_or_null<mlir::relalg::SelectionOp>(pred.getOperation())) {
            auto v = mlir::cast<mlir::relalg::ReturnOp>(selOp.getPredicateBlock().getTerminator()).results()[0];
            expressions.push_back(buildEvalExpr(v, mapping)); //todo: ignore failing ones?
         }
      }
      auto optionalCount = support::eval::countResults(sample, support::eval::createAnd(expressions));
      if (!optionalCount.has_value()) return {};
      auto count = optionalCount.value();
      if (count == 0) count = 1;
      return static_cast<double>(count) / static_cast<double>(sample->num_rows());
   }

   return {};
}
mlir::relalg::ColumnSet mlir::relalg::QueryGraph::getPKey(mlir::relalg::QueryGraph::Node& n) {
   if (!n.op) return {};
   if (auto baseTableOp = mlir::dyn_cast_or_null<mlir::relalg::BaseTableOp>(n.op.getOperation())) {
      auto meta = baseTableOp.meta().getMeta();
      mlir::relalg::ColumnSet attributes;
      std::unordered_map<std::string, const mlir::relalg::Column*> mapping;
      for (auto c : baseTableOp.columns()) {
         mapping[c.getName().str()] = &c.getValue().cast<mlir::relalg::ColumnDefAttr>().getColumn();
      }
      for (auto c : meta->getPrimaryKey()) {
         attributes.insert(mapping.at(c));
      }
      return attributes;
   }

   return {};
}
double getRows(mlir::relalg::QueryGraph::Node& n) {
   if (auto baseTableOp = mlir::dyn_cast_or_null<mlir::relalg::BaseTableOp>(n.op.getOperation())) {
      auto numRows = baseTableOp.meta().getMeta()->getNumRows();
      baseTableOp->setAttr("rows", mlir::FloatAttr::get(mlir::FloatType::getF64(n.op.getContext()), numRows));
      return numRows == 0 ? 1 : numRows;
   }
   return 1;
}
void mlir::relalg::QueryGraph::estimate() {
   for (auto& node : nodes) {
      node.selectivity = 1;
      if (node.op) {
         node.rows = getRows(node);
         auto availableLeft = node.op.getAvailableColumns();
         mlir::relalg::ColumnSet availableRight;
         std::vector<Predicate> predicates;
         for (auto pred : node.additionalPredicates) {
            addPredicates(predicates, pred, availableLeft, availableRight);
         }
         ColumnSet pkey = getPKey(node);
         ColumnSet predicatesLeft;
         for (auto predicate : predicates) {
            predicatesLeft.insert(predicate.left);
         }
         bool pKeyIncluded = !pkey.empty() && pkey.isSubsetOf(predicatesLeft);
         if (pKeyIncluded) {
            node.selectivity = 1 / node.rows;
         } else {
            auto estimation = estimateUsingSample(node);
            if (estimation.has_value()) {
               node.selectivity = estimation.value();
            } else {
               for (auto predicate : predicates) {
                  if (predicate.isEq) {
                     node.selectivity *= 0.1;
                  } else {
                     node.selectivity *= 0.25;
                  }
               }
            }
         }
      }
   }
   for (auto& edge : joins) {
      edge.selectivity = estimateSelectivity(edge.op, edge.left, edge.right);
   }
   for (auto& edge : selections) {
      if (edge.required.count() == 2) {
         auto left = NodeSet::single(this->numNodes, edge.required.findFirst());
         edge.selectivity = estimateSelectivity(edge.op, left, edge.required);
      }
   }
}
double mlir::relalg::QueryGraph::calculateSelectivity(SelectionEdge& edge, NodeSet left, NodeSet right) {
   if (edge.required.count() == 2 && left.any() && right.any()) return edge.selectivity;
   auto key = left & edge.required;
   if (edge.cachedSel.contains(key)) {
      return edge.cachedSel[key];
   }
   double selectivity = estimateSelectivity(edge.op, left, right);
   edge.cachedSel[key] = selectivity;
   return selectivity;
}
double mlir::relalg::QueryGraph::estimateSelectivity(Operator op, NodeSet left, NodeSet right) {
   auto availableLeft = getAttributesForNodeSet(left);
   auto availableRight = getAttributesForNodeSet(right);
   std::vector<Predicate> predicates;
   addPredicates(predicates, op, availableLeft, availableRight);
   double selectivity = 1.0;
   std::vector<std::pair<double, ColumnSet>> pkeysLeft;
   std::vector<std::pair<double, ColumnSet>> pkeysRight;
   iterateNodes(left, [&](auto node) {
      if (node.op) {
         if (auto baseTableOp = mlir::dyn_cast_or_null<mlir::relalg::BaseTableOp>(node.op.getOperation())) {
            pkeysLeft.push_back({node.rows, getPKey(node)});
         }
      }
   });
   iterateNodes(right, [&](auto node) {
      if (node.op) {
         if (auto baseTableOp = mlir::dyn_cast_or_null<mlir::relalg::BaseTableOp>(node.op.getOperation())) {
            pkeysRight.push_back({node.rows, getPKey(node)});
         }
      }
   });

   ColumnSet predicatesLeft;
   ColumnSet predicatesRight;
   for (auto predicate : predicates) {
      predicatesLeft.insert(predicate.left);
      predicatesRight.insert(predicate.right);
   }
   for (auto p : pkeysLeft) {
      auto [rows, pkey] = p;
      if (pkey.isSubsetOf(predicatesLeft)) {
         selectivity *= 1 / rows;
         predicatesLeft.remove(pkey);
      }
   }
   for (auto p : pkeysRight) {
      auto [rows, pkey] = p;
      if (pkey.isSubsetOf(predicatesRight)) {
         selectivity *= 1 / rows;
         predicatesRight.remove(pkey);
      }
   }
   for (auto predicate : predicates) {
      if (predicate.left.isSubsetOf(predicatesLeft) && predicate.right.isSubsetOf(predicatesRight)) {
         if (predicate.isEq) {
            selectivity *= 0.1;
         } else {
            selectivity *= 0.25;
         }
      }
   }
   return selectivity;
}