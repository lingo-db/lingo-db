#include "lingodb/compiler/Dialect/RelAlg/Transforms/queryopt/QueryGraph.h"

#include "lingodb/compiler/mlir-support/eval.h"
#include "lingodb/compiler/mlir-support/parsing.h"

#include <iostream>

#include "json.h"
#include <arrow/record_batch.h>
namespace {
std::unique_ptr<lingodb::compiler::support::eval::expr> buildConstant(mlir::Type type, std::variant<int64_t, double, std::string> parseArg) {
   namespace db = lingodb::compiler::dialect::db;
   ::arrow::Type::type typeConstant = ::arrow::Type::type::NA;
   uint32_t param1 = 0, param2 = 0;
   if (isIntegerType(type, 1)) {
      typeConstant = ::arrow::Type::type::BOOL;
   } else if (auto intWidth = getIntegerWidth(type, false)) {
      switch (intWidth) {
         case 8: typeConstant = ::arrow::Type::type::INT8; break;
         case 16: typeConstant = ::arrow::Type::type::INT16; break;
         case 32: typeConstant = ::arrow::Type::type::INT32; break;
         case 64: typeConstant = ::arrow::Type::type::INT64; break;
      }
   } else if (auto uIntWidth = getIntegerWidth(type, true)) {
      switch (uIntWidth) {
         case 8: typeConstant = ::arrow::Type::type::UINT8; break;
         case 16: typeConstant = ::arrow::Type::type::UINT16; break;
         case 32: typeConstant = ::arrow::Type::type::UINT32; break;
         case 64: typeConstant = ::arrow::Type::type::UINT64; break;
      }
   } else if (auto decimalType = mlir::dyn_cast_or_null<db::DecimalType>(type)) {
      typeConstant = ::arrow::Type::type::DECIMAL128;
      param1 = decimalType.getP();
      param2 = decimalType.getS();
   } else if (auto floatType = mlir::dyn_cast_or_null<mlir::FloatType>(type)) {
      switch (floatType.getWidth()) {
         case 16: typeConstant = ::arrow::Type::type::HALF_FLOAT; break;
         case 32: typeConstant = ::arrow::Type::type::FLOAT; break;
         case 64: typeConstant = ::arrow::Type::type::DOUBLE; break;
      }
   } else if (auto stringType = mlir::dyn_cast_or_null<db::StringType>(type)) {
      typeConstant = ::arrow::Type::type::STRING;
   } else if (auto dateType = mlir::dyn_cast_or_null<db::DateType>(type)) {
      if (dateType.getUnit() == db::DateUnitAttr::day) {
         typeConstant = ::arrow::Type::type::DATE32;
      } else {
         typeConstant = ::arrow::Type::type::DATE64;
      }
   } else if (auto charType = mlir::dyn_cast_or_null<db::CharType>(type)) {
      // we need to multiply by 4 to get the maximum number of required bytes (4 bytes per character utf-8)
      if (charType.getLen() <= 1) {
         typeConstant = ::arrow::Type::type::FIXED_SIZE_BINARY;
         param1 = charType.getLen() * 4;
      } else {
         typeConstant = ::arrow::Type::type::STRING;
      }
   } else if (auto intervalType = mlir::dyn_cast_or_null<db::IntervalType>(type)) {
      if (intervalType.getUnit() == db::IntervalUnitAttr::months) {
         typeConstant = ::arrow::Type::type::INTERVAL_MONTHS;
      } else {
         typeConstant = ::arrow::Type::type::INTERVAL_DAY_TIME;
      }
   } else if (auto timestampType = mlir::dyn_cast_or_null<db::TimestampType>(type)) {
      typeConstant = ::arrow::Type::type::TIMESTAMP;
      param1 = static_cast<uint32_t>(timestampType.getUnit());
   }
   assert(typeConstant != ::arrow::Type::type::NA);

   auto parseResult = lingodb::compiler::support::parse(parseArg, typeConstant, param1);
   return lingodb::compiler::support::eval::createLiteral(parseResult, std::make_tuple(typeConstant, param1, param2));
}
} //namespace
namespace lingodb::compiler::dialect::relalg {
void QueryGraph::print(llvm::raw_ostream& out) {
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

std::unique_ptr<lingodb::compiler::support::eval::expr> buildEvalExpr(mlir::Value val, llvm::DenseMap<const tuples::Column*, std::string>& mapping) {
   auto* op = val.getDefiningOp();
   if (!op) return support::eval::createInvalid();
   if (auto constantOp = mlir::dyn_cast_or_null<db::ConstantOp>(op)) {
      std::variant<int64_t, double, std::string> parseArg;
      if (auto integerAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(constantOp.getValue())) {
         parseArg = integerAttr.getInt();
      } else if (auto floatAttr = mlir::dyn_cast_or_null<mlir::FloatAttr>(constantOp.getValue())) {
         parseArg = floatAttr.getValueAsDouble();
      } else if (auto stringAttr = mlir::dyn_cast_or_null<mlir::StringAttr>(constantOp.getValue())) {
         parseArg = stringAttr.str();
      } else {
         return support::eval::createInvalid();
      }
      auto type = constantOp.getType();
      return buildConstant(type, parseArg);
   } else if (auto attrRefOp = mlir::dyn_cast_or_null<tuples::GetColumnOp>(op)) {
      return support::eval::createAttrRef(mapping.at(&attrRefOp.getAttr().getColumn()));
   } else if (auto cmpOp = mlir::dyn_cast_or_null<CmpOpInterface>(op)) {
      auto left = cmpOp.getLeft();
      auto right = cmpOp.getRight();
      if (cmpOp.isEqualityPred(false)) {
         return support::eval::createEq(buildEvalExpr(left, mapping), buildEvalExpr(right, mapping));
      } else if (cmpOp.isLessPred(false)) {
         return support::eval::createLt(buildEvalExpr(left, mapping), buildEvalExpr(right, mapping));
      } else if (cmpOp.isGreaterPred(false)) {
         return support::eval::createGt(buildEvalExpr(left, mapping), buildEvalExpr(right, mapping));
      } else if (cmpOp.isLessPred(true)) {
         return support::eval::createLte(buildEvalExpr(left, mapping), buildEvalExpr(right, mapping));
      } else if (cmpOp.isGreaterPred(true)) {
         return support::eval::createGte(buildEvalExpr(left, mapping), buildEvalExpr(right, mapping));
      } else if (cmpOp.isUnequalityPred()) {
         return support::eval::createNot(support::eval::createEq(buildEvalExpr(left, mapping), buildEvalExpr(right, mapping)));
      }
   } else if (auto betweenOp = mlir::dyn_cast_or_null<db::BetweenOp>(op)) {
      std::vector<std::unique_ptr<support::eval::expr>> expressions;
      expressions.push_back(support::eval::createLt(buildEvalExpr(betweenOp.getVal(), mapping), buildEvalExpr(betweenOp.getUpper(), mapping)));
      expressions.push_back(support::eval::createGt(buildEvalExpr(betweenOp.getVal(), mapping), buildEvalExpr(betweenOp.getLower(), mapping)));
      return support::eval::createAnd(expressions);
   } else if (auto oneOfOp = mlir::dyn_cast_or_null<db::OneOfOp>(op)) {
      std::vector<std::unique_ptr<support::eval::expr>> expressions;
      for (auto v : oneOfOp.getVals()) {
         expressions.push_back(support::eval::createEq(buildEvalExpr(oneOfOp.getVal(), mapping), buildEvalExpr(v, mapping)));
      }
      return support::eval::createOr(expressions);
   } else if (auto notOp = mlir::dyn_cast_or_null<db::NotOp>(op)) {
      return support::eval::createNot(buildEvalExpr(notOp.getVal(), mapping));
   } else if (auto isNullOp = mlir::dyn_cast_or_null<db::IsNullOp>(op)) {
      return support::eval::createIsNull(buildEvalExpr(isNullOp.getVal(), mapping));
   } else if (auto andOp = mlir::dyn_cast_or_null<db::AndOp>(op)) {
      std::vector<std::unique_ptr<support::eval::expr>> expressions;
      for (auto v : andOp.getVals()) {
         expressions.push_back(buildEvalExpr(v, mapping));
      }
      return support::eval::createAnd(expressions);
   } else if (auto orOp = mlir::dyn_cast_or_null<db::OrOp>(op)) {
      std::vector<std::unique_ptr<support::eval::expr>> expressions;
      for (auto v : orOp.getVals()) {
         expressions.push_back(buildEvalExpr(v, mapping));
      }
      return support::eval::createOr(expressions);
   } else if (auto runtimeCall = mlir::dyn_cast_or_null<db::RuntimeCall>(op)) {
      if (runtimeCall.getFn() == "ConstLike" || runtimeCall.getFn() == "Like") {
         if (auto constantOp = mlir::dyn_cast_or_null<db::ConstantOp>(runtimeCall.getArgs()[1].getDefiningOp())) {
            return support::eval::createLike(buildEvalExpr(runtimeCall.getArgs()[0], mapping), mlir::cast<mlir::StringAttr>(constantOp.getValue()).str());
         }
      }
   } else if (auto castOp = mlir::dyn_cast_or_null<db::CastOp>(op)) {
      const auto fromType = getBaseType(castOp.getVal().getType());
      const auto toType = getBaseType(castOp.getRes().getType());
      auto fromExpr = buildEvalExpr(castOp.getVal(), mapping);
      if (fromType == toType) {
         return fromExpr;
      }
      if (const auto charType = mlir::dyn_cast_or_null<db::CharType>(fromType); charType && mlir::isa<db::StringType>(toType)) {
         assert(charType.getLen() > 1 && "Expected char<1> that are to be compared to str to be converted to string type instead of db::cast-ed");
         // char types with len > 1 are stored as strings in the arrow format => we don't need to cast them
         return fromExpr;
      }
   }
   // val.dump();
   return support::eval::createInvalid();
}
namespace {
void appendRestrictions(relalg::BaseTableOp baseTableOp, std::vector<std::unique_ptr<lingodb::compiler::support::eval::expr>>& expressions) {
   std::unordered_map<std::string, mlir::Type> typeMapping;
   for (auto c : baseTableOp.getColumns()) {
      typeMapping[c.getName().str()] = mlir::cast<tuples::ColumnDefAttr>(c.getValue()).getColumn().type;
   }
   if (baseTableOp->hasAttr("restriction")) {
      auto restrictionsStr = cast<mlir::StringAttr>(baseTableOp->getAttr("restriction")).getValue();
      auto restrictions = nlohmann::json::parse(restrictionsStr.str()).get<nlohmann::json::array_t>();
      for (auto& r : restrictions) {
         auto cmp = r["cmp"].get<std::string>();
         auto columnName = r["column"].get<std::string>();
         if (cmp == "isnotnull") {
            auto colExpr = support::eval::createAttrRef(columnName);
            expressions.push_back(support::eval::createNot(support::eval::createIsNull(std::move(colExpr))));
            continue;
         }
         auto type = getBaseType(typeMapping.at(columnName));
         if (cmp == "in") {
            std::vector<std::unique_ptr<support::eval::expr>> orExpressions;
            for (auto value : r["values"]) {
               std::variant<int64_t, double, std::string> parseArg;
               if (value.is_string()) {
                  parseArg = value.get<std::string>();
               } else if (value.is_number_integer()) {
                  parseArg = value.get<int64_t>();
               } else if (value.is_number_float()) {
                  parseArg = value.get<double>();
               } else {
                  assert(false);
                  continue;
               }
               orExpressions.push_back(support::eval::createEq(support::eval::createAttrRef(columnName), buildConstant(type, parseArg)));
            }
            expressions.push_back(support::eval::createOr(orExpressions));
            continue;
         }
         auto value = r["value"];

         std::variant<int64_t, double, std::string> parseArg;
         if (value.is_string()) {
            parseArg = value.get<std::string>();
         } else if (value.is_number_integer()) {
            parseArg = value.get<int64_t>();
         } else if (value.is_number_float()) {
            parseArg = value.get<double>();
         } else {
            assert(false);
            continue;
         }
         auto colExpr = support::eval::createAttrRef(columnName);
         auto valueExpr = buildConstant(type, parseArg);
         if (cmp == "=") {
            expressions.push_back(support::eval::createEq(std::move(colExpr), std::move(valueExpr)));
         } else if (cmp == "!=") {
            expressions.push_back(support::eval::createNot(support::eval::createEq(std::move(colExpr), std::move(valueExpr))));
         } else if (cmp == "<") {
            expressions.push_back(support::eval::createLt(std::move(colExpr), std::move(valueExpr)));
         } else if (cmp == "<=") {
            expressions.push_back(support::eval::createLte(std::move(colExpr), std::move(valueExpr)));
         } else if (cmp == ">") {
            expressions.push_back(support::eval::createGt(std::move(colExpr), std::move(valueExpr)));
         } else if (cmp == ">=") {
            expressions.push_back(support::eval::createGte(std::move(colExpr), std::move(valueExpr)));
         } else {
            assert(false);
            continue;
         }
      }
   }
}
std::optional<double> estimateUsingSample(QueryGraph::Node& n) {
   if (!n.op) return {};
   if (n.additionalPredicates.empty() && !n.op->hasAttr("restriction")) return {};
   if (auto baseTableOp = mlir::dyn_cast_or_null<BaseTableOp>(n.op.getOperation())) {
      llvm::DenseMap<const tuples::Column*, std::string> mapping;
      std::unordered_map<std::string, mlir::Type> typeMapping;
      for (auto c : baseTableOp.getColumns()) {
         mapping[&mlir::cast<tuples::ColumnDefAttr>(c.getValue()).getColumn()] = c.getName().str();
      }
      auto meta = mlir::dyn_cast_or_null<TableMetaDataAttr>(baseTableOp->getAttr("meta"));
      if (!meta) return {};
      auto sample = meta.getMeta()->getSample();
      if (!sample) return {};
      std::vector<std::unique_ptr<lingodb::compiler::support::eval::expr>> expressions;
      appendRestrictions(baseTableOp, expressions);
      for (auto pred : n.additionalPredicates) {
         if (auto selOp = mlir::dyn_cast_or_null<SelectionOp>(pred.getOperation())) {
            auto v = mlir::cast<tuples::ReturnOp>(selOp.getPredicateBlock().getTerminator()).getResults()[0];
            expressions.push_back(buildEvalExpr(v, mapping)); //todo: ignore failing ones?
         }
      }
      auto optionalCount = lingodb::compiler::support::eval::countResults(sample.getSampleData(), lingodb::compiler::support::eval::createAnd(expressions));
      if (!optionalCount.has_value()) return {};
      auto count = optionalCount.value();
      if (count == 0) count = 1;
      return static_cast<double>(count) / static_cast<double>(sample.getSampleData()->num_rows());
   }

   return {};
}
double getRows(QueryGraph::Node& n) {
   if (auto baseTableOp = mlir::dyn_cast_or_null<BaseTableOp>(n.op.getOperation())) {
      auto meta = mlir::dyn_cast_or_null<TableMetaDataAttr>(baseTableOp->getAttr("meta"));
      if (meta) {
         auto numRows = meta.getMeta()->getNumRows();
         //baseTableOp->setAttr("rows", mlir::FloatAttr::get(mlir::Float64Type::get(n.op.getContext()), numRows));
         return numRows == 0 ? 1 : numRows;
      }
   }
   if (n.op) {
      if (n.op->hasAttr("rows")) {
         if (auto floatAttr = mlir::dyn_cast_or_null<mlir::FloatAttr>(n.op->getAttr("rows"))) {
            return floatAttr.getValueAsDouble();
         }
      }
   }
   return 1;
}
} // namespace
void annotateBaseTable(relalg::BaseTableOp baseTableOp) {
   auto meta = mlir::dyn_cast_or_null<TableMetaDataAttr>(baseTableOp->getAttr("meta"));
   if (meta) {
      auto totalRows = meta.getMeta()->getNumRows();
      baseTableOp->setAttr("total_rows", mlir::FloatAttr::get(mlir::Float64Type::get(baseTableOp.getContext()), totalRows));
      auto sample = meta.getMeta()->getSample();
      std::vector<std::unique_ptr<lingodb::compiler::support::eval::expr>> expressions;
      appendRestrictions(baseTableOp, expressions);
      double filteredRows = totalRows;
      if (sample && !expressions.empty()) {
         auto optionalCount = lingodb::compiler::support::eval::countResults(sample.getSampleData(), lingodb::compiler::support::eval::createAnd(expressions));
         if (optionalCount.has_value()) {
            auto count = optionalCount.value();
            if (count == 0) count = 1;
            double selectivity = static_cast<double>(count) / static_cast<double>(sample.getSampleData()->num_rows());
            filteredRows = static_cast<double>(totalRows) * selectivity;
         }
      }
      baseTableOp->setAttr("rows", mlir::FloatAttr::get(mlir::Float64Type::get(baseTableOp.getContext()), filteredRows));
   }
}
ColumnSet QueryGraph::getPKey(QueryGraph::Node& n) {
   if (!n.op) return {};
   if (auto baseTableOp = mlir::dyn_cast_or_null<BaseTableOp>(n.op.getOperation())) {
      auto meta = mlir::dyn_cast_or_null<TableMetaDataAttr>(baseTableOp->getAttr("meta"));
      if (meta) {
         ColumnSet attributes;
         std::unordered_map<std::string, const tuples::Column*> mapping;
         for (auto c : baseTableOp.getColumns()) {
            mapping[c.getName().str()] = &mlir::cast<tuples::ColumnDefAttr>(c.getValue()).getColumn();
         }
         for (auto c : meta.getMeta()->getPrimaryKey()) {
            if (mapping.contains(c)) {
               attributes.insert(mapping.at(c));
            } else {
               attributes.insert(nullptr);
            }
         }
         return attributes;
      }
   }

   return {};
}

void QueryGraph::estimate() {
   for (auto& node : nodes) {
      node.selectivity = 1;
      if (node.op) {
         node.rows = getRows(node);
         auto availableLeft = node.op.getAvailableColumns();
         ColumnSet availableRight;
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
double QueryGraph::calculateSelectivity(SelectionEdge& edge, NodeSet left, NodeSet right) {
   if (edge.required.count() == 2 && left.any() && right.any()) return edge.selectivity;
   auto key = left & edge.required;
   if (edge.cachedSel.contains(key)) {
      return edge.cachedSel[key];
   }
   double selectivity = estimateSelectivity(edge.op, left, right);
   edge.cachedSel[key] = selectivity;
   return selectivity;
}
double QueryGraph::estimateSelectivity(Operator op, NodeSet left, NodeSet right) {
   auto availableLeft = getAttributesForNodeSet(left);
   auto availableRight = getAttributesForNodeSet(right);
   std::vector<Predicate> predicates;
   addPredicates(predicates, op, availableLeft, availableRight);
   double selectivity = 1.0;
   std::vector<std::pair<double, ColumnSet>> pkeysLeft;
   std::vector<std::pair<double, ColumnSet>> pkeysRight;
   iterateNodes(left, [&](auto node) {
      if (node.op) {
         if (auto baseTableOp = mlir::dyn_cast_or_null<BaseTableOp>(node.op.getOperation())) {
            pkeysLeft.push_back({node.rows, getPKey(node)});
         }
      }
   });
   iterateNodes(right, [&](auto node) {
      if (node.op) {
         if (auto baseTableOp = mlir::dyn_cast_or_null<BaseTableOp>(node.op.getOperation())) {
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
} // namespace lingodb::compiler::dialect::relalg
