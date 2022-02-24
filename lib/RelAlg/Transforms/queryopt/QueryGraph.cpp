#include <iostream>

#define NDEBUG
//#include "arrow/compute/exec/expression.h"
#undef NDEBUG

#include "mlir-support/eval.h"
#include "mlir-support/parsing.h"
#include "mlir/Conversion/DBToArrowStd/ArrowTypes.h"
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

std::unique_ptr<support::eval::expr> buildEvalExpr(mlir::Value val, std::unordered_map<const mlir::relalg::RelationalAttribute*, std::string>& mapping) {
   auto *op = val.getDefiningOp();
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

      auto [arrowType, param1, param2] = mlir::db::codegen::convertTypeToArrow(constantOp.getType());
      auto parseResult = support::parse(parseArg, arrowType, param1);
      return support::eval::createLiteral(parseResult, std::make_tuple(arrowType, param1, param2));
   } else if (auto attrRefOp = mlir::dyn_cast_or_null<mlir::relalg::GetAttrOp>(op)) {
      return support::eval::createAttrRef(mapping.at(&attrRefOp.attr().getRelationalAttribute()));
   } else if (auto cmpOp = mlir::dyn_cast_or_null<mlir::db::CmpOp>(op)) {
      if (cmpOp.predicate() == mlir::db::DBCmpPredicate::eq) {
         return support::eval::createEq(buildEvalExpr(cmpOp.left(), mapping), buildEvalExpr(cmpOp.right(), mapping));
      } else if (cmpOp.predicate() == mlir::db::DBCmpPredicate::lt) {
         return support::eval::createLt(buildEvalExpr(cmpOp.left(), mapping), buildEvalExpr(cmpOp.right(), mapping));
      } else if (cmpOp.predicate() == mlir::db::DBCmpPredicate::gt) {
         return support::eval::createGt(buildEvalExpr(cmpOp.left(), mapping), buildEvalExpr(cmpOp.right(), mapping));
      } else {
         return support::eval::createInvalid();
      }
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
   }
   return support::eval::createInvalid();
}

std::optional<double> estimateUsingSample(mlir::relalg::QueryGraph::Node& n) {
   if (!n.op) return {};
   if (n.additionalPredicates.empty()) return {};
   if (auto baseTableOp = mlir::dyn_cast_or_null<mlir::relalg::BaseTableOp>(n.op.getOperation())) {
      std::unordered_map<const mlir::relalg::RelationalAttribute*, std::string> mapping;
      for (auto c : baseTableOp.columns()) {
         mapping[&c.getValue().cast<mlir::relalg::RelationalAttributeDefAttr>().getRelationalAttribute()] = c.getName().str();
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
mlir::relalg::Attributes mlir::relalg::QueryGraph::getPKey(mlir::relalg::QueryGraph::Node& n) {
   if (!n.op) return {};
   if (auto baseTableOp = mlir::dyn_cast_or_null<mlir::relalg::BaseTableOp>(n.op.getOperation())) {
      auto meta = baseTableOp.meta().getMeta();
      mlir::relalg::Attributes attributes;
      std::unordered_map<std::string, const mlir::relalg::RelationalAttribute*> mapping;
      for (auto c : baseTableOp.columns()) {
         mapping[c.getName().str()] = &c.getValue().cast<mlir::relalg::RelationalAttributeDefAttr>().getRelationalAttribute();
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
      return numRows == 0 ? 1 : numRows;
   }
   return 1;
}
void mlir::relalg::QueryGraph::estimate() {
   for (auto& node : nodes) {
      node.selectivity = 1;
      if (node.op) {
         node.rows = getRows(node);
         auto availableLeft = node.op.getAvailableAttributes();
         mlir::relalg::Attributes availableRight;
         std::vector<Predicate> predicates;
         for (auto pred : node.additionalPredicates) {
            addPredicates(predicates, pred, availableLeft, availableRight);
         }
         Attributes pkey = getPKey(node);
         Attributes predicatesLeft;
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
      if(edge.required.count()==2) {
         auto left = NodeSet::single(this->numNodes, edge.required.findFirst());
         edge.selectivity = estimateSelectivity(edge.op, left, edge.required);
      }
   }
}
double mlir::relalg::QueryGraph::calculateSelectivity(SelectionEdge& edge, NodeSet left, NodeSet right) {
   if(edge.required.count()==2&&left.any()&&right.any())return edge.selectivity;
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
   std::vector<std::pair<double, Attributes>> pkeysLeft;
   std::vector<std::pair<double, Attributes>> pkeysRight;
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

   Attributes predicatesLeft;
   Attributes predicatesRight;
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