#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/mlir-support/eval.h"
#include "lingodb/execution/Execution.h"
#include "lingodb/execution/ResultProcessing.h"
#include "lingodb/scheduler/Scheduler.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include "json.h"

namespace cl = llvm::cl;
namespace {
cl::opt<std::string> inputFilename(cl::Positional,
                                   cl::desc("<input mlir file>"),
                                   cl::init("-"),
                                   cl::value_desc("filename"));
cl::opt<std::string> databasePath(cl::Positional,
                                  cl::desc("<path to  database>"),
                                  cl::init("-"),
                                  cl::value_desc("database"));

using namespace lingodb::compiler::dialect;

int loadMLIR(mlir::MLIRContext& context, mlir::OwningOpRef<mlir::ModuleOp>& module) {
   llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
   if (std::error_code ec = fileOrErr.getError()) {
      llvm::errs() << "Could not open input file: " << ec.message() << "\n";
      return -1;
   }

   // Parse the input mlir.
   llvm::SourceMgr sourceMgr;
   sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
   module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
   if (!module) {
      llvm::errs() << "Error can't load file " << inputFilename << "\n";
      return 3;
   }
   return 0;
}

class ToJson {
   /// Attributes
   mlir::ModuleOp moduleOp;
   std::unordered_map<uint32_t, int64_t> tupleCounts;

   // Transform iu from symbol ref to string form
   std::string symbolRefToIuString(mlir::SymbolRefAttr symbolRefAttr) {
      return symbolRefAttr.getRootReference().str() + "::" + symbolRefAttr.getLeafReference().str();
   }

   // Convert mlir Type to umbra json type
   nlohmann::json convertDataType(mlir::Type type) {
      return llvm::TypeSwitch<mlir::Type, nlohmann::json>(type)
         .Case<db::DecimalType>([](db::DecimalType decimalType) {
            return nlohmann::json{
               {"type", "dataType"},
               {"name", "decimal"},
               {"precision", decimalType.getP()},
               {"scale", decimalType.getS()},
               {"nullable", false}};
         })
         .Case<mlir::IndexType>([](mlir::IndexType indexType) {
            return nlohmann::json{
               {"type", "dataType"},
               {"name", "index"},
               {"signed", false},
               {"nullable", false}};
         })
         .Case<mlir::IntegerType>([](mlir::IntegerType integerType) {
            return nlohmann::json{
               {"baseType", "int"},
               {"numBits", integerType.getWidth()},
               {"signed", integerType.isSigned()},
               {"nullable", false}};
         })
         .Case<db::DateType>([](db::DateType dateType) {
            return nlohmann::json{
               // Umbra treats date as int, we have (annotated) strings
               {"baseType", "text"}};
         })
         .Case<db::NullableType>([&](db::NullableType nullableType) {
            auto res = convertDataType(nullableType.getType());
            res["nullable"] = true;
            return res;
         })
         .Case<db::StringType>([&](db::StringType stringType) {
            return nlohmann::json{{"type", "text"}};
         })
         .Case<db::CharType>([](db::CharType charType) {
            return nlohmann::json{{"type", "text"}};
         })
         .Case<mlir::FloatType>([](mlir::FloatType floatType) {
            return nlohmann::json{{"type", "float"}};
         })
         .Case<db::IntervalType>([](db::IntervalType intervalType) {
            return nlohmann::json{{"type", "interval"},{"unit", stringifyIntervalUnitAttr(intervalType.getUnit())}};
         })
         .Default([](mlir::Type type) {
            llvm::errs() << "type could not be converted ";
            type.dump();
            llvm::errs() << "\n";
            return nlohmann::json();
         });
   }
   std::string convertCmpPredicate(db::DBCmpPredicate cmpPredicate) {
      switch (cmpPredicate) {
         case db::DBCmpPredicate::eq:
            return "=";
         case db::DBCmpPredicate::gt:
            return ">";
         case db::DBCmpPredicate::gte:
            return ">=";
         case db::DBCmpPredicate::lt:
            return "<";
         case db::DBCmpPredicate::lte:
            return "<=";
         case db::DBCmpPredicate::neq:
            return "<>";
         case db::DBCmpPredicate::isa:
            return "isa";
      }
      return "?";
   }
   nlohmann::json innerExpression(const std::vector<std::string> strings, std::vector<nlohmann::json> operands) {
      nlohmann::json result{
         {"type", "expression_inner"},
         {"strings", strings},
         {"subExpressions", operands}};
      return result;
   }

   nlohmann::json innerExpression(const std::vector<std::string> strings, mlir::ValueRange operands) {
      nlohmann::json result{
         {"type", "expression_inner"},
         {"strings", strings},
         {"subExpressions", nlohmann::json::array()}};
      for (auto operand : operands) {
         result["subExpressions"].push_back(convertExpression(operand.getDefiningOp()));
      }
      return result;
   }
   nlohmann::json convertConstant(mlir::Attribute attr, mlir::Type type) {
      nlohmann::json result{
         {"type", "expression_leaf"},
         {"leaf_type", "constant"}};
      result["data_type"] = convertDataType(type);
      if (auto integerAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(attr)) {
         if (type.isInteger(1)) {
            result["value"] = (bool) integerAttr.getInt();
         } else {
            result["value"] = integerAttr.getInt();
         }

      } else if (auto floatAttr = mlir::dyn_cast_or_null<mlir::FloatAttr>(attr)) {
         result["value"] = floatAttr.getValueAsDouble();
      } else if (auto stringAttr = mlir::dyn_cast_or_null<mlir::StringAttr>(attr)) {
         result["value"] = stringAttr.str();
      } else if (auto boolAttr = mlir::dyn_cast_or_null<mlir::BoolAttr>(attr)) {
         result["value"] = boolAttr.getValue();
      } else {
         llvm::errs() << "Constant could not be converted ";
         attr.dump();
         return nlohmann::json{{"type", "expression_leaf"}, {"leaf_type", "unknown"}};
      }
      return result;
   }
   nlohmann::json convertExpression(mlir::Operation* operation) {
      return llvm::TypeSwitch<mlir::Operation*, nlohmann::json>(operation)
         .Case<tuples::ReturnOp>([&](tuples::ReturnOp returnOp) {
            if (returnOp->getOperands().size()) return convertExpression(returnOp.getOperand(0).getDefiningOp());
            return nlohmann::json{};
         })
         .Case<mlir::arith::ExtUIOp>([&](mlir::arith::ExtUIOp extUiOp) { return innerExpression({"cast(", ")"}, extUiOp.getIn()); })
         .Case<mlir::arith::ExtSIOp>([&](mlir::arith::ExtSIOp extUiOp) { return innerExpression({"cast(", ")"}, extUiOp.getIn()); })
         .Case<mlir::arith::SelectOp>([&](mlir::arith::SelectOp selectOp) {
            return innerExpression({"", " ? ", " : ", ""}, selectOp.getOperands());
         })
         .Case<db::NotOp>([&](db::NotOp notOp) {
            return innerExpression({"not ", ""}, notOp.getVal());
         })
         .Case<db::NullOp>([&](db::NullOp isNullOp) {
            return nlohmann::json{
               {"type", "expression_leaf"},
               {"leaf_type", "null"}};
         })
         .Case<db::IsNullOp>([&](db::IsNullOp isNullOp) {
            return innerExpression({"", " is null"}, isNullOp.getOperand());
         })
         .Case<db::AsNullableOp>([&](db::AsNullableOp asNullableOp) {
            return convertExpression(asNullableOp->getOperand(0).getDefiningOp());
         })
         .Case<db::NullableGetVal>([&](db::NullableGetVal nullableGetVal) {
            return convertExpression(nullableGetVal->getOperand(0).getDefiningOp());
         })
         .Case<db::BetweenOp>([&](db::BetweenOp betweenOp) {
            return innerExpression({"", " between ", " and ", ""}, betweenOp.getOperands());
         })
         .Case<db::OneOfOp>([&](db::OneOfOp oneOfOp) {
            std::vector<std::string> seperators = {"", " in ["};

            for (size_t i = 1; i < oneOfOp.getVals().size(); i++) {
               seperators.push_back(", ");
            }
            seperators.push_back("]");
            return innerExpression(seperators, oneOfOp.getOperands());
         })
         .Case<db::AndOp>([&](db::AndOp andOp) {
            std::vector<std::string> seperators = {""};
            for (size_t i = 1; i < andOp.getOperands().size(); i++) {
               seperators.push_back(" and ");
            }
            seperators.push_back("");
            return innerExpression(seperators, andOp.getOperands());
         })
         .Case<db::OrOp>([&](db::OrOp orOp) {
            std::vector<std::string> seperators = {"("};
            for (size_t i = 1; i < orOp.getOperands().size(); i++) {
               seperators.push_back(" or ");
            }
            seperators.push_back(")");
            return innerExpression(seperators, orOp.getOperands());
         })
         .Case<db::CmpOp>([&](db::CmpOp cmpOp) {
            return innerExpression({"", convertCmpPredicate(cmpOp.getPredicate()), ""}, cmpOp.getOperands());
         })
         .Case<db::ConstantOp>([&](db::ConstantOp constantOp) {
            return convertConstant(constantOp.getValue(), constantOp.getType());
         })
         .Case<tuples::GetColumnOp>([&](tuples::GetColumnOp getColumnOp) {
            return columnToJSON(getColumnOp.getAttr());
         })
         .Case<db::AddOp>([&](db::AddOp addOp) {
            return innerExpression({"", " + ", ""}, addOp.getOperands());
         })
         .Case<mlir::arith::AndIOp>([&](mlir::arith::AndIOp andIOp) {
            return innerExpression({"", " & ", ""}, andIOp.getOperands());
         })
         .Case<mlir::arith::OrIOp>([&](mlir::arith::OrIOp orIOp) {
            return innerExpression({"", " | ", ""}, orIOp.getOperands());
         })
         .Case<mlir::arith::ShRUIOp>([&](mlir::arith::ShRUIOp shRUIOp) {
            return innerExpression({"", " >> ", ""}, shRUIOp.getOperands());
         })
         .Case<db::SubOp>([&](db::SubOp subOp) {
            return innerExpression({"", " + ", ""}, subOp.getOperands());
         })
         .Case<db::MulOp>([&](db::MulOp mulOp) {
            return innerExpression({"", " * ", ""}, mulOp.getOperands());
         })
         .Case<db::DivOp>([&](db::DivOp divOp) {
            return innerExpression({"", " / ", ""}, divOp.getOperands());
         })
         .Case<db::DeriveTruth>([&](db::DeriveTruth deriveTruth) {
            return convertExpression(deriveTruth.getOperand().getDefiningOp());
         })
         .Case<db::CastOp>([&](db::CastOp castOp) {
            return innerExpression({"cast(", ")"}, castOp.getVal());
         })
         .Case<db::RuntimeCall>([&](db::RuntimeCall runtimeCall) {
            std::vector<std::string> seperators = {runtimeCall.getFn().str() + "("};
            for (size_t i = 1; i < runtimeCall.getOperands().size(); i++) {
               seperators.push_back(", ");
            }
            seperators.push_back(")");
            return innerExpression(seperators, runtimeCall.getOperands());
         })
         .Case<mlir::arith::ConstantOp>([&](mlir::arith::ConstantOp constantOp) {
            return convertConstant(constantOp.getValue(), constantOp.getType());
         })
         .Case<mlir::scf::YieldOp>([&](mlir::scf::YieldOp yieldOp) {
            return convertExpression(yieldOp.getOperand(0).getDefiningOp());
         })
         .Case<mlir::scf::IfOp>([&](mlir::scf::IfOp ifOp) {
            if (ifOp.getNumResults() == 1) {
               auto condition = convertExpression(ifOp.getOperand().getDefiningOp());
               auto thenExpr = convertExpression(ifOp.getThenRegion().getBlocks().begin()->getTerminator());
               auto elseExpr = convertExpression(ifOp.getElseRegion().getBlocks().begin()->getTerminator());
               return innerExpression({"if ", " then ", " else ", ""}, {condition, thenExpr, elseExpr});
            } else {
               return nlohmann::json{{"type", "expression_leaf"}, {"leaf_type", "unknown"}};
            }
         })
         .Default([](mlir::Operation* op) {
            llvm::errs() << "the following expression could not be converted";
            op->dump();
            return nlohmann::json{{"type", "expression_leaf"}, {"leaf_type", "unknown"}};
         });
   }

   nlohmann::json convertSortSpec(mlir::ArrayAttr sortspecs) {
      nlohmann::json result = nlohmann::json::array();
      for (auto attr : sortspecs) {
         auto sortspecAttr = mlir::cast<relalg::SortSpecificationAttr>(attr);
         result.push_back(nlohmann::json{
            {"value", columnToJSON(sortspecAttr.getAttr())},
            {"direction", sortspecAttr.getSortSpec() == relalg::SortSpec::asc ? "asc" : "desc"}});
      }
      return result;
   }
   nlohmann::json extractHashCondition(BinaryOperator joinOp, bool skipFirstLeft = false) {
      nlohmann::json result = nlohmann::json::array();
      assert(joinOp->hasAttr("leftHash") && joinOp->hasAttr("rightHash"));
      auto leftJoinColumns = mlir::dyn_cast_or_null<mlir::ArrayAttr>(joinOp->getAttr("leftHash"));
      auto rightJoinColumns = mlir::dyn_cast_or_null<mlir::ArrayAttr>(joinOp->getAttr("rightHash"));
      assert(leftJoinColumns && rightJoinColumns && leftJoinColumns.size() - skipFirstLeft == rightJoinColumns.size());
      for (size_t i = skipFirstLeft; i != leftJoinColumns.size(); i++) {
         auto leftColumn = mlir::dyn_cast_or_null<tuples::ColumnRefAttr>(leftJoinColumns[i]);
         auto rightColumn = mlir::dyn_cast_or_null<tuples::ColumnRefAttr>(rightJoinColumns[i - skipFirstLeft]);
         assert(leftColumn && rightColumn);
         result.push_back(innerExpression({"", "=", ""}, {columnToJSON(leftColumn), columnToJSON(rightColumn)}));
      }
      return result;
   }
   void convertJoin(nlohmann::json& result, BinaryOperator joinOp, mlir::Block& predicateBlock, std::string joinType) {
      result["operator"] = "join";
      result["joinType"] = joinType;

      nlohmann::json condition;
      std::string impl = joinOp->hasAttr("impl") ? mlir::cast<mlir::StringAttr>(joinOp->getAttr("impl")).str() : "";
      if (impl == "hash" || impl == "markhash") {
         result["joinImpl"] = "hash";
         result["comparisons"] = extractHashCondition(joinOp);
         result["condition"] = convertExpression(predicateBlock.getTerminator());

      } else if (impl == "indexNestedLoop") {
         result["joinImpl"] = "index";
         result["comparisons"] = extractHashCondition(joinOp, true);
         result["condition"] = convertExpression(predicateBlock.getTerminator());
      } else {
         result["joinImpl"] = "nested";
         result["comparisons"] = nlohmann::json::array();
         result["condition"] = convertExpression(predicateBlock.getTerminator());
      }
   }
   std::string getOperationReference(mlir::Operation* op) {
      if (auto lineOp = mlir::dyn_cast_or_null<mlir::FileLineColLoc>(op->getLoc())) {
         auto fileName = lineOp.getFilename().str();
         auto baseNameStarts = fileName.find_last_of("/");
         if (baseNameStarts != std::string::npos) {
            fileName = fileName.substr(baseNameStarts + 1);
         }
         auto endingStarts = fileName.find(".");
         if (endingStarts != std::string::npos) {
            fileName = fileName.substr(0, endingStarts);
         }
         return fileName + std::string(":") + std::to_string(lineOp.getLine());
      }
      return "";
   }
   double getOutputCardinality(mlir::Operation* op) {
      if (op->hasAttr("rows")) {
         if (auto floatAttr = mlir::dyn_cast_or_null<mlir::FloatAttr>(op->getAttr("rows"))) {
            return floatAttr.getValueAsDouble();
         } else if (auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(op->getAttr("rows"))) {
            return intAttr.getInt();
         }
      }
      return NAN;
   }
   nlohmann::json columnToJSON(tuples::ColumnDefAttr columnDefAttr) {
      return nlohmann::json{
         {"datatype", convertDataType(columnDefAttr.getColumn().type)},
         {"type", "expression_leaf"},
         {"leaf_type", "column"},
         {"displayName", symbolRefToIuString(columnDefAttr.getName())}};
   }
   nlohmann::json columnToJSON(tuples::ColumnRefAttr columnRefAttr) {
      return nlohmann::json{
         {"datatype", convertDataType(columnRefAttr.getColumn().type)},
         {"type", "expression_leaf"},
         {"leaf_type", "column"},
         {"displayName", symbolRefToIuString(columnRefAttr.getName())}};
   }
   nlohmann::json convertOperation(mlir::Operation* op) {
      nlohmann::json result;
      result["ref"] = getOperationReference(op);
      result["type"] = "operator";
      result["consuming"] = nlohmann::json::array();
      for (auto operand : op->getOperands()) {
         if (mlir::isa<tuples::TupleStreamType>(operand.getType())) {
            relalg::TrackTuplesOP trackTuplesOp;
            for (auto* user : operand.getUsers()) {
               if (auto tracking = mlir::dyn_cast_or_null<relalg::TrackTuplesOP>(user)) {
                  trackTuplesOp = tracking;
                  break;
               }
            }
            if (trackTuplesOp) {
               result["consuming"].push_back(nlohmann::json{
                  {"ref", getOperationReference(operand.getDefiningOp())},
                  {"cardinality", tupleCounts.contains(trackTuplesOp.getResultId()) ? tupleCounts.at(trackTuplesOp.getResultId()) : -1},
                  {"estimatedCardinality", getOutputCardinality(operand.getDefiningOp())}});
            } else {
               result["consuming"].push_back(nlohmann::json{
                  {"ref", getOperationReference(operand.getDefiningOp())},
                  {"estimatedCardinality", getOutputCardinality(operand.getDefiningOp())}});
            }
         };
      }

      return llvm::TypeSwitch<mlir::Operation*, nlohmann::json>(op)
         .Case<relalg::BaseTableOp>([&](relalg::BaseTableOp baseTable) {
            result["operator"] = "tablescan";
            result["tablename"] = baseTable.getTableIdentifier().str();
            result["tableSize"] = getOutputCardinality(op);
            result["attributes"] = nlohmann::json::array();
            for (auto columnAttr : baseTable.getColumnsAttr()) {
               result["attributes"].push_back(nlohmann::json{
                  {"attribute", columnAttr.getName().str()},
                  {"column", columnToJSON(mlir::cast<tuples::ColumnDefAttr>(columnAttr.getValue()))}});
            }
            return result;
         })
         .Case<relalg::SelectionOp>([&](relalg::SelectionOp selectionOp) {
            nlohmann::json expression = convertExpression(selectionOp.getPredicate().getBlocks().begin()->getTerminator());
            result["operator"] = "selection";
            result["condition"] = expression;
            return result;
         })
         .Case<relalg::TopKOp>([&](relalg::TopKOp topKOp) {
            result["operator"] = "topk";
            result["order"] = convertSortSpec(topKOp.getSortspecs());
            result["limit"] = topKOp.getMaxRows();
            return result;
         })
         .Case<relalg::SortOp>([&](relalg::SortOp sortOp) {
            result["operator"] = "sort";
            result["order"] = convertSortSpec(sortOp.getSortspecs());
            return result;
         })
         .Case<relalg::AggregationOp>([&](relalg::AggregationOp aggregationOp) {
            result["operator"] = "aggregation";
            result["keys"] = nlohmann::json::array();
            for (auto groupByCol : aggregationOp.getGroupByCols()) {
               auto colRefAttr = mlir::cast<tuples::ColumnRefAttr>(groupByCol);
               result["keys"].push_back(columnToJSON(colRefAttr));
            }
            result["aggregates"] = nlohmann::json::array();
            for (auto [aggregate, aggrResult] : llvm::zip(aggregationOp.getComputedCols(), aggregationOp.getAggrFunc().getBlocks().begin()->getTerminator()->getOperands())) {
               auto colDefAttr = mlir::cast<tuples::ColumnDefAttr>(aggregate);
               nlohmann::json aggrExpr;
               if (auto countStar = mlir::dyn_cast_or_null<relalg::CountRowsOp>(aggrResult.getDefiningOp())) {
                  aggrExpr = innerExpression({"count(*)"}, mlir::ValueRange{});
               } else if (auto aggrFunc = mlir::dyn_cast_or_null<relalg::AggrFuncOp>(aggrResult.getDefiningOp())) {
                  aggrExpr = innerExpression({stringifyEnum(aggrFunc.getFn()).str() + std::string("("), ")"}, {columnToJSON(aggrFunc.getAttr())});
               } else {
                  aggrExpr = nlohmann::json{{"type", "expression_leaf"}, {"leaf_type", "unknown"}};
               }
               result["aggregates"].push_back(nlohmann::json{{"computed", columnToJSON(colDefAttr)}, {"aggregation", aggrExpr}});
            }
            return result;
         })
         .Case<relalg::MaterializeOp>([&](relalg::MaterializeOp materializeOp) {
            result["operator"] = "materialize";
            for (auto [colNameAttr, colRefAttr] : llvm::zip(materializeOp.getColumns(), materializeOp.getCols())) {
               result["output"].push_back(nlohmann::json{
                  {"name", mlir::cast<mlir::StringAttr>(colNameAttr).str()},
                  {"column", columnToJSON(mlir::cast<tuples::ColumnRefAttr>(colRefAttr))}});
            }
            return result;
         })
         .Case<relalg::InnerJoinOp>([&](relalg::InnerJoinOp innerJoinOp) {
            convertJoin(result, innerJoinOp, innerJoinOp.getPredicateBlock(), "inner");
            return result;
         })
         .Case<relalg::SemiJoinOp>([&](relalg::SemiJoinOp semiJoinOp) {
            convertJoin(result, semiJoinOp, semiJoinOp.getPredicateBlock(), "leftsemi");
            return result;
         })
         .Case<relalg::AntiSemiJoinOp>([&](relalg::AntiSemiJoinOp antiSemiJoinOp) {
            convertJoin(result, antiSemiJoinOp, antiSemiJoinOp.getPredicateBlock(), "leftanti");
            return result;
         })
         .Case<relalg::OuterJoinOp>([&](relalg::OuterJoinOp outerJoinOp) {
            convertJoin(result, outerJoinOp, outerJoinOp.getPredicateBlock(), "leftouter");
            return result;
         })
         .Case<relalg::FullOuterJoinOp>([&](relalg::FullOuterJoinOp fullOuterJoinOp) {
            convertJoin(result, fullOuterJoinOp, fullOuterJoinOp.getPredicateBlock(), "outer");
            return result;
         })
         .Case<relalg::SingleJoinOp>([&](relalg::SingleJoinOp singleJoinOp) {
            convertJoin(result, singleJoinOp, singleJoinOp.getPredicateBlock(), "single");
            return result;
         })
         .Case<relalg::MarkJoinOp>([&](relalg::MarkJoinOp singleJoinOp) {
            convertJoin(result, singleJoinOp, singleJoinOp.getPredicateBlock(), "leftmark"); //todo: maybe we need to generate both variants
            result["operator"] = "markJoin";
            result["markColumn"] = columnToJSON(mlir::cast<tuples::ColumnDefAttr>(singleJoinOp.getMarkattr()));
            return result;
         })
         .Case<relalg::MapOp>([&](relalg::MapOp mapOp) {
            result["operator"] = "map";
            result["computed"] = nlohmann::json::array();
            for (auto [column, computed] : llvm::zip(mapOp.getComputedCols(), mapOp.getPredicate().front().getTerminator()->getOperands())) {
               auto columnDefAttr = mlir::cast<tuples::ColumnDefAttr>(column);
               result["computed"].push_back({{"computed", columnToJSON(columnDefAttr)}, {"expression", convertExpression(computed.getDefiningOp())}});
            }
            return result;
         })
         .Case<relalg::LimitOp>([&](relalg::LimitOp limitOp) {
            result["operator"] = "limit";
            result["limit"] = limitOp.getMaxRows();
            return result;
         })
         .Case<relalg::CrossProductOp>([&](relalg::CrossProductOp crossProductOp) {
            result["operator"] = "crossproduct";
            return result;
         })
         .Case<relalg::TmpOp>([&](relalg::TmpOp tmpOp) {
            result["operator"] = "tmp";
            return result;
         })
         .Case<relalg::RenamingOp>([&](relalg::RenamingOp renamingOp) {
            result["operator"] = "rename";
            result["renamed"] = nlohmann::json::array();
            for (auto col : renamingOp.getColumns()) {
               auto colDef = mlir::cast<tuples::ColumnDefAttr>(col);
               result["renamed"].push_back(nlohmann::json{{"old", columnToJSON(mlir::cast<tuples::ColumnRefAttr>(mlir::cast<mlir::ArrayAttr>(colDef.getFromExisting())[0]))}, {"new", columnToJSON(colDef)}});
            }
            return result;
         })
         .Case<relalg::ConstRelationOp>([&](relalg::ConstRelationOp constRelationOp) {
            result["operator"] = "constrelation";
            result["values"] = nlohmann::json::array();
            result["attributes"] = nlohmann::json::array();
            for (auto col : constRelationOp.getColumns()) {
               result["attributes"].push_back(columnToJSON(mlir::cast<tuples::ColumnDefAttr>(col)));
            }
            for (auto row : constRelationOp.getValues()) {
               nlohmann::json rowValues = nlohmann::json::array();
               for (auto c : mlir::cast<mlir::ArrayAttr>(row)) {
                  rowValues.push_back(convertConstant(c, mlir::cast<tuples::ColumnDefAttr>(constRelationOp.getColumns()[rowValues.size()]).getColumn().type));
               }
               result["values"].push_back(rowValues);
            }
            return result;
         })
         .Case<relalg::UnionOp>([&](relalg::UnionOp unionOp) {
            result["operator"] = "union";
            result["semantics"] = unionOp.getSetSemantic() == relalg::SetSemantic::distinct ? "distinct" : "all";
            return result;
         })
         .Case<relalg::ExceptOp>([&](relalg::ExceptOp exceptOp) {
            result["operator"] = "except";
            result["semantics"] = exceptOp.getSetSemantic() == relalg::SetSemantic::distinct ? "distinct" : "all";
            return result;
         })
         .Case<relalg::IntersectOp>([&](relalg::IntersectOp intersectOp) {
            result["operator"] = "intersect";
            result["semantics"] = intersectOp.getSetSemantic() == relalg::SetSemantic::distinct ? "distinct" : "all";
            return result;
         })
         .Case<relalg::WindowOp>([&](relalg::WindowOp windowOp) {
            result["operator"] = "window";
            result["order"] = convertSortSpec(windowOp.getOrderBy());
            result["partition"] = nlohmann::json::array();
            for (auto partitionAttr : windowOp.getPartitionBy()) {
               auto columnRef = mlir::cast<tuples::ColumnRefAttr>(partitionAttr);
               result["partition"].push_back(columnToJSON(columnRef));
            }
            result["frame"] = nlohmann::json::object();
            uint64_t from = windowOp.getFrom(), to = windowOp.getTo();
            int64_t frameStart = *reinterpret_cast<int64_t*>(&from), frameEnd = *reinterpret_cast<int64_t*>(&to);
            if (frameStart == std::numeric_limits<int64_t>::min()) {
               result["frame"]["start"] = "unbounded";
            } else {
               result["frame"]["start"] = std::to_string(-frameStart) + " preceding";
            }
            if (frameEnd == std::numeric_limits<int64_t>::max()) {
               result["frame"]["end"] = "unbounded";
            } else {
               result["frame"]["end"] = std::to_string(frameEnd) + " following";
            }
            result["aggregates"] = nlohmann::json::array();
            for (auto [aggregate, aggrResult] : llvm::zip(windowOp.getComputedCols(), windowOp.getAggrFunc().getBlocks().begin()->getTerminator()->getOperands())) {
               auto colDefAttr = mlir::cast<tuples::ColumnDefAttr>(aggregate);
               nlohmann::json aggrExpr;
               if (auto countStar = mlir::dyn_cast_or_null<relalg::CountRowsOp>(aggrResult.getDefiningOp())) {
                  aggrExpr = innerExpression({"count(*)"}, mlir::ValueRange{}); //todo: quick hack
               } else if (auto aggrFunc = mlir::dyn_cast_or_null<relalg::AggrFuncOp>(aggrResult.getDefiningOp())) {
                  aggrExpr = innerExpression({stringifyEnum(aggrFunc.getFn()).str() + std::string("("), ")"}, {columnToJSON(aggrFunc.getAttr())});
               } else if (auto rankOp = mlir::dyn_cast_or_null<relalg::RankOp>(aggrResult.getDefiningOp())) {
                  aggrExpr = innerExpression({"rank()"}, mlir::ValueRange{});
               } else {
                  aggrExpr = nlohmann::json{{"type", "expression_leaf"}, {"leaf_type", "unknown"}};
               }
               result["aggregates"].push_back(nlohmann::json{{"computed", columnToJSON(colDefAttr)}, {"aggregation", aggrExpr}});
            }
            return result;
         })
         .Case<relalg::ProjectionOp>([&](relalg::ProjectionOp projectionOp) {
            result["operator"] = "projection";
            result["semantics"] = projectionOp.getSetSemantic() == relalg::SetSemantic::distinct ? "distinct" : "all";
            result["columns"] = nlohmann::json::array();
            for (auto column : projectionOp.getCols()) {
               auto columnRef = mlir::cast<tuples::ColumnRefAttr>(column);
               result["columns"].push_back(columnToJSON(columnRef));
            }
            return result;
         })
         .Default([&](mlir::Operation* op) {
            llvm::errs() << "the following operation could not be converted";
            op->dump();
            result["operator"] = "unknown";
            return result;
         });
   }

   public:
   ToJson(mlir::ModuleOp moduleOp, std::unordered_map<uint32_t, int64_t> tupleCounts) : moduleOp(moduleOp), tupleCounts(tupleCounts) {}
   std::string run() {
      moduleOp = moduleOp.clone();
      mlir::func::FuncOp func = mlir::dyn_cast_or_null<mlir::func::FuncOp>(&moduleOp.getRegion().front().front());
      nlohmann::json plan;
      func->walk([&](mlir::Operation* operation) {
         if (auto setResultOp = mlir::dyn_cast_or_null<subop::SetResultOp>(operation)) {
            if (auto relalgQuery = mlir::dyn_cast_or_null<relalg::QueryOp>(setResultOp.getState().getDefiningOp())) {
               for (auto& op : relalgQuery.getQueryOps().front()) {
                  if (!mlir::isa<relalg::TrackTuplesOP, relalg::QueryReturnOp>(&op)) {
                     plan.push_back(convertOperation(&op));
                  }
               }
            }
         }
      });
      return to_string(plan);
   }
};
class TupleCountResultProcessor : public lingodb::execution::ResultProcessor {
   std::unordered_map<uint32_t, int64_t>& tupleCounts;

   public:
   TupleCountResultProcessor(std::unordered_map<uint32_t, int64_t>& tupleCounts) : tupleCounts(tupleCounts) {}
   void process(lingodb::runtime::ExecutionContext* executionContext) override {
      tupleCounts = executionContext->getTupleCounts();
   }
};
void execute(std::string inputFileName, std::string databasePath, std::unordered_map<uint32_t, int64_t>& tupleCounts) {
   auto session = lingodb::runtime::Session::createSession(databasePath, false);
   lingodb::compiler::support::eval::init();
   auto queryExecutionConfig = lingodb::execution::createQueryExecutionConfig(lingodb::execution::ExecutionMode::CHEAP, false);
   queryExecutionConfig->timingProcessor = {};
   queryExecutionConfig->queryOptimizer = {};
   queryExecutionConfig->resultProcessor = std::make_unique<TupleCountResultProcessor>(tupleCounts);
   auto scheduler = lingodb::scheduler::startScheduler();
   auto executer = lingodb::execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig), *session);
   executer->fromFile(inputFileName);
   lingodb::scheduler::awaitEntryTask(std::make_unique<lingodb::execution::QueryExecutionTask>(std::move(executer)));
}
} // namespace

int main(int argc, char** argv) {
   cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");
   mlir::MLIRContext context;
   lingodb::execution::initializeContext(context);
   mlir::OwningOpRef<mlir::ModuleOp> module;
   llvm::SourceMgr sourceMgr;
   mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
   if (int error = loadMLIR(context, module))
      return error;
   std::unordered_map<uint32_t, int64_t> tupleCounts;
   if (databasePath != "-") {
      mlir::PassManager pm(module->getContext());
      pm.addPass(relalg::createTrackTuplesPass());
      if (mlir::failed(pm.run(module.get()))) {
         return 1;
      }
      std::error_code errorCode;
      llvm::raw_fd_ostream file("tmp.mlir", errorCode);
      module.get().print(file);
      file.flush();
      file.close();
      execute("tmp.mlir", databasePath, tupleCounts);
   }

   ToJson toJson(module.get(), tupleCounts);
   llvm::outs() << toJson.run() << "\n";
}