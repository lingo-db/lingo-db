#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/SubOperator/SubOperatorDialect.h"
#include "mlir/Dialect/SubOperator/SubOperatorOps.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"

#include "execution/Execution.h"
#include "execution/ResultProcessing.h"
#include "mlir-support/eval.h"

#include "json.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include <iostream>
#include <list>
#include <queue>

namespace cl = llvm::cl;

cl::opt<std::string> inputFilename(cl::Positional,
                                   cl::desc("<input mlir file>"),
                                   cl::init("-"),
                                   cl::value_desc("filename"));
cl::opt<std::string> databasePath(cl::Positional,
                                  cl::desc("<path to  database>"),
                                  cl::init("-"),
                                  cl::value_desc("database"));

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
   std::unordered_map<mlir::Operation*, size_t> tmpRelations;
   std::unordered_map<uint32_t, int64_t> tupleCounts;
   size_t operatorId = 0;

   /// Common helper functions
   // Returns the value of the field "iu" in the nested json if present
   std::optional<std::string> getIuHelper(const nlohmann::json& json) {
      if (json.is_array()) {
         for (auto arrayElement : json) {
            if (auto result = getIuHelper(arrayElement)) {
               return result;
            }
         }
         return std::nullopt;
      } else if (json.contains("iu")) {
         return std::make_optional<std::string>(json["iu"]);
      } else if (json.contains("input")) {
         return getIuHelper(json["input"]);
      } else {
         if (json.contains("left") && getIuHelper((json["left"]))) {
            return getIuHelper(json["left"]);
         } else if (json.contains("right") && getIuHelper(json["right"])) {
            return getIuHelper((json["right"]));
         }
         return std::nullopt;
      }
   }

   // Return the value of the field "iu" in the nested json, fails if "iu" not present
   std::string getIu(const nlohmann::json& json) {
      if (auto res = getIuHelper(json)) {
         return res.value();
      } else {
         llvm::errs() << to_string(json) << "\n";
         assert(false && "did not find iu in");
         return {};
      }
   }

   // Returns the index of the "iu" variable in the array
   size_t getOffsetOfIuInArray(nlohmann::json array, std::string iu) {
      size_t result = 0;
      for (auto a : array) {
         if (a["iu"] == iu)
            return result;
         else
            result++;
      }
      assert(false && "array does not contain wanted iu");
   }

   // Check if there is an operation tracking the tuples produced by the current operation
   std::optional<int64_t> getAnalyzedCardinality(mlir::Operation* op) {
      for (auto* user : op->getUsers()) {
         if (auto trackTuplesOp = mlir::dyn_cast_or_null<mlir::relalg::TrackTuplesOP>(user)) {
            if (tupleCounts.contains(trackTuplesOp.getResultId())) {
               return tupleCounts.at(trackTuplesOp.getResultId());
            } else {
               return {};
            }
         }
      }
      return std::nullopt;
   }

   // Transform iu from symbol ref to string form
   std::string symbolRefToIuString(mlir::SymbolRefAttr symbolRefAttr) {
      return symbolRefAttr.getRootReference().str() + "@" + symbolRefAttr.getLeafReference().str();
   }

   /// Json producing helper functions
   // Produce iuref from columnRef
   nlohmann::json columnRefAttrToIuref(mlir::tuples::ColumnRefAttr columnRefAttr) {
      nlohmann::json result;
      result["iu"] = symbolRefToIuString(columnRefAttr.getName());
      result["type"] = convertType(columnRefAttr.getColumn().type);
      result["expression"] = "iuref";
      return result;
   }

   // Produce iuref from columnDef
   nlohmann::json columnDefAttrToIuref(mlir::tuples::ColumnDefAttr columnDefAttr) {
      return {
         {"iu", symbolRefToIuString(columnDefAttr.getName())},
         {"type", convertType(columnDefAttr.getColumn().type)}};
   }

   // Convert mlir Type to umbra json type
   nlohmann::json convertType(mlir::Type type) {
      return llvm::TypeSwitch<mlir::Type, nlohmann::json>(type)
         .Case<mlir::db::DecimalType>([](mlir::db::DecimalType decimalType) {
            return nlohmann::json{
               {"type", "numeric"},
               {"precision", decimalType.getP()},
               {"scale", decimalType.getS()}};
         })
         .Case<mlir::IndexType>([](mlir::IndexType indexType) {
            return nlohmann::json{
               {"type", "numeric"},
               {"numBits", 64},
               {"signed", false}};
         })
         .Case<mlir::IntegerType>([](mlir::IntegerType integerType) {
            return nlohmann::json{
               {"type", "numeric"},
               {"numBits", integerType.getWidth()},
               {"signed", integerType.isSigned()}};
         })
         .Case<mlir::db::DateType>([](mlir::db::DateType dateType) {
            return nlohmann::json{
               // Umbra treats date as int, we have (annotated) strings
               {"type", "text"}};
         })
         .Case<mlir::db::NullableType>([&](mlir::db::NullableType nullableType) {
            return convertType(nullableType.getType());
         })
         .Case<mlir::db::StringType>([&](mlir::db::StringType stringType) {
            return nlohmann::json{{"type", "text"}};
         })
         .Case<mlir::db::CharType>([](mlir::db::CharType charType) {
            return nlohmann::json{{"type", "text"}};
         })
         .Case<mlir::FloatType>([](mlir::FloatType floatType) {
            return nlohmann::json{{"type", "float"}};
         })
         .Default([](mlir::Type type) {
            llvm::errs() << "type could not be converted ";
            type.dump();
            llvm::errs() << "\n";
            return nlohmann::json();
         });
   }

   // Match mlir runtime call names to umbra function names
   std::optional<std::string> convertRuntimeCallFuncName(std::string unconvertedName) {
      if (unconvertedName == "Like") {
         return "like";
      }
      if (unconvertedName == "ExtractFromDate") {
         // TODO: is actually a different name for year, day, and month
         return "extract";
      } else {
         return std::nullopt;
      }
   }

   nlohmann::json convertConstant(mlir::Attribute attr, mlir::Type type) {
      nlohmann::json result{
         {"expression", "const"},
         {"value", nlohmann::json::object()}};
      result["value"]["type"] = convertType(type);
      if (auto integerAttr = attr.dyn_cast_or_null<mlir::IntegerAttr>()) {
         result["value"]["value"] = integerAttr.getInt();
      } else if (auto floatAttr = attr.dyn_cast_or_null<mlir::FloatAttr>()) {
         result["value"]["value"] = floatAttr.getValueAsDouble();
      } else if (auto stringAttr = attr.dyn_cast_or_null<mlir::StringAttr>()) {
         result["value"]["value"] = stringAttr.str();
      } else if (auto boolAttr = attr.dyn_cast_or_null<mlir::BoolAttr>()) {
         result["value"]["value"] = boolAttr.getValue();
      } else {
         llvm::errs() << "Constant could not be converted ";
         attr.dump();
      }
      return result;
   }

   std::string convertCmpPredicate(mlir::db::DBCmpPredicate cmpPredicate) {
      switch (cmpPredicate) {
         case mlir::db::DBCmpPredicate::eq:
            return "=";
         case mlir::db::DBCmpPredicate::gt:
            return ">";
         case mlir::db::DBCmpPredicate::gte:
            return ">=";
         case mlir::db::DBCmpPredicate::lt:
            return "<";
         case mlir::db::DBCmpPredicate::lte:
            return "<=";
         case mlir::db::DBCmpPredicate::neq:
            return "<>";
         case mlir::db::DBCmpPredicate::isa:
            return "isa";
         default:
            assert(false);
      }
   }

   nlohmann::json convertExpression(mlir::Operation* operation) {
      return llvm::TypeSwitch<mlir::Operation*, nlohmann::json>(operation)
         .Case<mlir::tuples::ReturnOp>([&](mlir::tuples::ReturnOp returnOp) {
            if (returnOp->getOperands().size()) return convertExpression(returnOp.getOperand(0).getDefiningOp());
            return nlohmann::json{};
         })
         .Case<mlir::arith::ExtUIOp>([&](mlir::arith::ExtUIOp extUiOp) { return convertExpression(extUiOp.getIn().getDefiningOp()); })
         .Case<mlir::arith::ExtSIOp>([&](mlir::arith::ExtSIOp extUiOp) { return convertExpression(extUiOp.getIn().getDefiningOp()); })
         .Case<mlir::arith::SelectOp>([&](mlir::arith::SelectOp selectOp) {
            return nlohmann::json{
               {"expression", "call"},
               {"fn", "select"},
               {"input", {convertExpression(selectOp.getCondition().getDefiningOp()), convertExpression(selectOp.getTrueValue().getDefiningOp()), convertExpression(selectOp.getFalseValue().getDefiningOp())}}};
         })
         .Case<mlir::db::NotOp>([&](mlir::db::NotOp notOp) {
            if (auto isNullOp = mlir::dyn_cast_or_null<mlir::db::IsNullOp>(notOp.getOperand().getDefiningOp())) {
               return nlohmann::json{
                  {"expression", "isnotnull"},
                  {"input", convertExpression(isNullOp.getOperand().getDefiningOp())}};
            } else {
               return nlohmann::json{
                  {"expression", "not"},
                  {"input", convertExpression(notOp.getOperand().getDefiningOp())}};
            }
         })
         .Case<mlir::db::NullOp>([&](mlir::db::NullOp isNullOp) {
            return nlohmann::json{
               {"expression", "null"},
            };
         })
         .Case<mlir::db::IsNullOp>([&](mlir::db::IsNullOp isNullOp) {
            return nlohmann::json{
               {"expression", "isnull"},
               {"input", convertExpression(isNullOp.getOperand().getDefiningOp())}};
         })
         .Case<mlir::db::AsNullableOp>([&](mlir::db::AsNullableOp asNullableOp) {
            return convertExpression(asNullableOp->getOperand(0).getDefiningOp());
         })
         .Case<mlir::db::BetweenOp>([&](mlir::db::BetweenOp betweenOp) {
            return nlohmann::json{
               {"expression", "between"},
               {"input", {convertExpression(betweenOp.getOperand(0).getDefiningOp()), convertExpression(betweenOp.getLower().getDefiningOp()), convertExpression(betweenOp.getUpper().getDefiningOp())}}};
         })
         .Case<mlir::db::OneOfOp>([&](mlir::db::OneOfOp oneOfOp) {
            nlohmann::json result{
               {"expression", "in"},
               {"input", {convertExpression(oneOfOp.getVal().getDefiningOp())}},
               {"values", nlohmann::json::array({})}};
            for (auto value : oneOfOp.getVals()) {
               result["values"].push_back(convertExpression(value.getDefiningOp())["value"]);
            }
            return result;
         })
         .Case<mlir::db::CmpOp>([&](mlir::db::CmpOp cmpOp) {
            return nlohmann::json{
               {"expression", "compare"},
               {"direction", convertCmpPredicate(cmpOp.getPredicate())},
               {"left", convertExpression(cmpOp.getLeft().getDefiningOp())},
               {"right", convertExpression(cmpOp.getRight().getDefiningOp())}};
         })
         .Case<mlir::db::AndOp>([&](mlir::db::AndOp andOp) {
            nlohmann::json result{
               {"expression", "and"},
               {"input", nlohmann::json::array({})}};
            for (auto input : andOp->getOperands()) {
               result["input"].push_back(convertExpression(input.getDefiningOp()));
            }
            return result;
         })
         .Case<mlir::db::OrOp>([&](mlir::db::OrOp orOp) {
            nlohmann::json result{
               {"expression", "or"},
               {"input", nlohmann::json::array({})}};
            for (auto input : orOp->getOperands()) {
               result["input"].push_back(convertExpression(input.getDefiningOp()));
            }
            return result;
         })
         .Case<mlir::db::AddOp>([&](mlir::db::AddOp addOp) {
            return nlohmann::json{
               {"expression", "add"},
               {"left", convertExpression(addOp.getLeft().getDefiningOp())},
               {"right", convertExpression(addOp.getRight().getDefiningOp())}};
         })
         .Case<mlir::db::SubOp>([&](mlir::db::SubOp subOp) {
            return nlohmann::json{
               {"expression", "sub"},
               {"left", convertExpression(subOp.getLeft().getDefiningOp())},
               {"right", convertExpression(subOp.getRight().getDefiningOp())}};
         })
         .Case<mlir::db::MulOp>([&](mlir::db::MulOp mulOp) {
            return nlohmann::json{
               {"expression", "mul"},
               {"left", convertExpression(mulOp.getLeft().getDefiningOp())},
               {"right", convertExpression(mulOp.getRight().getDefiningOp())}};
         })
         .Case<mlir::db::DivOp>([&](mlir::db::DivOp divOp) {
            return nlohmann::json{
               {"expression", "div"},
               {"left", convertExpression(divOp.getLeft().getDefiningOp())},
               {"right", convertExpression(divOp.getRight().getDefiningOp())}};
         })
         .Case<mlir::db::CastOp>([&](mlir::db::CastOp castOp) {
            return convertExpression(castOp.getOperand().getDefiningOp());
         })
         .Case<mlir::db::ConstantOp>([&](mlir::db::ConstantOp constantOp) {
            return convertConstant(constantOp.getValue(), constantOp.getType());
         })
         .Case<mlir::arith::ConstantOp>([&](mlir::arith::ConstantOp constantOp) {
            return convertConstant(constantOp.getValue(), constantOp.getType());
         })
         .Case<mlir::db::RuntimeCall>([&](mlir::db::RuntimeCall runtimeCall) {
            nlohmann::json result = {
               {"expression", "call"},
               {"fn", runtimeCall.getFn().str()},
               {"input", nlohmann::json::array()}};
            for (auto arg : runtimeCall->getOperands()) {
               result["input"].push_back(convertExpression(arg.getDefiningOp()));
            }
            return result;
         })
         .Case<mlir::scf::IfOp>([&](mlir::scf::IfOp ifOp) {
            return nlohmann::json{
               {"expression", "searchedcase"},
               {"cases", nlohmann::json::array({{{"condition", convertExpression(ifOp.getOperand().getDefiningOp())}, {"result", convertExpression(ifOp.getThenRegion().getBlocks().begin()->getTerminator())}}})},
               {"else", convertExpression(ifOp.getElseRegion().getBlocks().begin()->getTerminator())}};
         })
         .Case<mlir::db::DeriveTruth>([&](mlir::db::DeriveTruth deriveTruth) {
            return convertExpression(deriveTruth.getOperand().getDefiningOp());
         })
         .Case<mlir::scf::YieldOp>([&](mlir::scf::YieldOp yieldOp) {
            return convertExpression(yieldOp.getOperand(0).getDefiningOp());
         })
         .Case<mlir::tuples::GetColumnOp>([&](mlir::tuples::GetColumnOp getColumnOp) {
            return columnRefAttrToIuref(getColumnOp.getAttr());
         })
         .Default([](mlir::Operation* op) {
            llvm::errs() << "the following expression could not be converted";
            op->dump();
            return nlohmann::json();
         });
   }

   // creates new iu with the value computed in the predicate region
   nlohmann::json mapComputation(mlir::tuples::ColumnDefAttr columnDefAttr, mlir::Region& predicate) {
      nlohmann::json result;
      result["iu"] = columnDefAttrToIuref(columnDefAttr);
      result["exp"] = convertExpression(predicate.getBlocks().begin()->getTerminator());
      return result;
   }

   // Extracts join condition from leftHash and rightHash attributes
   // skipFirstLeft is used for indexNestedLoop joins as the first left value contains the table name
   nlohmann::json extractHashCondition(BinaryOperator joinOp, bool skipFirstLeft = false) {
      assert(joinOp->hasAttr("leftHash") && joinOp->hasAttr("rightHash"));
      auto leftJoinColumns = mlir::dyn_cast_or_null<mlir::ArrayAttr>(joinOp->getAttr("leftHash"));
      auto rightJoinColumns = mlir::dyn_cast_or_null<mlir::ArrayAttr>(joinOp->getAttr("rightHash"));
      assert(leftJoinColumns && rightJoinColumns && leftJoinColumns.size() - skipFirstLeft == rightJoinColumns.size());
      nlohmann::json result{
         {"expression", "and"},
         {"input", nlohmann::json::array()}};
      for (size_t i = skipFirstLeft; i != leftJoinColumns.size(); i++) {
         auto leftColumn = leftJoinColumns[i].dyn_cast_or_null<mlir::tuples::ColumnRefAttr>();
         auto rightColumn = rightJoinColumns[i - skipFirstLeft].dyn_cast_or_null<mlir::tuples::ColumnRefAttr>();
         assert(leftColumn && rightColumn);
         result["input"].push_back(nlohmann::json{
            {"expression", "compare"},
            {"left", columnRefAttrToIuref(leftColumn)},
            {"right", columnRefAttrToIuref(rightColumn)},
            {"direction", "="}});
      }
      return result;
   }

   /// Abstracted operator conversions
   nlohmann::json convertJoin(BinaryOperator joinOp, mlir::Block& predicateBlock, std::string joinType) {
      nlohmann::json condition;
      std::string impl = joinOp->hasAttr("impl") ? joinOp->getAttr("impl").cast<mlir::StringAttr>().str() : "";
      if (impl == "hash" || impl == "markhash") {
         impl += "join";
         condition = extractHashCondition(joinOp);
      } else if (impl == "indexNestedLoop") {
         condition = extractHashCondition(joinOp, true);
      } else {
         condition = convertExpression(predicateBlock.getTerminator());
      }
      std::optional<double> cardinality = std::nullopt;
      if (joinOp->hasAttr("rows")) {
         if (auto floatAttr = joinOp->getAttr("rows").dyn_cast_or_null<mlir::FloatAttr>()) {
            cardinality = floatAttr.getValueAsDouble();
         } else if (auto intAttr = joinOp->getAttr("rows").dyn_cast_or_null<mlir::IntegerAttr>()) {
            cardinality = intAttr.getInt();
         }
      }
      return {
         {"operator", "join"},
         {"physicalOperator", impl},
         {"operatorId", operatorId - 1},
         {"analyzePlanId", operatorId - 1},
         {"cardinality", cardinality ? cardinality.value() : 0},
         {"type", joinType},
         {"left", convertOperation(joinOp.leftChild())},
         {"right", convertOperation(joinOp.rightChild())},
         {"condition", condition}};
   }

   auto convertSetOperationArguments(std::vector<nlohmann::json> inputs, mlir::ArrayAttr mapping) {
      size_t numInputs = inputs.size();
      auto result = nlohmann::json::array({});
      for (size_t i = 0; i != numInputs; ++i) {
         nlohmann::json argument{
            {"input", inputs[i]},
            {"columns", nlohmann::json::array({})}};
         for (auto mappingForColumn : mapping) {
            auto columnDef = mappingForColumn.cast<mlir::tuples::ColumnDefAttr>();
            argument["columns"].push_back(nlohmann::json{
               {"expression", "iuref"},
               {"iu", symbolRefToIuString(columnDef.getFromExisting().cast<mlir::ArrayAttr>()[i].cast<mlir::tuples::ColumnRefAttr>().getName())}});
         }
         result.push_back(argument);
      }
      return result;
   };

   // Handles both sort and topK as they only differ by the optional limit attribute
   nlohmann::json convertSort(nlohmann::json result, mlir::Operation* input, mlir::ArrayAttr sortspecs, std::optional<uint32_t> maxRows = std::nullopt) {
      result["operator"] = "sort";
      result["physicalOperator"] = "sort";
      result["input"] = convertOperation(input);
      result["order"] = nlohmann::json::array();
      for (auto attr : sortspecs) {
         auto sortspecAttr = attr.cast<mlir::relalg::SortSpecificationAttr>();
         result["order"].push_back(nlohmann::json{
            {"value", {{"expression", "iuref"}, {"iu", symbolRefToIuString(sortspecAttr.getAttr().getName())}}},
            {"collate", ""}});
      }
      if (maxRows) result["limit"] = maxRows.value();
      return result;
   }

   /// Main conversion function
   nlohmann::json convertOperation(mlir::Operation* op) {
      nlohmann::json result;
      result["operatorId"] = operatorId++;
      result["analyzePlanId"] = result["operatorId"];
      if (auto analyzedCardinality = getAnalyzedCardinality(op)) result["analyzePlanCardinality"] = analyzedCardinality.value();
      if (op->hasAttr("rows")) {
         if (auto floatAttr = op->getAttr("rows").dyn_cast_or_null<mlir::FloatAttr>()) {
            result["cardinality"] = floatAttr.getValueAsDouble();
         } else if (auto intAttr = op->getAttr("rows").dyn_cast_or_null<mlir::IntegerAttr>()) {
            result["cardinality"] = intAttr.getInt();
         }
      }
      return llvm::TypeSwitch<mlir::Operation*, nlohmann::json>(op)
         .Case<mlir::relalg::BaseTableOp>([&](mlir::relalg::BaseTableOp baseTable) {
            result["operator"] = "tablescan";
            result["physicalOperator"] = "tablescan";
            result["tablename"] = baseTable.getTableIdentifier().str();
            // Cardinality estimate is accurate for base tables, if selections are applied directly estimated cardinality is overwritten later
            result["tableSize"] = result["cardinality"];
            result["attributes"] = nlohmann::json::array({});
            for (auto columnAttr : baseTable.getColumnsAttr()) {
               result["attributes"].push_back(nlohmann::json{
                  {"name", columnAttr.getName().str()},
                  {"iu", symbolRefToIuString(columnAttr.getValue().cast<mlir::tuples::ColumnDefAttr>().getName())}});
            }
            result["table"] = nlohmann::json{{"type", "table"}, {"id", 0}};
            result["tid"] = nlohmann::json{{"iu", "tid"}, {"type", {"type", "bigint"}}};
            result["tableoid"] = nlohmann::json{{"type", "tableoid"}, {"type", {"type", "integer"}}};
            result["rowstate"] = nlohmann::json{{"iu", "rowstate"}, {"type", {"type", "bigint"}}};
            // Filled by folded selections
            result["restrictions"] = nlohmann::json::array({});
            // Unused, functionality unclear
            result["residuals"] = nlohmann::json::array({});
            return result;
         })
         .Case<mlir::relalg::ConstRelationOp>([&](mlir::relalg::ConstRelationOp constRelationOp) {
            result["operator"] = "inlinetable";
            result["physicalOperator"] = "inlinetable";
            result["cardinality"] = constRelationOp.getValues().size();
            result["values"] = nlohmann::json::array({});
            result["attributes"] = nlohmann::json::array({});

            for (auto col : constRelationOp.getColumns()) {
               result["attributes"].push_back(nlohmann::json{
                  {"name", ""},
                  {"iu", columnDefAttrToIuref(col.cast<mlir::tuples::ColumnDefAttr>())}});
            }
            for (auto row : constRelationOp.getValues()) {
               for (auto c : llvm::zip(row.cast<mlir::ArrayAttr>(), constRelationOp.getColumns())) {
                  result["values"].push_back(convertConstant(std::get<0>(c), std::get<1>(c).cast<mlir::tuples::ColumnDefAttr>().getColumn().type));
               }
            }
            // Filled by folded selections
            result["restrictions"] = nlohmann::json::array({});
            // Unused, functionality unclear
            result["residuals"] = nlohmann::json::array({});
            return result;
         })
         .Case<mlir::relalg::SelectionOp>([&](mlir::relalg::SelectionOp selectionOp) {
            nlohmann::json expression = convertExpression(selectionOp.getPredicate().getBlocks().begin()->getTerminator());
            mlir::Operation* inputOp = selectionOp.getOperand().getDefiningOp();
            nlohmann::json innerResult = convertOperation(inputOp);
            assert(innerResult.contains("operator"));
            // Fold selection ops into other selection ops, joins or table scans for better visual clarity
            if (innerResult["operator"] == "tablescan" && expression.contains("direction")) {
               assert(innerResult.contains("restrictions"));
               assert(innerResult.contains("cardinality"));
               assert(expression.contains("left") && expression["left"].contains("iu"));
               innerResult["restrictions"].push_back(nlohmann::json{
                  {"attribute", getOffsetOfIuInArray(innerResult["attributes"], expression["left"]["iu"])},
                  {"mode", expression["direction"]},
                  {"value", expression["right"]}});
               innerResult["cardinality"] = selectionOp->getAttr("rows").cast<mlir::FloatAttr>().getValueAsDouble();
               if (result.contains("analyzePlanCardinality")) innerResult["analyzePlanCardinality"] = result["analyzePlanCardinality"];
               return innerResult;
            } else if (innerResult["operator"] == "tablescan") {
               assert(innerResult.contains("restrictions"));
               assert(innerResult.contains("cardinality"));
               assert(innerResult.contains("attributes"));
               innerResult["restrictions"].push_back(nlohmann::json{
                  {"attribute", getOffsetOfIuInArray(innerResult["attributes"], getIu(expression))},
                  {"mode", "filter"},
                  {"value", expression}});
               innerResult["cardinality"] = selectionOp->getAttr("rows").cast<mlir::FloatAttr>().getValueAsDouble();
               if (result.contains("analyzePlanCardinality")) innerResult["analyzePlanCardinality"] = result["analyzePlanCardinality"];
               return innerResult;
            } else {
               // No operation to fold into found, use standalone select
               result["operator"] = "select";
               result["physicalOperator"] = "select";
               result["input"] = innerResult;
               result["condition"] = expression;
               return result;
            }
         })
         .Case<mlir::relalg::MapOp>([&](mlir::relalg::MapOp mapOp) {
            // Fold consecutive map operations for better visual clarity
            if (mlir::dyn_cast_or_null<mlir::relalg::MapOp>(mapOp.getOperand().getDefiningOp())) {
               result = convertOperation(mapOp.getOperand().getDefiningOp());
            } else {
               result["operator"] = "map";
               result["physicalOperator"] = "map";
               result["input"] = convertOperation(mapOp.getOperand().getDefiningOp());
               result["values"] = nlohmann::json::array();
            }
            for (auto column : mapOp.getComputedCols()) {
               auto columnDefAttr = column.dyn_cast_or_null<mlir::tuples::ColumnDefAttr>();
               result["values"].push_back(mapComputation(columnDefAttr, mapOp.getPredicate()));
            }
            return result;
         })
         .Case<mlir::relalg::LimitOp>([&](mlir::relalg::LimitOp limitOp) {
            return convertOperation(limitOp.getOperand().getDefiningOp());
         })
         .Case<mlir::relalg::TopKOp>([&](mlir::relalg::TopKOp topKOp) {
            return convertSort(result, topKOp.getOperand().getDefiningOp(), topKOp.getSortspecs(), topKOp.getMaxRows());
         })
         .Case<mlir::relalg::CrossProductOp>([&](mlir::relalg::CrossProductOp crossProductOp) {
            result["operator"] = "join";
            result["physicalOperator"] = "nljoin";
            result["type"] = "inner";
            result["condition"] = nlohmann::json{};
            result["condition"]["expression"] = "const";
            result["condition"]["value"] = nlohmann::json{};
            result["condition"]["value"]["type"] = nlohmann::json{{"type", "bool"}};
            result["left"] = convertOperation(crossProductOp.getLeft().getDefiningOp());
            result["right"] = convertOperation(crossProductOp.getRight().getDefiningOp());
            return result;
         })
         .Case<mlir::relalg::RenamingOp>([&](mlir::relalg::RenamingOp renamingOp) {
            return convertOperation(renamingOp.getOperand().getDefiningOp());
         })
         .Case<mlir::relalg::TmpOp>([&](mlir::relalg::TmpOp tmpOp) {
            result["operator"] = "pipelinebreakerscan";
            result["physicalOperator"] = "tempscan";
            if (!tmpRelations.contains(tmpOp.getOperation())) {
               size_t tmpOperatorId = operatorId++;
               result["pipelineBreaker"] = nlohmann::json{
                  {"operator", "temp"},
                  {"physicalOperator", "temp"},
                  {"operatorId", tmpOperatorId},
                  {"input", convertOperation(tmpOp.getOperand().getDefiningOp())}};
               tmpRelations[tmpOp.getOperation()] = tmpOperatorId;
            }
            result["scannedOperator"] = tmpRelations[tmpOp.getOperation()];
            return result;
         })
         .Case<mlir::relalg::AggregationOp>([&](mlir::relalg::AggregationOp aggregationOp) {
            result["operator"] = "groupby";
            result["physicalOperator"] = "groupby";
            result["input"] = convertOperation(aggregationOp.getOperand().getDefiningOp());
            result["key"] = nlohmann::json::array();
            result["values"] = nlohmann::json::array();
            size_t arg = 0;
            for (auto groupByCol : aggregationOp.getGroupByCols()) {
               auto colRefAttr = groupByCol.cast<mlir::tuples::ColumnRefAttr>();
               result["key"].push_back(nlohmann::json{
                  {"arg", arg++},
                  {"iu", columnRefAttrToIuref(colRefAttr)}});
               result["values"].push_back(nlohmann::json{
                  {"expression", "iuref"},
                  {"iu", symbolRefToIuString(colRefAttr.getName())}});
            }
            result["aggregates"] = nlohmann::json::array();
            std::vector<nlohmann::json> computedColIus;
            for (auto aggregate : aggregationOp.getComputedCols()) {
               auto colDefAttr = aggregate.cast<mlir::tuples::ColumnDefAttr>();
               computedColIus.push_back(columnDefAttrToIuref(colDefAttr));
            }
            size_t index = 0;
            for (auto aggrResult : aggregationOp.getAggrFunc().getBlocks().begin()->getTerminator()->getOperands()) {
               std::string op;
               if (auto countStar = mlir::dyn_cast_or_null<mlir::relalg::CountRowsOp>(aggrResult.getDefiningOp())) {
                  op = "count";
                  result["values"].push_back(nlohmann::json{
                     {"expression", "const"},
                     {"value", {{"type", "bool"}, {"value", true}}}});
               } else if (auto aggrFunc = mlir::dyn_cast_or_null<mlir::relalg::AggrFuncOp>(aggrResult.getDefiningOp())) {
                  op = stringifyEnum(aggrFunc.getFn());
                  result["values"].push_back(nlohmann::json{
                     {"expression", "iuref"},
                     {"iu", symbolRefToIuString(aggrFunc.getAttr().getName())}});
               } else {
                  assert(false && "unknown aggregate function for group by");
               }
               result["aggregates"].push_back(nlohmann::json{
                  {"op", op},
                  {"iu", computedColIus[index]},
                  {"arg", result["key"].size() + index++}});
            }
            result["order"] = nlohmann::json::array({});
            result["groupingsets"] = nlohmann::json::array({});
            return result;
         })
         .Case<mlir::relalg::GroupJoinOp>([&](mlir::relalg::GroupJoinOp groupJoinOp) {
            result["operator"] = "groupjoin";
            result["physicalOperator"] = "groupjoin";
            result["behavior"] = stringifyEnum(groupJoinOp.getBehaviorAttr().getValue());
            result["left"] = convertOperation(groupJoinOp.getLeft().getDefiningOp());
            result["right"] = convertOperation(groupJoinOp.getRight().getDefiningOp());
            result["keyLeft"] = nlohmann::json::array({});
            result["valuesLeft"] = nlohmann::json::array({});
            result["aggregatesLeft"] = nlohmann::json::array({});
            size_t leftIndex = 0;
            for (auto keyColumn : groupJoinOp.getLeftCols()) {
               result["valuesLeft"].push_back(columnRefAttrToIuref(keyColumn.cast<mlir::tuples::ColumnRefAttr>()));
               result["keyLeft"].push_back(nlohmann::json{
                  {"arg", leftIndex++},
                  {"iu", columnRefAttrToIuref(keyColumn.cast<mlir::tuples::ColumnRefAttr>())}});
            }
            result["keyRight"] = nlohmann::json::array({});
            result["valuesRight"] = nlohmann::json::array({});
            result["aggregatesRight"] = nlohmann::json::array({});
            size_t rightIndex = 0;
            for (auto keyColumn : groupJoinOp.getRightCols()) {
               result["valuesRight"].push_back(columnRefAttrToIuref(keyColumn.cast<mlir::tuples::ColumnRefAttr>()));
               result["keyRight"].push_back(nlohmann::json{
                  {"arg", rightIndex++},
                  {"iu", columnRefAttrToIuref(keyColumn.cast<mlir::tuples::ColumnRefAttr>())}});
            }

            std::unordered_map<std::string, nlohmann::json> mappedCols;
            auto mapReturn = mlir::cast<mlir::tuples::ReturnOp>(groupJoinOp.getMapFunc().getBlocks().begin()->getTerminator());
            size_t mapIndex = 0;
            for (auto returnValue : mapReturn->getOperands()) {
               std::string mappedColName = symbolRefToIuString(groupJoinOp.getMappedCols()[mapIndex++].cast<mlir::tuples::ColumnDefAttr>().getName());
               mappedCols[mappedColName] = convertExpression(returnValue.getDefiningOp());
            }

            auto aggrReturn = mlir::cast<mlir::tuples::ReturnOp>(groupJoinOp.getAggrFunc().getBlocks().begin()->getTerminator());
            size_t aggrIndex = 0;
            for (auto returnValue : aggrReturn->getOperands()) {
               // We don't make the distinction between left and right aggregates -> put everything to left
               auto aggrFuncOp = mlir::cast<mlir::relalg::AggrFuncOp>(returnValue.getDefiningOp());
               std::string aggrFuncColName = symbolRefToIuString(aggrFuncOp.getAttr().getName());
               if (mappedCols.contains(aggrFuncColName)) {
                  result["valuesLeft"].push_back(mappedCols[aggrFuncColName]);
               } else {
                  result["valuesLeft"].push_back(columnRefAttrToIuref(aggrFuncOp.getAttr()));
               }
               result["aggregatesLeft"].push_back(nlohmann::json{
                  {"op", stringifyEnum(aggrFuncOp.getFn())},
                  {"arg", leftIndex++},
                  {"iu", columnDefAttrToIuref(groupJoinOp.getComputedCols()[aggrIndex++].cast<mlir::tuples::ColumnDefAttr>())}});
            };

            return result;
         })
         .Case<mlir::relalg::WindowOp>([&](mlir::relalg::WindowOp windowOp) {
            result["operator"] = "window";
            result["physicalOperator"] = "window";
            result["input"] = convertOperation(windowOp.getOperand().getDefiningOp());
            result["values"] = nlohmann::json::array({});

            // Collect values for partitioning and sorting
            for (auto partitionAttr : windowOp.getPartitionBy()) {
               auto columnRef = mlir::cast<mlir::tuples::ColumnRefAttr>(partitionAttr);
               result["values"].push_back(nlohmann::json{
                  {"expression", "iuref"},
                  {"iu", symbolRefToIuString(columnRef.getName())}});
            }
            for (auto orderAttr : windowOp.getPartitionBy()) {
               auto columnRef = mlir::cast<mlir::tuples::ColumnRefAttr>(orderAttr);
               result["values"].push_back(nlohmann::json{
                  {"expression", "iuref"},
                  {"iu", symbolRefToIuString(columnRef.getName())}});
            }

            size_t index = 0;
            for (auto windowResult : windowOp.getAggrFunc().getBlocks().front().getTerminator()->getOperands()) {
               auto aggrFunc = mlir::cast<mlir::relalg::AggrFuncOp>(windowResult.getDefiningOp());
               // Add aggregated value to used values
               result["values"].push_back(nlohmann::json{
                  {"expression", "iuref"},
                  {"iu", symbolRefToIuString(aggrFunc.getAttr().getName())}});

               // Save for each function individually the used partitioning and sorting keys -> assumed identical
               nlohmann::json partitionJson{
                  {"key", nlohmann::json::array()},
                  {"orders", nlohmann::json::array({})}};
               for (size_t i = 0; i != windowOp.getPartitionBy().size(); ++i) {
                  partitionJson["key"].push_back(nlohmann::json{{"value", i}});
               }
               nlohmann::json ordersJson{
                  {"order", nlohmann::json::array({})},
                  {"operations", nlohmann::json::array({})}};
               for (size_t i = 0; i != windowOp.getOrderBy().size(); ++i) {
                  ordersJson["order"].push_back(nlohmann::json{{"value", i + windowOp.getPartitionBy().size()}});
               }
               nlohmann::json operationJson{
                  {"frame", nlohmann::json::object({})},
                  {"op", nlohmann::json::object({})}};
               operationJson["frame"]["range"] = "rows";
               operationJson["frame"]["exclude"] = "none";
               uint64_t from = windowOp.getFrom(), to = windowOp.getTo();

               // Convert the start and end of frame from singular integer to umbra json
               int64_t frameStart = *reinterpret_cast<int64_t*>(&from), frameEnd = *reinterpret_cast<int64_t*>(&to);
               if (frameStart == std::numeric_limits<int64_t>::min()) {
                  operationJson["frame"]["start"] = nlohmann::json{{"mode", "unbounded"}};
               } else {
                  operationJson["frame"]["start"] = nlohmann::json{
                     {"mode", "preceding"},
                     {"value", {{"expression", "const"}, {"value", {{"type", {{"type", "bigint"}}}, {"value", -frameStart}}}}}};
               }
               if (frameEnd == std::numeric_limits<int64_t>::max()) {
                  operationJson["frame"]["end"] = nlohmann::json{{"mode", "unbounded"}};
               } else {
                  operationJson["frame"]["end"] = nlohmann::json{
                     {"mode", "following"},
                     {"value", {{"expression", "const"}, {"value", {{"type", {{"type", "bigint"}}}, {"value", frameEnd}}}}}};
               }

               // Specify applied operation, argument and new iu
               operationJson["op"]["op"] = stringifyEnum(aggrFunc.getFn());
               operationJson["op"]["arg"] = index + windowOp.getPartitionBy().size() + windowOp.getOrderBy().size();
               auto columnDef = windowOp.getComputedCols()[index].cast<mlir::tuples::ColumnDefAttr>();
               operationJson["op"]["iu"] = nlohmann::json{
                  {"iu", symbolRefToIuString(columnDef.getName())},
                  {"type", convertType(columnDef.getColumn().type)}};
               ordersJson["operations"].push_back(operationJson);
               partitionJson["orders"].push_back(ordersJson);
               result["partitions"].push_back(partitionJson);
               index++;
            }

            return result;
         })
         .Case<mlir::relalg::SortOp>([&](mlir::relalg::SortOp sortOp) {
            return convertSort(result, sortOp.getOperand().getDefiningOp(), sortOp.getSortspecs());
         })
         .Case<mlir::relalg::InnerJoinOp>([&](mlir::relalg::InnerJoinOp innerJoinOp) {
            return convertJoin(innerJoinOp, innerJoinOp.getPredicateBlock(), "inner");
         })
         .Case<mlir::relalg::SemiJoinOp>([&](mlir::relalg::SemiJoinOp semiJoinOp) {
            return convertJoin(semiJoinOp, semiJoinOp.getPredicateBlock(), "leftSemi");
         })
         .Case<mlir::relalg::AntiSemiJoinOp>([&](mlir::relalg::AntiSemiJoinOp antiSemiJoinOp) {
            return convertJoin(antiSemiJoinOp, antiSemiJoinOp.getPredicateBlock(), "leftAntiSemi");
         })
         .Case<mlir::relalg::OuterJoinOp>([&](mlir::relalg::OuterJoinOp outerJoinOp) {
            return convertJoin(outerJoinOp, outerJoinOp.getPredicateBlock(), "leftOuter");
         })
         .Case<mlir::relalg::FullOuterJoinOp>([&](mlir::relalg::FullOuterJoinOp fullOuterJoinOp) {
            return convertJoin(fullOuterJoinOp, fullOuterJoinOp.getPredicateBlock(), "outer");
         })
         .Case<mlir::relalg::SingleJoinOp>([&](mlir::relalg::SingleJoinOp singleJoinOp) {
            return convertJoin(singleJoinOp, singleJoinOp.getPredicateBlock(), "single");
         })
         .Case<mlir::relalg::CollectionJoinOp>([&](mlir::relalg::CollectionJoinOp collectionJoinOp) {
            // TODO
            result["operator"] = "collectionJoin";
            return result;
         })
         .Case<mlir::relalg::MarkJoinOp>([&](mlir::relalg::MarkJoinOp markJoinOp) {
            return convertJoin(markJoinOp, markJoinOp.getPredicateBlock(), "leftmark");
         })
         .Case<mlir::relalg::ProjectionOp>([&](mlir::relalg::ProjectionOp) {
            // TODO
            result["operator"] = "projection";
            return result;
         })
         .Case<mlir::relalg::UnionOp>([&](mlir::relalg::UnionOp unionOp) {
            result["operator"] = "setoperation";
            result["physicalOperator"] = (unionOp.getSetSemantic() == mlir::relalg::SetSemantic::all ? "unionall" : "union");
            result["operation"] = result["physicalOperator"];
            std::vector<nlohmann::json> inputs{convertOperation(unionOp.getLeft().getDefiningOp()), convertOperation(unionOp.getRight().getDefiningOp())};
            result["arguments"] = convertSetOperationArguments(inputs, unionOp.getMapping());
            return result;
         })
         .Case<mlir::relalg::IntersectOp>([&](mlir::relalg::IntersectOp intersectOp) {
            result["operator"] = "setoperation";
            result["physicalOperator"] = (intersectOp.getSetSemantic() == mlir::relalg::SetSemantic::all ? "intersectall" : "intersect");
            result["operation"] = result["physicalOperator"];
            std::vector<nlohmann::json> inputs{convertOperation(intersectOp.getLeft().getDefiningOp()), convertOperation(intersectOp.getRight().getDefiningOp())};
            result["arguments"] = convertSetOperationArguments(inputs, intersectOp.getMapping());
            return result;
         })
         .Case<mlir::relalg::NestedOp>([&](mlir::relalg::NestedOp nestedOp) {
            // TODO
            result["operator"] = "nested";
            return result;
         })
         .Case<mlir::relalg::MaterializeOp>([&](mlir::relalg::MaterializeOp materializeOp) {
            return convertOperation(materializeOp.getOperand().getDefiningOp());
         })
         .Case<mlir::subop::SetResultOp>([&](mlir::subop::SetResultOp setResultOp) {
            return convertOperation(setResultOp.getState().getDefiningOp());
         })
         .Default([&](mlir::Operation* op) {
            llvm::errs() << "The following operation could not be converted ";
            op->dump();
            assert(false && "Failed to convert operation");
            return result;
         });
   }

   public:
   ToJson(mlir::ModuleOp moduleOp, std::unordered_map<uint32_t, int64_t> tupleCounts) : moduleOp(moduleOp), tupleCounts(tupleCounts) {}
   std::string run() {
      mlir::func::FuncOp func = mlir::dyn_cast_or_null<mlir::func::FuncOp>(&moduleOp.getRegion().front().front());
      nlohmann::json plan;
      func->walk([&](mlir::Operation* operation) {
         if (mlir::isa<mlir::subop::SetResultOp>(operation)) {
            plan = convertOperation(operation);
         }
      });
      nlohmann::json planWrapper{
         {"name", "plan"}};
      planWrapper["plan"] = nlohmann::json{
         {"plan", plan},
         {"output", nlohmann::json::array({})},
         {"type", 0},
         {"query", true}};
      return to_string(planWrapper);
   }
};
class TupleCountResultProcessor : public execution::ResultProcessor {
   std::unordered_map<uint32_t, int64_t>& tupleCounts;

   public:
   TupleCountResultProcessor(std::unordered_map<uint32_t, int64_t>& tupleCounts) : tupleCounts(tupleCounts) {}
   void process(runtime::ExecutionContext* executionContext) override {
      tupleCounts = executionContext->getTupleCounts();
   }
};
void execute(std::string inputFileName, std::string databasePath, std::unordered_map<uint32_t, int64_t>& tupleCounts) {
   auto session = runtime::Session::createSession(databasePath,false);
   support::eval::init();
   auto queryExecutionConfig = execution::createQueryExecutionConfig(execution::ExecutionMode::CHEAP, false);
   queryExecutionConfig->timingProcessor = {};
   queryExecutionConfig->queryOptimizer = {};
   queryExecutionConfig->resultProcessor = std::make_unique<TupleCountResultProcessor>(tupleCounts);
   auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig), *session);
   executer->fromFile(inputFileName);
   executer->execute();
}

int main(int argc, char** argv) {
   cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

   mlir::DialectRegistry registry;
   registry.insert<mlir::relalg::RelAlgDialect>();
   registry.insert<mlir::subop::SubOperatorDialect>();
   registry.insert<mlir::tuples::TupleStreamDialect>();
   registry.insert<mlir::db::DBDialect>();
   registry.insert<mlir::dsa::DSADialect>();
   registry.insert<mlir::func::FuncDialect>();
   registry.insert<mlir::util::UtilDialect>();
   registry.insert<mlir::scf::SCFDialect>();
   registry.insert<mlir::arith::ArithDialect>();

   mlir::MLIRContext context;
   context.appendDialectRegistry(registry);
   context.allowUnregisteredDialects();
   mlir::OwningOpRef<mlir::ModuleOp> module;
   llvm::SourceMgr sourceMgr;
   mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
   if (int error = loadMLIR(context, module))
      return error;

   bool trackingTuples = false;
   module.get().walk([&](mlir::Operation* operation) {
      trackingTuples |= mlir::isa<mlir::relalg::TrackTuplesOP>(operation);
   });
   std::unordered_map<uint32_t, int64_t> tupleCounts;
   if (trackingTuples) {
      execute(inputFilename, databasePath, tupleCounts);
   }

   ToJson toJson(module.get(), tupleCounts);
   llvm::outs() << toJson.run() << "\n";
}