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
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/Dialect/SubOperator/SubOperatorDialect.h"
#include "mlir/Dialect/SubOperator/SubOperatorOps.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/Dialect/util/UtilOps.h"

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
#include <mlir/Dialect/util/UtilOps.h.inc>
#include <queue>

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
         .Case<mlir::db::DecimalType>([](mlir::db::DecimalType decimalType) {
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
         .Case<mlir::db::DateType>([](mlir::db::DateType dateType) {
            return nlohmann::json{
               // Umbra treats date as int, we have (annotated) strings
               {"baseType", "text"}};
         })
         .Case<mlir::db::NullableType>([&](mlir::db::NullableType nullableType) {
            auto res = convertDataType(nullableType.getType());
            res["nullable"] = true;
            return res;
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
   nlohmann::json innerExpression(const std::vector<std::string> strings, std::vector<nlohmann::json> operands) {
      nlohmann::json result{
         {"type", "expression_inner"},
         {"strings", strings},
         {"subExpressions", operands}};
      return result;
   }

   nlohmann::json innerExpression(const std::vector<std::string> strings, mlir::ValueRange operands, std::function<nlohmann::json(mlir::BlockArgument)> resolveBlockArgs) {
      nlohmann::json result{
         {"type", "expression_inner"},
         {"strings", strings},
         {"subExpressions", nlohmann::json::array()}};
      for (auto operand : operands) {
         result["subExpressions"].push_back(convertExpression(operand, resolveBlockArgs));
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
   nlohmann::json convertExpression(mlir::Operation* operation, std::function<nlohmann::json(mlir::BlockArgument)> resolveBlockArgs = nullptr) {
      return llvm::TypeSwitch<mlir::Operation*, nlohmann::json>(operation)
         .Case<mlir::tuples::ReturnOp>([&](mlir::tuples::ReturnOp returnOp) {
            if (returnOp->getOperands().size()) return convertExpression(returnOp.getOperand(0), resolveBlockArgs);
            return nlohmann::json{};
         })
         .Case<mlir::arith::ExtUIOp>([&](mlir::arith::ExtUIOp extUiOp) { return innerExpression({"cast(", ")"}, extUiOp.getIn(), resolveBlockArgs); })
         .Case<mlir::arith::ExtSIOp>([&](mlir::arith::ExtSIOp extUiOp) { return innerExpression({"cast(", ")"}, extUiOp.getIn(), resolveBlockArgs); })
         .Case<mlir::arith::SelectOp>([&](mlir::arith::SelectOp selectOp) {
            return innerExpression({"", " ? ", " : ", ""}, selectOp.getOperands(), resolveBlockArgs);
         })
         .Case<mlir::db::NotOp>([&](mlir::db::NotOp notOp) {
            return innerExpression({"not ", ""}, notOp.getVal(), resolveBlockArgs);
         })
         .Case<mlir::db::NullOp>([&](mlir::db::NullOp isNullOp) {
            return nlohmann::json{
               {"type", "expression_leaf"},
               {"leaf_type", "null"}};
         })
         .Case<mlir::db::IsNullOp>([&](mlir::db::IsNullOp isNullOp) {
            return innerExpression({"", " is null"}, isNullOp.getOperand(), resolveBlockArgs);
         })
         .Case<mlir::db::AsNullableOp>([&](mlir::db::AsNullableOp asNullableOp) {
            return convertExpression(asNullableOp->getOperand(0), resolveBlockArgs);
         })
         .Case<mlir::db::BetweenOp>([&](mlir::db::BetweenOp betweenOp) {
            return innerExpression({"", " between ", " and ", ""}, betweenOp.getOperands(), resolveBlockArgs);
         })
         .Case<mlir::db::OneOfOp>([&](mlir::db::OneOfOp oneOfOp) {
            std::vector<std::string> seperators = {"", " in ["};

            for (size_t i = 1; i < oneOfOp.getVals().size(); i++) {
               seperators.push_back(", ");
            }
            seperators.push_back("]");
            return innerExpression(seperators, oneOfOp.getOperands(), resolveBlockArgs);
         })
         .Case<mlir::db::AndOp>([&](mlir::db::AndOp andOp) {
            std::vector<std::string> seperators = {""};
            for (size_t i = 1; i < andOp.getOperands().size(); i++) {
               seperators.push_back(" and ");
            }
            seperators.push_back("");
            return innerExpression(seperators, andOp.getOperands(), resolveBlockArgs);
         })
         .Case<mlir::db::OrOp>([&](mlir::db::OrOp orOp) {
            std::vector<std::string> seperators = {"("};
            for (size_t i = 1; i < orOp.getOperands().size(); i++) {
               seperators.push_back(" or ");
            }
            seperators.push_back(")");
            return innerExpression(seperators, orOp.getOperands(), resolveBlockArgs);
         })
         .Case<mlir::db::CmpOp>([&](mlir::db::CmpOp cmpOp) {
            return innerExpression({"", convertCmpPredicate(cmpOp.getPredicate()), ""}, cmpOp.getOperands(), resolveBlockArgs);
         })
         .Case<mlir::db::ConstantOp>([&](mlir::db::ConstantOp constantOp) {
            return convertConstant(constantOp.getValue(), constantOp.getType());
         })
         .Case<mlir::tuples::GetColumnOp>([&](mlir::tuples::GetColumnOp getColumnOp) {
            return columnToJSON(getColumnOp.getAttr());
         })
         .Case<mlir::db::AddOp>([&](mlir::db::AddOp addOp) {
            return innerExpression({"", " + ", ""}, addOp.getOperands(), resolveBlockArgs);
         })
         .Case<mlir::db::Hash>([&](mlir::db::Hash hashOp) {
            return innerExpression({"hash(", ")"}, hashOp.getVal(), resolveBlockArgs);
         })
         .Case<mlir::util::PackOp>([&](mlir::util::PackOp packOp) {
            std::vector<std::string> seperators = {"pack("};
            for (size_t i = 1; i < packOp.getOperands().size(); i++) {
               seperators.push_back(", ");
            }
            seperators.push_back(")");
            return innerExpression(seperators, packOp.getOperands(), resolveBlockArgs);
         })
         .Case<mlir::util::InvalidRefOp>([&](mlir::util::InvalidRefOp) {
            return innerExpression({"nullptr"}, {});
         })
         .Case<mlir::db::SubOp>([&](mlir::db::SubOp subOp) {
            return innerExpression({"", " + ", ""}, subOp.getOperands(), resolveBlockArgs);
         })
         .Case<mlir::db::MulOp>([&](mlir::db::MulOp mulOp) {
            return innerExpression({"", " * ", ""}, mulOp.getOperands(), resolveBlockArgs);
         })
         .Case<mlir::db::DivOp>([&](mlir::db::DivOp divOp) {
            return innerExpression({"", " / ", ""}, divOp.getOperands(), resolveBlockArgs);
         })
         .Case<mlir::db::DeriveTruth>([&](mlir::db::DeriveTruth deriveTruth) {
            return convertExpression(deriveTruth.getOperand(), resolveBlockArgs);
         })
         .Case<mlir::db::CastOp>([&](mlir::db::CastOp castOp) {
            return innerExpression({"cast(", ")"}, castOp.getVal(), resolveBlockArgs);
         })
         .Case<mlir::db::RuntimeCall>([&](mlir::db::RuntimeCall runtimeCall) {
            std::vector<std::string> seperators = {runtimeCall.getFn().str() + "("};
            for (size_t i = 1; i < runtimeCall.getOperands().size(); i++) {
               seperators.push_back(", ");
            }
            seperators.push_back(")");
            return innerExpression(seperators, runtimeCall.getOperands(), resolveBlockArgs);
         })
         .Case<mlir::arith::ConstantOp>([&](mlir::arith::ConstantOp constantOp) {
            return convertConstant(constantOp.getValue(), constantOp.getType());
         })
         .Case<mlir::scf::YieldOp>([&](mlir::scf::YieldOp yieldOp) {
            return convertExpression(yieldOp.getOperand(0), resolveBlockArgs);
         })
         .Case<mlir::scf::IfOp>([&](mlir::scf::IfOp ifOp) {
            if (ifOp.getNumResults() == 1) {
               auto condition = convertExpression(ifOp.getOperand(), resolveBlockArgs);
               auto thenExpr = convertExpression(ifOp.getThenRegion().getBlocks().begin()->getTerminator()->getOperand(0), resolveBlockArgs);
               auto elseExpr = convertExpression(ifOp.getElseRegion().getBlocks().begin()->getTerminator()->getOperand(0), resolveBlockArgs);
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
   nlohmann::json convertExpression(mlir::Value v, std::function<nlohmann::json(mlir::BlockArgument)> resolveBlockArgs = nullptr) {
      if (auto m = mlir::dyn_cast_or_null<mlir::OpResult>(v)) {
         return convertExpression(v.getDefiningOp(), resolveBlockArgs);
      } else if (auto m = mlir::dyn_cast_or_null<mlir::BlockArgument>(v)) {
         return resolveBlockArgs(m);
      } else {
         return nlohmann::json{{"type", "expression_leaf"}, {"leaf_type", "unknown"}};
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
   nlohmann::json getOperandReference(mlir::Value val) {
      if (auto* op = val.getDefiningOp()) {
         return nlohmann::json{{"producing", getOperationReference(op)}, {"resnr", mlir::cast<mlir::OpResult>(val).getResultNumber()}};
      } else if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(val)) {
         return nlohmann::json{{"producing", "parent"}, {"parent", getOperationReference(blockArg.getOwner()->getParentOp())}, {"argnr", blockArg.getArgNumber()}};
      } else {
         assert(false);
      }
   }

   nlohmann::json columnToJSON(mlir::tuples::ColumnDefAttr columnDefAttr) {
      return nlohmann::json{
         {"datatype", convertDataType(columnDefAttr.getColumn().type)},
         {"type", "expression_leaf"},
         {"leaf_type", "column"},
         {"displayName", symbolRefToIuString(columnDefAttr.getName())}};
   }
   nlohmann::json columnToJSON(mlir::tuples::ColumnRefAttr columnRefAttr) {
      return nlohmann::json{
         {"datatype", convertDataType(columnRefAttr.getColumn().type)},
         {"type", "expression_leaf"},
         {"leaf_type", "column"},
         {"displayName", symbolRefToIuString(columnRefAttr.getName())}};
   }
   nlohmann::json serializeDefMapping(mlir::DictionaryAttr dictAttr) {
      auto result = nlohmann::json::array();
      for (auto x : dictAttr) {
         result.push_back(nlohmann::json{{"member", x.getName().str()}, {"column", columnToJSON(mlir::cast<mlir::tuples::ColumnDefAttr>(x.getValue()))}});
      }
      return result;
   }
   nlohmann::json serializeRefMapping(mlir::DictionaryAttr dictAttr) {
      auto result = nlohmann::json::array();
      for (auto x : dictAttr) {
         result.push_back(nlohmann::json{{"member", x.getName().str()}, {"column", columnToJSON(mlir::cast<mlir::tuples::ColumnRefAttr>(x.getValue()))}});
      }
      return result;
   }
   nlohmann::json convertOperation(mlir::Operation* op) {
      nlohmann::json result;
      result["ref"] = getOperationReference(op);
      result["type"] = "suboperator";
      result["consuming"] = nlohmann::json::array();
      result["accesses"] = nlohmann::json::array();
      for (auto operand : op->getOperands()) {
         if (mlir::isa<mlir::tuples::TupleStreamType>(operand.getType())) {
            result["consuming"].push_back(getOperationReference(operand.getDefiningOp()));
         };
      }

      return llvm::TypeSwitch<mlir::Operation*, nlohmann::json>(op)
         .Case<mlir::subop::ExecutionStepOp>([&](mlir::subop::ExecutionStepOp executionGroupOp) {
            result["type"] = "execution_step";
            //todo: handle arguments etc
            result["subops"] = nlohmann::json::array();
            for (auto& op : executionGroupOp.getSubOps().front()) {
               if (!mlir::isa_and_nonnull<mlir::subop::ExecutionStepReturnOp>(&op)) {
                  result["subops"].push_back(convertOperation(&op));
               }
            }
            result["inputs"] = nlohmann::json::array();
            for (auto [i, a] : llvm::zip(op->getOperands(), executionGroupOp.getSubOps().getArguments())) {
               result["inputs"].push_back(nlohmann::json{{"input", getOperandReference(i)}, {"argument", getOperandReference(a)}});
            }
            result["results"] = nlohmann::json::array();
            for (auto t : executionGroupOp.getSubOps().front().getTerminator()->getOperands()) {
               result["results"].push_back(getOperandReference(t));
            }
            return result;
         })

         .Case<mlir::subop::CreateThreadLocalOp>([&](mlir::subop::CreateThreadLocalOp op) {
            result["subop"] = "create_thread_local";
            return result;
         })
         .Case<mlir::subop::GetExternalOp>([&](mlir::subop::GetExternalOp op) {
            result["subop"] = "get_external";
            result["meta"] = nlohmann::json::parse(op.getDescr().str());
            return result;
         })
         .Case<mlir::subop::CreateThreadLocalOp>([&](mlir::subop::CreateThreadLocalOp createThreadLocalOp) {
            result["subop"] = "create_thread_local";
            result["resultType"] = convertDataType(createThreadLocalOp.getRes().getType());
            return result;
         })
         .Case<mlir::subop::MergeOp>([&](mlir::subop::MergeOp op) {
            result["subop"] = "merge";
            result["accesses"].push_back(getOperandReference(op.getThreadLocal()));
            return result;
         })
         //TODO .Case<mlir::subop::LockOp>([&](mlir::subop::LockOp lockOp) {})
         .Case<mlir::subop::ScanOp>([&](mlir::subop::ScanOp scanOp) {
            result["subop"] = "scan";
            result["mapping"] = serializeDefMapping(scanOp.getMapping());
            result["accesses"].push_back(getOperandReference(scanOp.getState()));
            return result;
         })
         .Case<mlir::subop::ScanListOp>([&](mlir::subop::ScanListOp scanListOp) {
            result["subop"] = "scan_list";
            result["elem"] = columnToJSON(scanListOp.getElem());
            result["accesses"].push_back(getOperandReference(scanListOp.getList()));
            return result;
         })
         .Case<mlir::subop::ScanRefsOp>([&](mlir::subop::ScanRefsOp op) {
            result["subop"] = "scan_ref";
            result["reference"] = columnToJSON(op.getRef());
            result["accesses"].push_back(getOperandReference(op.getState()));
            return result;
         })
         .Case<mlir::subop::GenerateOp>([&](mlir::subop::GenerateOp generateOp) {
            result["subop"] = "generate";
            result["generated"] = nlohmann::json::array();
            for (auto column : generateOp.getGeneratedColumns()) {
               result["generated"].push_back(columnToJSON(mlir::cast<mlir::tuples::ColumnDefAttr>(column)));
            }
            return result;
         })
         .Case<mlir::subop::UnionOp>([&](mlir::subop::UnionOp unionOp) {
            result["subop"] = "union";
            return result;
         })
         .Case<mlir::subop::MapOp>([&](mlir::subop::MapOp mapOp) {
            result["subop"] = "map";
            result["computed"] = nlohmann::json::array();
            for (auto [column, computed] : llvm::zip(mapOp.getComputedCols(), mapOp.getFn().front().getTerminator()->getOperands())) {
               auto columnDefAttr = mlir::cast<mlir::tuples::ColumnDefAttr>(column);
               result["computed"].push_back({{"computed", columnToJSON(columnDefAttr)}, {"expression", convertExpression(computed, [&](mlir::BlockArgument ba) {
                                                                                            if (ba.getOwner()->getParentOp() == mapOp.getOperation()) {
                                                                                               auto accessedTuple = mlir::cast<mlir::tuples::ColumnRefAttr>(mapOp.getInputCols()[ba.getArgNumber()]);
                                                                                               return columnToJSON(accessedTuple);
                                                                                            } else {
                                                                                               return nlohmann::json{{"type", "expression_leaf"}, {"leaf_type", "unknown"}};
                                                                                            }
                                                                                         })}});
            }
            return result;
         })
         .Case<mlir::subop::FilterOp>([&](mlir::subop::FilterOp filterOp) {
            result["subop"] = "filter";
            result["semantic"] = filterOp.getFilterSemantic() == mlir::subop::FilterSemantic::all_true ? "all_true" : "all_false";
            result["columns"] = nlohmann::json::array();
            for (auto x : filterOp.getConditions()) {
               result["columns"].push_back(columnToJSON(mlir::cast<mlir::tuples::ColumnRefAttr>(x)));
            }
            return result;
         })
         .Case<mlir::subop::RenamingOp>([&](mlir::subop::RenamingOp renamingOp) {
            result["subop"] = "renaming";
            result["renamed"] = nlohmann::json::array();
            for (auto column : renamingOp.getColumns()) {
               auto newDef = mlir::cast<mlir::tuples::ColumnDefAttr>(column);
               auto old = mlir::cast<mlir::tuples::ColumnRefAttr>(mlir::cast<mlir::ArrayAttr>(newDef.getFromExisting())[0]);
               result["renamed"].push_back(nlohmann::json{{"new", columnToJSON(newDef)}, {"old", columnToJSON(old)}});
            }
            return result;
         })
         .Case<mlir::subop::NestedMapOp>([&](mlir::subop::NestedMapOp nestedMapOp) {
            result["subop"] = "nested_map";
            result["inputs"] = nlohmann::json::array();
            result["implicitInputs"] = nlohmann::json::array();
            result["implicitEdges"] = nlohmann::json::array();
            for (auto [col, a] : llvm::zip(nestedMapOp.getParameters(), nestedMapOp.getBody()->getArguments().drop_front())) {
               result["inputs"].push_back(nlohmann::json{{"inputCol", columnToJSON(mlir::cast<mlir::tuples::ColumnRefAttr>(col))}, {"argument", getOperandReference(a)}});
            }
            if (nestedMapOp.getBody()->begin() != nestedMapOp.getBody()->end()) {
               mlir::Operation* firstOp = &*nestedMapOp.getBody()->begin();
               if (auto nestedExecutionGroup = mlir::dyn_cast_or_null<mlir::subop::NestedExecutionGroupOp>(firstOp)) {
                  for (auto [i, a] : llvm::zip(nestedExecutionGroup->getOperands(), nestedExecutionGroup.getSubOps().getArguments())) {
                     if (!nestedMapOp.getRegion().isAncestor(i.getParentRegion())) {
                        result["implicitInputs"].push_back(getOperandReference(i));
                     }
                     result["implicitEdges"].push_back(nlohmann::json{{"from", getOperandReference(i)}, {"to", getOperandReference(a)}});
                  }

                  result["subops"] = nlohmann::json::array();
                  for (auto& op : nestedExecutionGroup.getSubOps().front()) {
                     if (!mlir::isa_and_nonnull<mlir::subop::ExecutionStepReturnOp>(&op)) {
                        result["subops"].push_back(convertOperation(&op));
                     }
                  }
               }
            }
            return result;
         })

         .Case<mlir::subop::CreateFrom>([&](mlir::subop::CreateFrom op) {
            result["subop"] = "create_from";
            return result;
         })
         .Case<mlir::subop::CreateSimpleStateOp>([&](mlir::subop::CreateSimpleStateOp op) {
            result["subop"] = "create_simple_state";
            return result;
         })
         .Case<mlir::subop::CreateHeapOp>([&](mlir::subop::CreateHeapOp op) {
            result["subop"] = "create_heap";
            return result;
         })
         .Case<mlir::subop::GenericCreateOp>([&](mlir::subop::GenericCreateOp op) {
            result["subop"] = "generic_create";
            return result;
         })
         .Case<mlir::subop::CreateArrayOp>([&](mlir::subop::CreateArrayOp op) {
            result["subop"] = "create_array";
            return result;
         })
         .Case<mlir::subop::CreateSortedViewOp>([&](mlir::subop::CreateSortedViewOp op) {
            result["subop"] = "create_sorted_view";
            return result;
         })
         .Case<mlir::subop::CreateHashIndexedView>([&](mlir::subop::CreateHashIndexedView op) {
            result["subop"] = "create_hash_indexed_view";
            return result;
         })
         .Case<mlir::subop::CreateContinuousView>([&](mlir::subop::CreateContinuousView op) {
            result["subop"] = "create_continuous_view";
            return result;
         })
         .Case<mlir::subop::CreateSegmentTreeView>([&](mlir::subop::CreateSegmentTreeView op) {
            result["subop"] = "create_segment_tree_view";
            return result;
         })

         .Case<mlir::subop::MaterializeOp>([&](mlir::subop::MaterializeOp op) {
            result["subop"] = "materialize";
            result["accesses"].push_back(getOperandReference(op.getState()));
            result["mapping"] = serializeRefMapping(op.getMapping());
            return result;
         })
         .Case<mlir::subop::LookupOrInsertOp>([&](mlir::subop::LookupOrInsertOp op) {
            result["subop"] = "lookup_or_insert";
            result["accesses"].push_back(getOperandReference(op.getState()));
            result["reference"] = columnToJSON(op.getRef());
            return result;
         })
         .Case<mlir::subop::InsertOp>([&](mlir::subop::InsertOp op) {
            result["subop"] = "insert";
            result["accesses"].push_back(getOperandReference(op.getState()));
            result["mapping"] = serializeRefMapping(op.getMapping());
            return result;
         })
         .Case<mlir::subop::LookupOp>([&](mlir::subop::LookupOp op) {
            result["subop"] = "lookup";
            result["accesses"].push_back(getOperandReference(op.getState()));
            result["reference"] = columnToJSON(op.getRef());
            return result;
         })
         .Case<mlir::subop::GetBeginReferenceOp>([&](mlir::subop::GetBeginReferenceOp op) {
            result["subop"] = "get_begin_reference";
            result["accesses"].push_back(getOperandReference(op.getState()));
            result["reference"] = columnToJSON(op.getRef());
            return result;
         })
         .Case<mlir::subop::GetEndReferenceOp>([&](mlir::subop::GetEndReferenceOp op) {
            result["subop"] = "get_end_reference";
            result["accesses"].push_back(getOperandReference(op.getState()));
            result["reference"] = columnToJSON(op.getRef());
            return result;
         })
         .Case<mlir::subop::EntriesBetweenOp>([&](mlir::subop::EntriesBetweenOp op) {
            result["subop"] = "entries_between";
            result["leftRef"] = columnToJSON(op.getLeftRef());
            result["rightRef"] = columnToJSON(op.getRightRef());
            result["between"] = columnToJSON(op.getBetween());
            return result;
         })
         .Case<mlir::subop::OffsetReferenceBy>([&](mlir::subop::OffsetReferenceBy op) {
            result["subop"] = "offset_reference_by";
            result["reference"] = columnToJSON(op.getRef());
            result["offset"] = columnToJSON(op.getIdx());
            result["newRef"] = columnToJSON(op.getNewRef());
            return result;
         })
         .Case<mlir::subop::UnwrapOptionalRefOp>([&](mlir::subop::UnwrapOptionalRefOp op) {
            result["subop"] = "unwrap_optional_ref";
            result["reference"] = columnToJSON(op.getRef());
            result["optionalRef"] = columnToJSON(op.getOptionalRef());
            return result;
         })

         .Case<mlir::subop::GatherOp>([&](mlir::subop::GatherOp op) {
            result["subop"] = "gather";
            result["reference"] = columnToJSON(op.getRef());
            result["mapping"] = serializeDefMapping(op.getMapping());
            return result;
         })
         .Case<mlir::subop::ScatterOp>([&](mlir::subop::ScatterOp op) {
            result["subop"] = "scatter";
            result["reference"] = columnToJSON(op.getRef());
            result["mapping"] = serializeRefMapping(op.getMapping());
            return result;
         })
         .Case<mlir::subop::ReduceOp>([&](mlir::subop::ReduceOp op) {
            result["subop"] = "reduce";
            result["reference"] = columnToJSON(op.getRef());
            result["updated"] = nlohmann::json::array();
           for (auto [member, computed] : llvm::zip(op.getMembers(), op.getRegion().front().getTerminator()->getOperands())) {

              result["updated"].push_back({{"member", mlir::cast<mlir::StringAttr>(member).str()}, {"expression", convertExpression(computed, [&](mlir::BlockArgument ba) {
                                                                                           if (ba.getOwner()->getParentOp() == op.getOperation()) {
                                                                                              auto argNr=ba.getArgNumber();
                                                                                              if(argNr<op.getColumns().size()) {
                                                                                                 auto accessedTuple = mlir::cast<mlir::tuples::ColumnRefAttr>(op.getColumns()[argNr]);;
                                                                                                 return columnToJSON(accessedTuple);
                                                                                              }else {
                                                                                                 return nlohmann::json{{"type", "expression_leaf"}, {"leaf_type", "member"}, {"member", mlir::cast<mlir::StringAttr>(op.getMembers()[argNr-op.getColumns().size()]).str()}};
                                                                                              }
                                                                                           } else {
                                                                                              return nlohmann::json{{"type", "expression_leaf"}, {"leaf_type", "unknown"}};
                                                                                           }
                                                                                        })}});
           }
            return result;
         })
         //TODO: Loop
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
         if (auto setResultOp = mlir::dyn_cast_or_null<mlir::subop::SetResultOp>(operation)) {
            if (auto relalgQuery = mlir::dyn_cast_or_null<mlir::subop::ExecutionGroupOp>(setResultOp.getState().getDefiningOp())) {
               for (auto& op : relalgQuery.getSubOps().front()) {
                  if (!mlir::isa<mlir::subop::ExecutionGroupReturnOp>(&op)) {
                     plan.push_back(convertOperation(&op));
                  }
               }
            }
         }
      });
      return to_string(plan);
   }
};
} // namespace

int main(int argc, char** argv) {
   cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");
   mlir::MLIRContext context;
   execution::initializeContext(context);
   mlir::OwningOpRef<mlir::ModuleOp> module;
   llvm::SourceMgr sourceMgr;
   mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
   if (int error = loadMLIR(context, module))
      return error;
   std::unordered_map<uint32_t, int64_t> tupleCounts;

   ToJson toJson(module.get(), tupleCounts);
   llvm::outs() << toJson.run() << "\n";
}