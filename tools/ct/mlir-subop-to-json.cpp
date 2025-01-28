#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "lingodb/execution/Execution.h"
#include "lingodb/execution/ResultProcessing.h"

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

#include <llvm/IR/Type.h>

namespace cl = llvm::cl;
namespace {
using namespace lingodb::compiler::dialect;
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
   nlohmann::json stateType(mlir::Type type) {
      return llvm::TypeSwitch<mlir::Type, nlohmann::json>(type)
         .Case<subop::TableType>([&](subop::TableType resultTableType) {
            return "Table";
         })
         .Case<subop::ResultTableType>([&](subop::ResultTableType resultTableType) {
            return "ResultTable";
         })
         .Case<subop::HashIndexedViewType>([&](subop::HashIndexedViewType resultTableType) {
            return "HashIndexedView";
         })
         .Case<subop::HashMapType>([&](subop::HashMapType resultTableType) {
            return "HashMap";
         })
         .Case<subop::HeapType>([&](subop::HeapType resultTableType) {
            return "Heap";
         })
         .Case<subop::ArrayType>([&](subop::ArrayType arrayType) {
            return "Array";
         })
         .Case<subop::BufferType>([&](subop::BufferType bufferType) {
            return "Buffer";
         })
         .Case<subop::PreAggrHtType>([&](subop::PreAggrHtType preAggrHtType) {
            return "PreAggrHt";
         })
         .Case<subop::PreAggrHtFragmentType>([&](subop::PreAggrHtFragmentType preAggrHtFragmentType) {
            return "PreAggrHt";
         })
         .Case<subop::ContinuousViewType>([&](subop::ContinuousViewType continuousViewType) {
            return "ContinuousView";
         })
         .Case<subop::SortedViewType>([&](subop::SortedViewType sortedViewType) {
            return "SortedView";
         })
         .Case<subop::SimpleStateType>([&](subop::SimpleStateType resultTableType) {
            return "SimpleState";
         })
         .Default([](mlir::Type type) {
            llvm::errs() << "state type could not be converted ";
            type.dump();
            llvm::errs() << "\n";
            return nlohmann::json{};
         });
   }

   // Convert mlir Type to umbra json type
   std::string convertDataType(mlir::Type type) {
      auto membersToString = [this](subop::StateMembersAttr attr) {
         std::string result = "[";
         for (auto [member, type] : llvm::zip(attr.getNames(), attr.getTypes())) {
            result += mlir::cast<mlir::StringAttr>(member).str() + ":" + convertDataType(mlir::cast<mlir::TypeAttr>(type).getValue()) + ",";
         }
         result += "]";
         return result;
      };
      return llvm::TypeSwitch<mlir::Type, nlohmann::json>(type)
         .Case<db::DecimalType>([](db::DecimalType decimalType) {
            return "decimal(" + std::to_string(decimalType.getP()) + "," + std::to_string(decimalType.getS()) + ")";
         })
         .Case<mlir::IndexType>([](mlir::IndexType indexType) {
            return "index";
         })
         .Case<mlir::IntegerType>([](mlir::IntegerType integerType) {
            return (integerType.isSigned() ? "int" : "uint") + std::to_string(integerType.getWidth());
         })
         .Case<db::DateType>([](db::DateType dateType) {
            return "date";
         })
         .Case<db::NullableType>([&](db::NullableType nullableType) {
            auto res = convertDataType(nullableType.getType());
            return "nullable(" + res + ")";
         })
         .Case<db::StringType>([&](db::StringType stringType) {
            return "str";
         })
         .Case<db::CharType>([](db::CharType charType) {
            return "char" + std::to_string(charType.getBytes());
         })
         .Case<mlir::FloatType>([](mlir::FloatType floatType) {
            return "float" + std::to_string(floatType.getWidth());
         })
         .Case<subop::BufferType>([&](subop::BufferType bufferType) {
            return "Buffer" + membersToString(bufferType.getMembers());
         })
         .Case<subop::ResultTableType>([&](subop::ResultTableType resultTableType) {
            return "ResultTable" + membersToString(resultTableType.getMembers());
         })
         .Case<subop::EntryRefType>([&](subop::EntryRefType entryRefType) {
            return "EntryRef " + convertDataType(entryRefType.getState());
         })
         .Case<subop::SortedViewType>([&](subop::SortedViewType sortedViewType) {
            return "SortedView " + convertDataType(sortedViewType.getBasedOn());
         })
         .Case<subop::TableType>([&](subop::TableType resultTableType) {
            return "Table" + membersToString(resultTableType.getMembers());
         })
         .Default([](mlir::Type type) {
            llvm::errs() << "type could not be converted ";
            type.dump();
            llvm::errs() << "\n";
            return "?";
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

   nlohmann::json innerExpression(const std::vector<std::string> strings, mlir::ValueRange operands, std::function<nlohmann::json(mlir::BlockArgument, bool)> resolveBlockArgs) {
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
   nlohmann::json convertExpression(mlir::Operation* operation, std::function<nlohmann::json(mlir::BlockArgument, bool)> resolveBlockArgs = nullptr) {
      return llvm::TypeSwitch<mlir::Operation*, nlohmann::json>(operation)
         .Case<tuples::ReturnOp>([&](tuples::ReturnOp returnOp) {
            if (returnOp->getOperands().size()) return convertExpression(returnOp.getOperand(0), resolveBlockArgs);
            return nlohmann::json{};
         })
         .Case<mlir::arith::ExtUIOp>([&](mlir::arith::ExtUIOp extUiOp) { return innerExpression({"cast(", ")"}, extUiOp.getIn(), resolveBlockArgs); })
         .Case<mlir::arith::ExtSIOp>([&](mlir::arith::ExtSIOp extUiOp) { return innerExpression({"cast(", ")"}, extUiOp.getIn(), resolveBlockArgs); })
         .Case<mlir::arith::SelectOp>([&](mlir::arith::SelectOp selectOp) {
            return innerExpression({"", " ? ", " : ", ""}, selectOp.getOperands(), resolveBlockArgs);
         })
         .Case<db::NotOp>([&](db::NotOp notOp) {
            return innerExpression({"not ", ""}, notOp.getVal(), resolveBlockArgs);
         })
         .Case<db::NullOp>([&](db::NullOp isNullOp) {
            return nlohmann::json{
               {"type", "expression_leaf"},
               {"leaf_type", "null"}};
         })
         .Case<db::IsNullOp>([&](db::IsNullOp isNullOp) {
            return innerExpression({"", " is null"}, isNullOp.getOperand(), resolveBlockArgs);
         })
         .Case<db::AsNullableOp>([&](db::AsNullableOp asNullableOp) {
            return convertExpression(asNullableOp->getOperand(0), resolveBlockArgs);
         })
         .Case<db::BetweenOp>([&](db::BetweenOp betweenOp) {
            return innerExpression({"", " between ", " and ", ""}, betweenOp.getOperands(), resolveBlockArgs);
         })
         .Case<db::OneOfOp>([&](db::OneOfOp oneOfOp) {
            std::vector<std::string> seperators = {"", " in ["};

            for (size_t i = 1; i < oneOfOp.getVals().size(); i++) {
               seperators.push_back(", ");
            }
            seperators.push_back("]");
            return innerExpression(seperators, oneOfOp.getOperands(), resolveBlockArgs);
         })
         .Case<db::AndOp>([&](db::AndOp andOp) {
            std::vector<std::string> seperators = {""};
            for (size_t i = 1; i < andOp.getOperands().size(); i++) {
               seperators.push_back(" and ");
            }
            seperators.push_back("");
            return innerExpression(seperators, andOp.getOperands(), resolveBlockArgs);
         })
         .Case<db::OrOp>([&](db::OrOp orOp) {
            std::vector<std::string> seperators = {"("};
            for (size_t i = 1; i < orOp.getOperands().size(); i++) {
               seperators.push_back(" or ");
            }
            seperators.push_back(")");
            return innerExpression(seperators, orOp.getOperands(), resolveBlockArgs);
         })
         .Case<db::CmpOp>([&](db::CmpOp cmpOp) {
            return innerExpression({"", convertCmpPredicate(cmpOp.getPredicate()), ""}, cmpOp.getOperands(), resolveBlockArgs);
         })
         .Case<db::ConstantOp>([&](db::ConstantOp constantOp) {
            return convertConstant(constantOp.getValue(), constantOp.getType());
         })
         .Case<tuples::GetColumnOp>([&](tuples::GetColumnOp getColumnOp) {
            return columnToJSON(getColumnOp.getAttr());
         })
         .Case<db::AddOp>([&](db::AddOp addOp) {
            return innerExpression({"", " + ", ""}, addOp.getOperands(), resolveBlockArgs);
         })
         .Case<db::Hash>([&](db::Hash hashOp) {
            return innerExpression({"hash(", ")"}, hashOp.getVal(), resolveBlockArgs);
         })
         .Case<util::PackOp>([&](util::PackOp packOp) {
            std::vector<std::string> seperators = {"pack("};
            for (size_t i = 1; i < packOp.getOperands().size(); i++) {
               seperators.push_back(", ");
            }
            seperators.push_back(")");
            return innerExpression(seperators, packOp.getOperands(), resolveBlockArgs);
         })
         .Case<util::InvalidRefOp>([&](util::InvalidRefOp) {
            return innerExpression({"nullptr"}, {});
         })
         .Case<db::SubOp>([&](db::SubOp subOp) {
            return innerExpression({"", " + ", ""}, subOp.getOperands(), resolveBlockArgs);
         })
         .Case<db::MulOp>([&](db::MulOp mulOp) {
            return innerExpression({"", " * ", ""}, mulOp.getOperands(), resolveBlockArgs);
         })
         .Case<db::DivOp>([&](db::DivOp divOp) {
            return innerExpression({"", " / ", ""}, divOp.getOperands(), resolveBlockArgs);
         })
         .Case<db::DeriveTruth>([&](db::DeriveTruth deriveTruth) {
            return convertExpression(deriveTruth.getOperand(), resolveBlockArgs);
         })
         .Case<db::CastOp>([&](db::CastOp castOp) {
            return innerExpression({"cast(", ")"}, castOp.getVal(), resolveBlockArgs);
         })
         .Case<db::RuntimeCall>([&](db::RuntimeCall runtimeCall) {
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
   nlohmann::json convertExpression(mlir::Value v, std::function<nlohmann::json(mlir::BlockArgument, bool)> resolveBlockArgs = nullptr) {
      if (auto m = mlir::dyn_cast_or_null<mlir::OpResult>(v)) {
         return convertExpression(v.getDefiningOp(), resolveBlockArgs);
      } else if (auto m = mlir::dyn_cast_or_null<mlir::BlockArgument>(v)) {
         return resolveBlockArgs(m, true);
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
   nlohmann::json getOperandReference(mlir::Value val, std::function<nlohmann::json(mlir::BlockArgument, bool)> resolveBlockArgs) {
      if (auto* op = val.getDefiningOp()) {
         return nlohmann::json{{"type", "node"}, {"ref", getOperationReference(op)}, {"resnr", mlir::cast<mlir::OpResult>(val).getResultNumber()}};
      } else if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(val)) {
         if (resolveBlockArgs) {
            return resolveBlockArgs(blockArg, false);
         }
         assert(false);
      } else {
         assert(false);
      }
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
   nlohmann::json serializeDefMapping(mlir::DictionaryAttr dictAttr) {
      auto result = nlohmann::json::array();
      for (auto x : dictAttr) {
         result.push_back(nlohmann::json{{"member", x.getName().str()}, {"column", columnToJSON(mlir::cast<tuples::ColumnDefAttr>(x.getValue()))}});
      }
      return result;
   }
   nlohmann::json serializeRefMapping(mlir::DictionaryAttr dictAttr) {
      auto result = nlohmann::json::array();
      for (auto x : dictAttr) {
         result.push_back(nlohmann::json{{"member", x.getName().str()}, {"column", columnToJSON(mlir::cast<tuples::ColumnRefAttr>(x.getValue()))}});
      }
      return result;
   }
   nlohmann::json convertOperation(mlir::Operation* op, std::function<nlohmann::json(mlir::BlockArgument, bool)> resolveBlockArgs = nullptr) {
      nlohmann::json result;
      result["ref"] = getOperationReference(op);
      result["type"] = "suboperator";
      result["outerEdges"] = nlohmann::json::array();
      result["accesses"] = nlohmann::json::array();
      for (auto operand : op->getOperands()) {
         if (mlir::isa<tuples::TupleStreamType>(operand.getType())) {
            result["outerEdges"].push_back({{"type", "stream"}, {"input", getOperandReference(operand, resolveBlockArgs)}, {"output", {{"type", "node"}, {"ref", getOperationReference(op)}}}});
         };
      }

      return llvm::TypeSwitch<mlir::Operation*, nlohmann::json>(op)
         .Case<subop::ExecutionStepOp>([&](subop::ExecutionStepOp executionStepOp) {
            result["type"] = "execution_step";
            //todo: handle arguments etc
            result["subops"] = nlohmann::json::array();
            result["inputs"] = nlohmann::json::array();
            llvm::DenseMap<mlir::BlockArgument, size_t> blockArgToIndex;

            for (auto& op : executionStepOp.getSubOps().front()) {
               if (!mlir::isa_and_nonnull<subop::ExecutionStepReturnOp>(&op)) {
                  auto r = convertOperation(&op, [&](mlir::BlockArgument ba, bool isExpression) {
                     if (isExpression) {
                        if (resolveBlockArgs) {
                           if (ba.getOwner()->getParentOp() == executionStepOp.getOperation()) {
                              auto selfInput = executionStepOp.getInputs()[ba.getArgNumber()];
                              if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(selfInput)) {
                                 return resolveBlockArgs(blockArg, isExpression);
                              } else {
                                 return nlohmann::json{{"type", "expression_leaf"}, {"leaf_type", "unknown"}};
                              }
                           } else {
                              return resolveBlockArgs(ba, isExpression);
                           }
                        } else {
                           return nlohmann::json{{"type", "expression_leaf"}, {"leaf_type", "unknown"}};
                        }
                     } else {
                        if (ba.getOwner()->getParentOp() == executionStepOp.getOperation()) {
                           if (!blockArgToIndex.contains(ba)) {
                              blockArgToIndex[ba] = blockArgToIndex.size();
                              result["inputs"].push_back(nlohmann::json{{"type", convertDataType(ba.getType())}});
                              result["outerEdges"].push_back({{"type", "requiredInput"}, {"input", getOperandReference(executionStepOp.getInputs()[ba.getArgNumber()], resolveBlockArgs)}, {"output", {{"type", "node"}, {"ref", getOperationReference(executionStepOp)}, {"argnr", result["inputs"].size() - 1}}}});
                           }
                           return nlohmann::json{{"type", "parentArg"}, {"argnr", blockArgToIndex[ba]}};
                        } else {
                           assert(false);
                        }
                     }
                  });
                  result["subops"].push_back(r);
               }
            }

            result["results"] = nlohmann::json::array();
            for (auto t : executionStepOp.getSubOps().front().getTerminator()->getOperands()) {
               result["results"].push_back(nlohmann::json{{"type", convertDataType(t.getType())}});
               result["innerEdges"].push_back({{"type", "resultEdge"}, {"input", getOperandReference(t, resolveBlockArgs)}, {"output", {{"type", "parentResult"}, {"resnr", result["results"].size() - 1}}}});
            }
            return result;
         })

         .Case<subop::CreateThreadLocalOp>([&](subop::CreateThreadLocalOp op) {
            result["subop"] = "create_thread_local";
            return result;
         })
         .Case<subop::GetExternalOp>([&](subop::GetExternalOp op) {
            result["subop"] = "get_external";
            result["meta"] = nlohmann::json::parse(op.getDescr().str());
            return result;
         })
         .Case<subop::CreateThreadLocalOp>([&](subop::CreateThreadLocalOp createThreadLocalOp) {
            result["subop"] = "create_thread_local";
            result["resultType"] = convertDataType(createThreadLocalOp.getRes().getType());
            return result;
         })
         .Case<subop::MergeOp>([&](subop::MergeOp op) {
            result["subop"] = "merge";
            result["stateType"] = stateType(op.getRes().getType());
            result["accesses"].push_back(getOperandReference(op.getThreadLocal(), resolveBlockArgs));
            return result;
         })
         //TODO .Case<subop::LockOp>([&](subop::LockOp lockOp) {})
         .Case<subop::ScanOp>([&](subop::ScanOp scanOp) {
            result["subop"] = "scan";
            result["mapping"] = serializeDefMapping(scanOp.getMapping());
            result["accesses"].push_back(getOperandReference(scanOp.getState(), resolveBlockArgs));
            return result;
         })
         .Case<subop::ScanListOp>([&](subop::ScanListOp scanListOp) {
            result["subop"] = "scan_list";
            result["elem"] = columnToJSON(scanListOp.getElem());
            result["accesses"].push_back(getOperandReference(scanListOp.getList(), resolveBlockArgs));
            return result;
         })
         .Case<subop::ScanRefsOp>([&](subop::ScanRefsOp op) {
            result["subop"] = "scan_ref";
            result["stateType"] = stateType(op.getState().getType());
            result["reference"] = columnToJSON(op.getRef());
            result["accesses"].push_back(getOperandReference(op.getState(), resolveBlockArgs));
            return result;
         })
         .Case<subop::GenerateOp>([&](subop::GenerateOp generateOp) {
            result["subop"] = "generate";
            result["generated"] = nlohmann::json::array();
            for (auto column : generateOp.getGeneratedColumns()) {
               result["generated"].push_back(columnToJSON(mlir::cast<tuples::ColumnDefAttr>(column)));
            }
            return result;
         })
         .Case<subop::UnionOp>([&](subop::UnionOp unionOp) {
            result["subop"] = "union";
            return result;
         })
         .Case<subop::MapOp>([&](subop::MapOp mapOp) {
            result["subop"] = "map";
            result["computed"] = nlohmann::json::array();
            for (auto [column, computed] : llvm::zip(mapOp.getComputedCols(), mapOp.getFn().front().getTerminator()->getOperands())) {
               auto columnDefAttr = mlir::cast<tuples::ColumnDefAttr>(column);
               result["computed"].push_back({{"computed", columnToJSON(columnDefAttr)}, {"expression", convertExpression(computed, [&](mlir::BlockArgument ba, bool isExpression) {
                                                                                            if (ba.getOwner()->getParentOp() == mapOp.getOperation()) {
                                                                                               auto accessedTuple = mlir::cast<tuples::ColumnRefAttr>(mapOp.getInputCols()[ba.getArgNumber()]);
                                                                                               return columnToJSON(accessedTuple);
                                                                                            }
                                                                                            if (resolveBlockArgs) {
                                                                                               return resolveBlockArgs(ba, true);
                                                                                            } else {
                                                                                               return nlohmann::json{{"type", "expression_leaf"}, {"leaf_type", "unknown"}};
                                                                                            }
                                                                                         })}});
            }
            return result;
         })
         .Case<subop::FilterOp>([&](subop::FilterOp filterOp) {
            result["subop"] = "filter";
            result["semantic"] = filterOp.getFilterSemantic() == subop::FilterSemantic::all_true ? "all_true" : "all_false";
            result["columns"] = nlohmann::json::array();
            for (auto x : filterOp.getConditions()) {
               result["columns"].push_back(columnToJSON(mlir::cast<tuples::ColumnRefAttr>(x)));
            }
            return result;
         })
         .Case<subop::RenamingOp>([&](subop::RenamingOp renamingOp) {
            result["subop"] = "renaming";
            result["renamed"] = nlohmann::json::array();
            for (auto column : renamingOp.getColumns()) {
               auto newDef = mlir::cast<tuples::ColumnDefAttr>(column);
               auto old = mlir::cast<tuples::ColumnRefAttr>(mlir::cast<mlir::ArrayAttr>(newDef.getFromExisting())[0]);
               result["renamed"].push_back(nlohmann::json{{"new", columnToJSON(newDef)}, {"old", columnToJSON(old)}});
            }
            return result;
         })
         .Case<subop::NestedMapOp>([&](subop::NestedMapOp nestedMapOp) {
            result["subop"] = "nested_map";
            result["inputs"] = nlohmann::json::array();
            llvm::DenseMap<mlir::Value, size_t> externalValueToIndex;
            if (nestedMapOp.getBody()->begin() != nestedMapOp.getBody()->end()) {
               mlir::Operation* firstOp = &*nestedMapOp.getBody()->begin();
               if (auto nestedExecutionGroup = mlir::dyn_cast_or_null<subop::NestedExecutionGroupOp>(firstOp)) {
                  result["subops"] = nlohmann::json::array();
                  for (auto& op : nestedExecutionGroup.getSubOps().front()) {
                     if (!mlir::isa_and_nonnull<subop::NestedExecutionGroupReturnOp>(&op)) {
                        result["subops"].push_back(convertOperation(&op, [&](mlir::BlockArgument ba, bool isExpression) {
                           if (isExpression) {
                              if (ba.getOwner()->getParentOp() == nestedExecutionGroup.getOperation()) {
                                 if (auto operand = mlir::dyn_cast_or_null<mlir::BlockArgument>(nestedExecutionGroup.getOperands()[ba.getArgNumber()])) {
                                    if (operand.getOwner()->getParentOp() == nestedMapOp.getOperation()) {
                                       auto column = mlir::cast<tuples::ColumnRefAttr>(nestedMapOp.getParameters()[operand.getArgNumber() - 1]); //to account for the extra tuples parameter
                                       auto res = columnToJSON(column);
                                       res["leaf_type"] = "external_column";
                                       return res;
                                    }
                                 }
                              }
                              if (resolveBlockArgs) {
                                 return resolveBlockArgs(ba, isExpression);

                              } else {
                                 return nlohmann::json{{"type", "expression_leaf"}, {"leaf_type", "unknown"}};
                              }
                           } else {
                              if (ba.getOwner()->getParentOp() == nestedExecutionGroup.getOperation()) {
                                 auto realInput = nestedExecutionGroup.getOperands()[ba.getArgNumber()];
                                 if (auto operand = mlir::dyn_cast_or_null<mlir::BlockArgument>(realInput)) {
                                    if (operand.getOwner()->getParentOp() == nestedMapOp.getOperation()) {
                                       auto column = mlir::cast<tuples::ColumnRefAttr>(nestedMapOp.getParameters()[operand.getArgNumber() - 1]); //to account for the extra tuples parameter
                                       return nlohmann::json{{"type", "nested_map_arg"}, {"column", columnToJSON(column)}, {"id", std::string(result["ref"]) + "_" + std::to_string(externalValueToIndex.size())}};
                                    }
                                 }
                                 if (!externalValueToIndex.contains(realInput)) {
                                    externalValueToIndex[realInput] = externalValueToIndex.size();
                                    result["inputs"].push_back(nlohmann::json{{"type", convertDataType(realInput.getType())}});
                                    result["outerEdges"].push_back({{"type", "requiredInput"}, {"input", getOperandReference(realInput, resolveBlockArgs)}, {"output", {{"type", "node"}, {"ref", getOperationReference(nestedMapOp)}, {"argnr", result["inputs"].size() - 1}}}});
                                 }
                                 return nlohmann::json{{"type", "parentArg"}, {"argnr", externalValueToIndex[realInput]}};

                              } else {
                                 assert(false);
                              }
                           }
                        }));
                     }
                  }
               }
            }
            return result;
         })

         .Case<subop::CreateFrom>([&](subop::CreateFrom op) {
            result["subop"] = "create_from";
            result["accesses"].push_back(getOperandReference(op.getState(), resolveBlockArgs));
            return result;
         })
         .Case<subop::CreateSimpleStateOp>([&](subop::CreateSimpleStateOp op) {
            result["subop"] = "create_simple_state";
            return result;
         })
         .Case<subop::CreateHeapOp>([&](subop::CreateHeapOp op) {
            result["subop"] = "create_heap";
            return result;
         })
         .Case<subop::GenericCreateOp>([&](subop::GenericCreateOp op) {
            result["subop"] = "generic_create";
            return result;
         })
         .Case<subop::CreateArrayOp>([&](subop::CreateArrayOp op) {
            result["subop"] = "create_array";
            return result;
         })
         .Case<subop::CreateSortedViewOp>([&](subop::CreateSortedViewOp op) {
            result["subop"] = "create_sorted_view";
            result["accesses"].push_back(getOperandReference(op.getToSort(), resolveBlockArgs));
            return result;
         })
         .Case<subop::CreateHashIndexedView>([&](subop::CreateHashIndexedView op) {
            result["subop"] = "create_hash_indexed_view";
            result["accesses"].push_back(getOperandReference(op.getSource(), resolveBlockArgs));
            return result;
         })
         .Case<subop::CreateContinuousView>([&](subop::CreateContinuousView op) {
            result["subop"] = "create_continuous_view";
            result["accesses"].push_back(getOperandReference(op.getSource(), resolveBlockArgs));
            return result;
         })
         .Case<subop::CreateSegmentTreeView>([&](subop::CreateSegmentTreeView op) {
            result["subop"] = "create_segment_tree_view";
            result["accesses"].push_back(getOperandReference(op.getSource(), resolveBlockArgs));
            return result;
         })

         .Case<subop::MaterializeOp>([&](subop::MaterializeOp op) {
            result["subop"] = "materialize";
            result["stateType"] = stateType(op.getState().getType());
            result["accesses"].push_back(getOperandReference(op.getState(), resolveBlockArgs));
            result["mapping"] = serializeRefMapping(op.getMapping());
            return result;
         })
         .Case<subop::LookupOrInsertOp>([&](subop::LookupOrInsertOp op) {
            result["subop"] = "lookup_or_insert";
            result["stateType"] = stateType(op.getState().getType());
            result["accesses"].push_back(getOperandReference(op.getState(), resolveBlockArgs));
            result["reference"] = columnToJSON(op.getRef());
            return result;
         })
         .Case<subop::InsertOp>([&](subop::InsertOp op) {
            result["subop"] = "insert";
            result["stateType"] = stateType(op.getState().getType());
            result["accesses"].push_back(getOperandReference(op.getState(), resolveBlockArgs));
            result["mapping"] = serializeRefMapping(op.getMapping());
            return result;
         })
         .Case<subop::LookupOp>([&](subop::LookupOp op) {
            result["subop"] = "lookup";
            result["stateType"] = stateType(op.getState().getType());
            result["accesses"].push_back(getOperandReference(op.getState(), resolveBlockArgs));
            result["reference"] = columnToJSON(op.getRef());
            return result;
         })
         .Case<subop::GetBeginReferenceOp>([&](subop::GetBeginReferenceOp op) {
            result["subop"] = "get_begin_reference";
            result["accesses"].push_back(getOperandReference(op.getState(), resolveBlockArgs));
            result["reference"] = columnToJSON(op.getRef());
            return result;
         })
         .Case<subop::GetEndReferenceOp>([&](subop::GetEndReferenceOp op) {
            result["subop"] = "get_end_reference";
            result["accesses"].push_back(getOperandReference(op.getState(), resolveBlockArgs));
            result["reference"] = columnToJSON(op.getRef());
            return result;
         })
         .Case<subop::EntriesBetweenOp>([&](subop::EntriesBetweenOp op) {
            result["subop"] = "entries_between";
            result["leftRef"] = columnToJSON(op.getLeftRef());
            result["rightRef"] = columnToJSON(op.getRightRef());
            result["between"] = columnToJSON(op.getBetween());
            return result;
         })
         .Case<subop::OffsetReferenceBy>([&](subop::OffsetReferenceBy op) {
            result["subop"] = "offset_reference_by";
            result["reference"] = columnToJSON(op.getRef());
            result["offset"] = columnToJSON(op.getIdx());
            result["newRef"] = columnToJSON(op.getNewRef());
            return result;
         })
         .Case<subop::UnwrapOptionalRefOp>([&](subop::UnwrapOptionalRefOp op) {
            result["subop"] = "unwrap_optional_ref";
            result["reference"] = columnToJSON(op.getRef());
            result["optionalRef"] = columnToJSON(op.getOptionalRef());
            return result;
         })

         .Case<subop::GatherOp>([&](subop::GatherOp op) {
            result["subop"] = "gather";
            result["reference"] = columnToJSON(op.getRef());
            result["mapping"] = serializeDefMapping(op.getMapping());
            return result;
         })
         .Case<subop::ScatterOp>([&](subop::ScatterOp op) {
            result["subop"] = "scatter";
            result["reference"] = columnToJSON(op.getRef());
            result["mapping"] = serializeRefMapping(op.getMapping());
            return result;
         })
         .Case<subop::ReduceOp>([&](subop::ReduceOp op) {
            result["subop"] = "reduce";
            result["reference"] = columnToJSON(op.getRef());
            result["updated"] = nlohmann::json::array();
            for (auto [member, computed] : llvm::zip(op.getMembers(), op.getRegion().front().getTerminator()->getOperands())) {
               result["updated"].push_back({{"member", mlir::cast<mlir::StringAttr>(member).str()}, {"expression", convertExpression(computed, [&](mlir::BlockArgument ba, bool isExpression) {
                                                                                                        if (ba.getOwner()->getParentOp() == op.getOperation()) {
                                                                                                           auto argNr = ba.getArgNumber();
                                                                                                           if (argNr < op.getColumns().size()) {
                                                                                                              auto accessedTuple = mlir::cast<tuples::ColumnRefAttr>(op.getColumns()[argNr]);
                                                                                                              ;
                                                                                                              return columnToJSON(accessedTuple);
                                                                                                           } else {
                                                                                                              return nlohmann::json{{"type", "expression_leaf"}, {"leaf_type", "member"}, {"member", mlir::cast<mlir::StringAttr>(op.getMembers()[argNr - op.getColumns().size()]).str()}};
                                                                                                           }
                                                                                                        } else {
                                                                                                           if (resolveBlockArgs) {
                                                                                                              return resolveBlockArgs(ba, true);
                                                                                                           } else {
                                                                                                              return nlohmann::json{{"type", "expression_leaf"}, {"leaf_type", "unknown"}};
                                                                                                           }
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
      auto plan = nlohmann::json::array();
      func->walk([&](mlir::Operation* operation) {
         if (auto setResultOp = mlir::dyn_cast_or_null<subop::SetResultOp>(operation)) {
            if (auto relalgQuery = mlir::dyn_cast_or_null<subop::ExecutionGroupOp>(setResultOp.getState().getDefiningOp())) {
               for (auto& op : relalgQuery.getSubOps().front()) {
                  if (!mlir::isa<subop::ExecutionGroupReturnOp>(&op)) {
                     auto r = convertOperation(&op);
                     if (!plan.empty()) {
                        r["outerEdges"].push_back({{"type", "order"}, {"input", {{"type", "node"}, {"ref", plan[plan.size() - 1]["ref"]}}}, {"output", {{"type", "node"}, {"ref", r["ref"]}}}});
                     }
                     plan.push_back(r);
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
   lingodb::execution::initializeContext(context);
   mlir::OwningOpRef<mlir::ModuleOp> module;
   llvm::SourceMgr sourceMgr;
   mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
   if (int error = loadMLIR(context, module))
      return error;
   std::unordered_map<uint32_t, int64_t> tupleCounts;

   ToJson toJson(module.get(), tupleCounts);
   llvm::outs() << toJson.run() << "\n";
}