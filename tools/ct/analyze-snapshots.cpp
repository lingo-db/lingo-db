#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/DSA/IR/DSADialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/execution/ResultProcessing.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include "json.h"

#include <fstream>
#include <iostream>

namespace cl = llvm::cl;
namespace {
using namespace lingodb::compiler::dialect;
cl::opt<std::string> jsonInputFilename(cl::Positional,
                                       cl::desc("<input json file>"),
                                       cl::init("-"),
                                       cl::value_desc("filename"));

cl::opt<std::string> jsonOutputFilename(cl::Positional,
                                        cl::desc("<output json file>"),
                                        cl::init("-"),
                                        cl::value_desc("filename"));
struct OpElement {
   mlir::Operation* op;
   std::string id;
   std::string mappedId;
};
struct ResultGroupElement {
   std::vector<mlir::Value> values;
};

struct ValueUseElement {
   mlir::Value value;
};
struct BlockArgDefElement {
   mlir::BlockArgument blockArg;
};
struct BlockElement {
   mlir::Block* block;
   bool startsWithCurlyBrace;
   bool endsWithCurlyBrace;
};
struct DummyElement {
};

struct MLIRElement {
   size_t start;
   size_t length;
   std::variant<OpElement, ResultGroupElement, ValueUseElement, BlockArgDefElement, BlockElement, DummyElement> element;
};

std::string get(const char* buf, size_t start, size_t length) {
   return std::string(buf + start, length);
}
void addRaw(nlohmann::json& j, std::string s) {
   size_t pos = 0;
   while ((pos = s.find("loc(", pos)) != std::string::npos) {
      size_t end = s.find(")", pos);
      if (end == std::string::npos) {
         break;
      }
      j.push_back({{"type", "raw"}, {"value", s.substr(0, pos)}});
      j.push_back({{"type", "loc"}, {"value", s.substr(pos, end - pos + 1)}});
      s = s.substr(end + 1);
      pos = 0;
   }
   j.push_back({{"type", "raw"}, {"value", s}});
}

nlohmann::json toJSON(size_t& startElement, std::vector<MLIRElement>& elements, size_t startPos, size_t endPos, const char* buf, std::function<void(nlohmann::json&, std::string)> processRaw) {
   nlohmann::json j;
   auto pos = startPos;
   while (pos < endPos) {
      auto oldPos = pos;
      auto element = elements[startElement];
      if (element.start == pos) {
         startElement++;
         if (std::holds_alternative<OpElement>(element.element)) {
            auto opElement = std::get<OpElement>(element.element);
            std::vector<std::pair<std::string, std::function<void(nlohmann::json&, std::string)>>> toHandle;
            for (auto t : opElement.op->getResultTypes()) {
               std::string s;
               llvm::raw_string_ostream os(s);
               t.print(os);
               toHandle.push_back({os.str(), [](nlohmann::json& j, std::string s) {
                                      j.push_back({{"type", "type"}, {"value", s}});
                                   }});
               auto startOfDialect = s.find("<");
               if (startOfDialect != std::string::npos) {
                  toHandle.push_back({s.substr(startOfDialect), [](nlohmann::json& j, std::string s) {
                                         j.push_back({{"type", "type"}, {"value", s}});
                                      }});
               }
            }
            for (auto t : opElement.op->getOperandTypes()) {
               std::string s;
               llvm::raw_string_ostream os(s);
               t.print(os);
               toHandle.push_back({os.str(), [](nlohmann::json& j, std::string s) {
                                      j.push_back({{"type", "type"}, {"value", s}});
                                   }});
               auto startOfDialect = s.find("<");
               if (startOfDialect != std::string::npos) {
                  toHandle.push_back({s.substr(startOfDialect), [](nlohmann::json& j, std::string s) {
                                         j.push_back({{"type", "type"}, {"value", s}});
                                      }});
               }
            }
            for (auto a : opElement.op->getAttrs()) {
               std::string s;
               llvm::raw_string_ostream os(s);
               a.getValue().print(os);
               toHandle.push_back({os.str(), [](nlohmann::json& j, std::string s) {
                                      j.push_back({{"type", "attribute"}, {"value", s}});
                                   }});
            }
            toHandle.push_back({opElement.op->getName().getStringRef().str(), [](nlohmann::json& j, std::string s) {
                                   j.push_back({{"type", "opName"}, {"value", s}});
                                }});
            nlohmann::json op;
            op["type"] = "op";
            //op["original"] = get(buf, element.start, element.length);
            op["id"] = opElement.id;
            op["mappedId"] = opElement.mappedId;
            op["children"] = toJSON(startElement, elements, element.start, element.start + element.length, buf, [&](nlohmann::json& j, std::string s) {
               std::vector<std::tuple<size_t, size_t, std::function<void(nlohmann::json&, std::string)>>> matches;
               for (auto [pattern, handler] : toHandle) {
                  size_t pos = 0;
                  while ((pos = s.find(pattern, pos)) != std::string::npos) {
                     matches.push_back({pos, pattern.length(), handler});
                     pos += pattern.length();
                  }
               }
               std::sort(matches.begin(), matches.end(), [](const auto& a, const auto& b) {
                  if (std::get<0>(a) < std::get<0>(b)) {
                     return true;
                  } else if (std::get<0>(a) == std::get<0>(b)) {
                     return std::get<1>(a) > std::get<1>(b);
                  } else {
                     return false;
                  }
               });
               matches.push_back({s.length(), 0, [](nlohmann::json& j, std::string s) {}});
               size_t pos = 0;
               size_t matchPos = 0;
               while (pos < s.length()) {
                  auto [matchStart, matchLength, handler] = matches[matchPos];
                  if (pos < matchStart) {
                     addRaw(j, s.substr(pos, matchStart - pos));
                     pos = matchStart;
                  } else if (pos == matchStart) {
                     handler(j, s.substr(matchStart, matchLength));
                     pos = matchStart + matchLength;
                     matchPos++;
                  } else {
                     matchPos++;
                  }
               }
            });
            j.push_back(std::move(op));
         } else if (std::holds_alternative<ResultGroupElement>(element.element)) {
            nlohmann::json resultGroup;
            resultGroup["type"] = "resultGroup";
            resultGroup["value"] = get(buf, pos, element.length);
            j.push_back(std::move(resultGroup));
         } else if (std::holds_alternative<ValueUseElement>(element.element)) {
            nlohmann::json valueUse;
            valueUse["type"] = "valueUse";
            valueUse["value"] = get(buf, pos, element.length);
            std::string mlirType;
            llvm::raw_string_ostream os(mlirType);
            mlir::cast<mlir::Type>(std::get<ValueUseElement>(element.element).value.getType()).print(os);
            valueUse["mlirType"] = mlirType;
            j.push_back(std::move(valueUse));
         } else if (std::holds_alternative<BlockArgDefElement>(element.element)) {
            nlohmann::json blockArgDef;
            blockArgDef["type"] = "blockArgDef";
            blockArgDef["value"] = get(buf, pos, element.length);
            j.push_back(std::move(blockArgDef));
         } else if (std::holds_alternative<BlockElement>(element.element)) {
            nlohmann::json block;
            block["type"] = "block";
            block["sCB"] = std::get<BlockElement>(element.element).startsWithCurlyBrace;
            block["eCB"] = std::get<BlockElement>(element.element).endsWithCurlyBrace;
            block["children"] = toJSON(startElement, elements, element.start, element.start + element.length, buf, addRaw);
            j.push_back(std::move(block));
         } else if (std::holds_alternative<DummyElement>(element.element)) {
         } else {
            processRaw(j, get(buf, pos, element.length));
         }
         assert(element.start + element.length > pos);
         pos = element.start + element.length;
      } else if (pos < element.start) {
         size_t end = std::min(element.start, endPos);
         processRaw(j, get(buf, pos, end - pos));
         assert(end > pos);
         pos = end;
      } else {
         assert(false);
      }
      assert(oldPos < pos);
   }
   return j;
}
std::string referenceFromFileAndLine(std::string fileName, size_t line) {
   auto baseNameStarts = fileName.find_last_of("/");
   if (baseNameStarts != std::string::npos) {
      fileName = fileName.substr(baseNameStarts + 1);
   }
   auto endingStarts = fileName.find(".");
   if (endingStarts != std::string::npos) {
      fileName = fileName.substr(0, endingStarts);
   }
   return fileName + std::string(":") + std::to_string(line);
}
std::string referenceFromLineLoc(mlir::FileLineColLoc lineLoc) {
   auto fileName = lineLoc.getFilename().str();
   return referenceFromFileAndLine(fileName, lineLoc.getLine());
}
} // namespace
int main(int argc, char** argv) {
   cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

   mlir::DialectRegistry registry;
   registry.insert<relalg::RelAlgDialect>();
   registry.insert<tuples::TupleStreamDialect>();
   registry.insert<subop::SubOperatorDialect>();
   registry.insert<db::DBDialect>();
   registry.insert<dsa::DSADialect>();
   registry.insert<mlir::func::FuncDialect>();
   registry.insert<mlir::arith::ArithDialect>();

   registry.insert<mlir::memref::MemRefDialect>();
   registry.insert<util::UtilDialect>();
   registry.insert<mlir::cf::ControlFlowDialect>();
   registry.insert<mlir::LLVM::LLVMDialect>();
   registry.insert<mlir::gpu::GPUDialect>();
   registry.insert<mlir::scf::SCFDialect>();
   registry.insert<mlir::DLTIDialect>();
   std::ifstream fileIn(jsonInputFilename);

   auto inputJson = nlohmann::json::parse(fileIn);
   nlohmann::json jsonOutput;
   for (auto step : inputJson) {
      mlir::MLIRContext context;
      context.appendDialectRegistry(registry);
      auto inputFilename = step["file"].get<std::string>();
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
      if (std::error_code ec = fileOrErr.getError()) {
         llvm::errs() << "Could not open input file: " << ec.message() << "\n";
         return -1;
      }

      // Parse the input mlir.
      llvm::SourceMgr sourceMgr;
      auto bufferId = sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
      mlir::Block block;
      mlir::AsmParserState state;
      if (mlir::parseAsmSourceFile(sourceMgr, &block, mlir::ParserConfig(&context), &state).failed()) {
         break;
      }
      const auto* basePtr = sourceMgr.getBufferInfo(bufferId).Buffer->getBuffer().data();
      std::vector<MLIRElement> elements;
      for (auto opDef : state.getOpDefs()) {
         size_t start = opDef.scopeLoc.Start.getPointer() - basePtr;
         auto realStart = start;
         for (size_t resultGroupPosition = 0; resultGroupPosition < opDef.resultGroups.size(); resultGroupPosition++) {
            auto resultGroup = opDef.resultGroups[resultGroupPosition];
            auto resultGroupSize = (resultGroupPosition + 1 < opDef.resultGroups.size() ? opDef.resultGroups[resultGroupPosition + 1].startIndex : opDef.op->getNumResults()) - resultGroup.startIndex;
            size_t resultGroupStart = resultGroup.definition.loc.Start.getPointer() - basePtr;
            size_t resultGroupLength = resultGroup.definition.loc.End.getPointer() - resultGroup.definition.loc.Start.getPointer();
            realStart = std::min(realStart, resultGroupStart);
            std::vector<mlir::Value> resultGroupValues;
            auto list = opDef.op->getResults().drop_front(resultGroup.startIndex).take_front(resultGroupSize);
            for (auto value : list) {
               resultGroupValues.push_back(value);
            }
            elements.push_back({resultGroupStart, resultGroupLength, ResultGroupElement{resultGroupValues}});
            for (auto use : resultGroup.definition.uses) {
               size_t useStart = use.Start.getPointer() - basePtr;
               size_t useLength = use.End.getPointer() - use.Start.getPointer();
               elements.push_back({useStart, useLength, ValueUseElement{resultGroupValues.front()}});
            }
         }
         size_t length = (opDef.scopeLoc.End.getPointer() - basePtr) - realStart;
         std::string mappedId;
         if (auto fileLineLoc = mlir::dyn_cast_or_null<mlir::FileLineColLoc>(opDef.op->getLoc())) {
            mappedId = referenceFromLineLoc(fileLineLoc);
         }
         auto [line, _] = sourceMgr.getLineAndColumn(opDef.loc.Start);
         elements.push_back({realStart, length, OpElement{opDef.op, referenceFromFileAndLine(inputFilename, line), mappedId}});
      }
      for (auto blockDef : state.getBlockDefs()) {
         size_t start = blockDef.definition.loc.Start.getPointer() - basePtr;
         bool startsWithCurlyBrace = false;
         if (auto* firstOpDef = state.getOpDef(&*blockDef.block->getOperations().begin())) {
            startsWithCurlyBrace = get(basePtr, start, firstOpDef->loc.Start.getPointer() - basePtr - start).find("{") != std::string::npos;
         }

         auto* end = blockDef.definition.loc.End.getPointer();
         for (auto rit = blockDef.block->getOperations().rbegin(); rit != blockDef.block->getOperations().rend(); rit++) {
            auto& lastOp = *rit;
            const auto* lastOpDef = state.getOpDef(&lastOp);
            if (lastOpDef) {
               end = lastOpDef->scopeLoc.End.getPointer();
               break;
            }
         }
         bool endsWithCurlyBrace = false;
         while (*end != '}' && *end != '^') {
            end++;
         }
         if (*end != '^') {
            end++;
            endsWithCurlyBrace = true;
         }
         size_t length = end - blockDef.definition.loc.Start.getPointer();
         elements.push_back({start, length, BlockElement{blockDef.block, startsWithCurlyBrace, endsWithCurlyBrace}});
         for (auto [blockArgDef, blockArg] : llvm::zip(blockDef.arguments, blockDef.block->getArguments())) {
            size_t start = blockArgDef.loc.Start.getPointer() - basePtr;
            size_t length = blockArgDef.loc.End.getPointer() - blockArgDef.loc.Start.getPointer();
            elements.push_back({start, length, BlockArgDefElement{blockArg}});
            for (auto use : blockArgDef.uses) {
               size_t start = use.Start.getPointer() - basePtr;
               size_t length = use.End.getPointer() - use.Start.getPointer();
               elements.push_back({start, length, ValueUseElement{blockArg}});
            }
         }
      }
      for (auto aliasDef : state.getAttributeAliasDefs()) {
         if (mlir::isa<mlir::LocationAttr>(aliasDef.value)) {
            size_t start = aliasDef.definition.loc.Start.getPointer() - basePtr;
            const auto* end = aliasDef.definition.loc.End.getPointer();
            while (*end != '\n') {
               end++;
            }
            end++;
            size_t length = end - aliasDef.definition.loc.Start.getPointer();
            elements.push_back({start, length, DummyElement{}});
         }
      }
      std::sort(elements.begin(), elements.end(), [](const MLIRElement& a, const MLIRElement& b) {
         if (a.start < b.start) {
            return true;
         } else if (a.start == b.start) {
            return a.length > b.length;
         } else {
            return false;
         }
      });

      elements.push_back(MLIRElement{sourceMgr.getBufferInfo(bufferId).Buffer->getBuffer().size(), 0ull, DummyElement{}});
      size_t startElement = 0;
      auto parsed = toJSON(startElement, elements, 0, sourceMgr.getBufferInfo(bufferId).Buffer->getBuffer().size(), basePtr, addRaw);
      jsonOutput.push_back({{"parsed", std::move(parsed)}, {"passInfo", step}});
   }
   if (jsonOutputFilename == "-") {
      std::cout << jsonOutput.dump() << std::endl;
   } else {
      std::ofstream file(jsonOutputFilename);
      file << jsonOutput.dump() << std::endl;
   }
}