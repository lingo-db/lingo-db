//emits C code for a MLIR module containing operations of the func, arith, scf, and util dialects

// Derived from https://github.com/llvm/llvm-project/blob/56470b72f1fc1727d5ee87e2fed96e7dad286230/mlir/lib/Target/Cpp/TranslateToCpp.cpp
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "lingodb/compiler/Dialect/util/FunctionHelper.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "lingodb/execution/CBackend.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/IndentedOstream.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include <iomanip>
#include <regex>
#include <unordered_set>
#include <utility>

using namespace mlir;
using llvm::formatv;
namespace {

using namespace lingodb::compiler::dialect;
/// Convenience functions to produce interleaved output with functions returning
/// a LogicalResult. This is different than those in STLExtras as functions used
/// on each element doesn't return a string.
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor>
inline LogicalResult
interleaveWithError(ForwardIterator begin, ForwardIterator end,
                    UnaryFunctor eachFn, NullaryFunctor betweenFn) {
   if (begin == end)
      return success();
   if (failed(eachFn(*begin)))
      return failure();
   ++begin;
   for (; begin != end; ++begin) {
      betweenFn();
      if (failed(eachFn(*begin)))
         return failure();
   }
   return success();
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult interleaveWithError(const Container& c,
                                         UnaryFunctor eachFn,
                                         NullaryFunctor betweenFn) {
   return interleaveWithError(c.begin(), c.end(), eachFn, betweenFn);
}

template <typename Container, typename UnaryFunctor>
inline LogicalResult interleaveCommaWithError(const Container& c,
                                              raw_ostream& os,
                                              UnaryFunctor eachFn) {
   return interleaveWithError(c.begin(), c.end(), eachFn, [&]() { os << ", "; });
}
template <typename Container, typename UnaryFunctor>
inline LogicalResult interleaveSemicolonWithError(const Container& c,
                                                  raw_ostream& os,
                                                  UnaryFunctor eachFn) {
   return interleaveWithError(c.begin(), c.end(), eachFn, [&]() { os << "; "; });
}
} // end anonymous namespace

namespace {
/// Emitter that uses dialect specific emitters to emit C++ code.
struct CppEmitter {
   std::unordered_map<std::string, void*> symbolMap;
   void* resolveFunction(std::string name) {
      return symbolMap.at(name);
   }
   explicit CppEmitter(raw_ostream& os, bool declareVariablesAtTop);

   /// Emits attribute or returns failure.
   LogicalResult emitAttribute(Location loc, Attribute attr);

   /// Emits operation 'op' with/without training semicolon or returns failure.
   LogicalResult emitOperation(Operation& op, bool trailingSemicolon);
   LogicalResult typeToString(Location loc, Type type, std::string& string);

   /// Emits type 'type' or returns failure.
   LogicalResult emitType(Location loc, Type type, llvm::raw_ostream& os);
   LogicalResult emitType(Location loc, Type type) {
      return emitType(loc, type, os);
   }

   /// Emits array of types as a std::tuple of the emitted types.
   /// - emits void for an empty array;
   /// - emits the type of the only element for arrays of size one;
   /// - emits a std::tuple otherwise;
   LogicalResult emitTypes(Location loc, ArrayRef<Type> types,
                           llvm::raw_ostream& os);
   LogicalResult emitTypes(Location loc, ArrayRef<Type> types) {
      return emitTypes(loc, types, os);
   }

   /// Emits array of types as a std::tuple of the emitted types independently of
   /// the array size.
   LogicalResult emitTupleType(Location loc, ArrayRef<Type> types,
                               llvm::raw_ostream& os);
   LogicalResult emitTupleType(Location loc, ArrayRef<Type> types) {
      return emitTupleType(loc, types, os);
   }

   /// Emits an assignment for a variable which has been declared previously.
   LogicalResult emitVariableAssignment(OpResult result);

   /// Emits a variable declaration for a result of an operation.
   LogicalResult emitVariableDeclaration(OpResult result,
                                         bool trailingSemicolon);

   /// Emits the variable declaration and assignment prefix for 'op'.
   /// - emits separate variable followed by std::tie for multi-valued operation;
   /// - emits single type followed by variable for single result;
   /// - emits nothing if no value produced by op;
   /// Emits final '=' operator where a type is produced. Returns failure if
   /// any result type could not be converted.
   LogicalResult emitAssignPrefix(Operation& op);

   /// Emits a label for the block.
   LogicalResult emitLabel(Block& block);

   /// Emits the operands and atttributes of the operation. All operands are
   /// emitted first and then all attributes in alphabetical order.
   LogicalResult emitOperandsAndAttributes(Operation& op,
                                           ArrayRef<StringRef> exclude = {});

   /// Emits the operands of the operation. All operands are emitted in order.
   LogicalResult emitOperands(Operation& op);

   /// Return the existing or a new name for a Value.
   StringRef getOrCreateName(Value val);
   std::string getVariableName();

   /// Return the existing or a new label of a Block.
   StringRef getOrCreateName(Block& block);

   /// Whether to map an mlir integer to a unsigned integer in C++.
   bool shouldMapToUnsigned(IntegerType::SignednessSemantics val);

   /// RAII helper function to manage entering/exiting C++ scopes.
   struct Scope {
      Scope(CppEmitter& emitter)
         : valueMapperScope(emitter.valueMapper),
           blockMapperScope(emitter.blockMapper), emitter(emitter) {
         emitter.valueInScopeCount.push(emitter.valueInScopeCount.top());
         emitter.labelInScopeCount.push(emitter.labelInScopeCount.top());
      }
      ~Scope() {
         emitter.valueInScopeCount.pop();
         emitter.labelInScopeCount.pop();
      }

      private:
      llvm::ScopedHashTableScope<Value, std::string> valueMapperScope;
      llvm::ScopedHashTableScope<Block*, std::string> blockMapperScope;
      CppEmitter& emitter;
   };

   /// Returns wether the Value is assigned to a C++ variable in the scope.
   bool hasValueInScope(Value val);

   // Returns whether a label is assigned to the block.
   bool hasBlockLabel(Block& block);

   /// Returns the output stream.
   raw_indented_ostream& ostream() { return os; };

   /// Returns if all variables for op results and basic block arguments need to
   /// be declared at the beginning of a function.
   bool shouldDeclareVariablesAtTop() { return declareVariablesAtTop; };
   std::string& getStructDefs() { return structDefs; }

   private:
   using ValueMapper = llvm::ScopedHashTable<Value, std::string>;
   using BlockMapper = llvm::ScopedHashTable<Block*, std::string>;

   /// Output stream to emit to.
   raw_indented_ostream os;

   /// Boolean to enforce that all variables for op results and block
   /// arguments are declared at the beginning of the function. This also
   /// includes results from ops located in nested regions.
   bool declareVariablesAtTop;

   /// Map from value to name of C++ variable that contain the name.
   ValueMapper valueMapper;

   /// Map from block to name of C++ label.
   BlockMapper blockMapper;

   /// The number of values in the current scope. This is used to declare the
   /// names of values in a scope.
   std::stack<int64_t> valueInScopeCount;
   std::stack<int64_t> labelInScopeCount;
   std::string structDefs;
   std::unordered_set<std::string> cachedStructs;
};

LogicalResult printConstantOp(CppEmitter& emitter, Operation* operation,
                              Attribute value) {
   OpResult result = operation->getResult(0);

   // Only emit an assignment as the variable was already declared when printing
   // the FuncOp.
   if (emitter.shouldDeclareVariablesAtTop()) {
      if (failed(emitter.emitVariableAssignment(result)))
         return failure();
      return emitter.emitAttribute(operation->getLoc(), value);
   }
   // Emit a variable declaration.
   if (failed(emitter.emitAssignPrefix(*operation)))
      return failure();
   return emitter.emitAttribute(operation->getLoc(), value);
}

LogicalResult printOperation(CppEmitter& emitter, util::GenericMemrefCastOp op) {
   raw_ostream& os = emitter.ostream();
   if (failed(emitter.emitAssignPrefix(*op.getOperation())))
      return failure();
   os << "(";
   if (failed(emitter.emitType(op.getLoc(), op.getType()))) {
      return failure();
   }
   os << ") " << emitter.getOrCreateName(op.getVal());
   return success();
}
LogicalResult printStandardOperation(CppEmitter& emitter, mlir::Operation* op, const std::function<void(raw_ostream&)>& fn) {
   if (op->getNumResults() != 1) {
      return failure();
   }
   raw_ostream& os = emitter.ostream();
   if (failed(emitter.emitAssignPrefix(*op)))
      return failure();
   fn(os);
   return success();
}

enum ArithmeticCOperation {
   ADD,
   SUBTRACT,
   MULTIPLY,
   DIVIDE,
   MODULO,
   MODULO_U,
   FMOD,
   AND,
   OR,
   MAX,
   MAX_U,
   MIN,
   MIN_U,
   NEGATE,
   SHIFT_LEFT,
   SHIFT_RIGHT,
   SHIFT_RIGHT_U,
   XOR,
   TERNARY_OP,
   COMPARE_EQUALS,
   COMPARE_NOT_EQUALS,
   COMPARE_LESS,
   COMPARE_GREATER,
   COMPARE_LESS_EQ,
   COMPARE_GREATER_EQ,
   COMPARE_LESS_U,
   COMPARE_GREATER_U,
   COMPARE_LESS_EQ_U,
   COMPARE_GREATER_EQ_U,
   CEIL_DIV,
   FLOOR_DIV
};

std::unordered_map<ArithmeticCOperation, std::string> opToFormatString{
   {ArithmeticCOperation::ADD, "@0 + @1"},
   {ArithmeticCOperation::SUBTRACT, "@0 - @1"},
   {ArithmeticCOperation::MULTIPLY, "@0 * @1"},
   {ArithmeticCOperation::DIVIDE, "@0 / @1"},
   {ArithmeticCOperation::MODULO, "@0 % @1"},
   {ArithmeticCOperation::MODULO_U, "static_cast<std::make_unsigned<decltype(@0)>::type>(@0) % static_cast<std::make_unsigned<decltype(@1)>::type>(@1)"},
   {ArithmeticCOperation::FMOD, "fmod(@0, @1)"},
   {ArithmeticCOperation::AND, "@0 & @1"},
   {ArithmeticCOperation::OR, "@0 | @1"},
   {ArithmeticCOperation::MAX_U, "@0 > @1 ? @0 : @1"},
   {ArithmeticCOperation::MAX, "@0 > @1 ? @0 : @1"},
   {ArithmeticCOperation::MIN_U, "@0 < @1 ? @0 : @1"},
   {ArithmeticCOperation::MIN, "@0 < @1 ? @0 : @1"},
   {ArithmeticCOperation::MAX_U, "static_cast<std::make_unsigned<decltype(@0)>::type>(@0) > static_cast<std::make_unsigned<decltype(@1)>::type>(@1) ? @0 : @1"},
   {ArithmeticCOperation::MIN_U, "static_cast<std::make_unsigned<decltype(@0)>::type>(@0) < static_cast<std::make_unsigned<decltype(@1)>::type>(@1) ? @0 : @1"},
   {ArithmeticCOperation::NEGATE, "-@0"},
   {ArithmeticCOperation::SHIFT_LEFT, "@0 << @1"},
   {ArithmeticCOperation::SHIFT_RIGHT, "@0 >> @1"},
   {ArithmeticCOperation::SHIFT_RIGHT_U, "static_cast<std::make_unsigned<decltype(@0)>::type>(@0) >> @1"},
   {ArithmeticCOperation::XOR, "@0 ^ @1"},
   {ArithmeticCOperation::TERNARY_OP, "@0 ? @1 : @2"},
   {ArithmeticCOperation::COMPARE_EQUALS, "@0 == @1"},
   {ArithmeticCOperation::COMPARE_NOT_EQUALS, "@0 != @1"},
   {ArithmeticCOperation::COMPARE_LESS, "@0 < @1"},
   {ArithmeticCOperation::COMPARE_LESS_EQ, "@0 <= @1"},
   {ArithmeticCOperation::COMPARE_GREATER, "@0 > @1"},
   {ArithmeticCOperation::COMPARE_GREATER_EQ, "@0 >= @1"},
   {ArithmeticCOperation::COMPARE_LESS_U, "static_cast<std::make_unsigned<decltype(@0)>::type>(@0) < static_cast<std::make_unsigned<decltype(@1)>::type>(@1)"},
   {ArithmeticCOperation::COMPARE_LESS_EQ_U, "static_cast<std::make_unsigned<decltype(@0)>::type>(@0) <= static_cast<std::make_unsigned<decltype(@1)>::type>(@1)"},
   {ArithmeticCOperation::COMPARE_GREATER_U, "static_cast<std::make_unsigned<decltype(@0)>::type>(@0) > static_cast<std::make_unsigned<decltype(@1)>::type>(@1)"},
   {ArithmeticCOperation::COMPARE_GREATER_EQ_U, "static_cast<std::make_unsigned<decltype(@0)>::type>(@0) >= static_cast<std::make_unsigned<decltype(@1)>::type>(@1)"},
   {ArithmeticCOperation::CEIL_DIV, "@0 / @1 + ((@0 % @1)>0)"},
   {ArithmeticCOperation::FLOOR_DIV, "@0 / @1 - ((@0 % @1 < 0)"},
};
LogicalResult printArithmeticOperation(CppEmitter& emitter, mlir::Operation* arithmeticOp, ArithmeticCOperation type) {
   raw_ostream& os = emitter.ostream();
   Operation& op = *arithmeticOp;

   if (failed(emitter.emitAssignPrefix(op)))
      return failure();

   std::string formatString = opToFormatString.at(type);
   size_t numOperands = arithmeticOp->getNumOperands();

   for (size_t i = 0; i < numOperands; ++i) {
      std::regex toReplace("@" + std::to_string(i));
      std::string operand{emitter.getOrCreateName(op.getOperand(i))};
      formatString = std::regex_replace(formatString, toReplace, operand);
   }
   os << formatString;

   return success();
}
LogicalResult printSimpleCast(CppEmitter& emitter, mlir::Operation* op) {
   if (op->getNumResults() != 1 || op->getNumOperands() != 1) {
      return failure();
   }
   raw_ostream& os = emitter.ostream();
   if (failed(emitter.emitAssignPrefix(*op)))
      return failure();
   os << "(";
   if (failed(emitter.emitType(op->getLoc(), op->getResult(0).getType()))) {
      return failure();
   }
   os << ")" << emitter.getOrCreateName(op->getOperand(0));
   return success();
}
LogicalResult printOperation(CppEmitter& emitter, util::AllocaOp op) {
   std::string baseType;
   if (failed(emitter.typeToString(op->getLoc(), op.getType().getElementType(), baseType))) {
      return failure();
   }
   if (op.getSize()) {
      return printStandardOperation(emitter, op, [&](auto& os) { os << "(" << baseType << "*) alloca(sizeof(" << baseType << ")*" << emitter.getOrCreateName(op.getSize()) << ")"; });
   } else {
      return printStandardOperation(emitter, op, [&](auto& os) { os << "(" << baseType << "*) alloca(sizeof(" << baseType << "))"; });
   }
}
LogicalResult printOperation(CppEmitter& emitter, util::AllocOp op) {
   std::string baseType;
   if (failed(emitter.typeToString(op->getLoc(), op.getType().getElementType(), baseType))) {
      return failure();
   }
   if (op.getSize()) {
      return printStandardOperation(emitter, op, [&](auto& os) { os << "(" << baseType << "*) malloc(sizeof(" << baseType << ")*" << emitter.getOrCreateName(op.getSize()) << ")"; });
   } else {
      return printStandardOperation(emitter, op, [&](auto& os) { os << "(" << baseType << "*) malloc(sizeof(" << baseType << "))"; });
   }
}
static std::string escapeQuotes(std::string s) {
   std::stringstream sstream;
   sstream << std::quoted(s);
   auto quoteEscaped = sstream.str().substr(1, sstream.str().length() - 2);
   auto nlEscaped = std::regex_replace(quoteEscaped, std::regex("\n"), "\\n");
   return nlEscaped;
}
LogicalResult printOperation(CppEmitter& emitter, util::DeAllocOp op) {
   emitter.ostream() << "free(" << emitter.getOrCreateName(op.getRef()) << ")";
   return mlir::success();
}
LogicalResult printOperation(CppEmitter& emitter, util::CreateConstVarLen op) {
   return printStandardOperation(emitter, op, [&](auto& os) { os << "runtime::VarLen32{reinterpret_cast<const uint8_t*>(\"" << escapeQuotes(op.getStr().str()) << "\"), " << op.getStr().size() << "}"; });
}
LogicalResult printOperation(CppEmitter& emitter, util::PackOp op) {
   return printStandardOperation(emitter, op, [&](auto& os) {
      os << "{";
      for (size_t i = 0; i < op.getOperands().size(); i++) {
         os << emitter.getOrCreateName(op->getOperand(i)) << (i == op.getOperands().size() - 1 ? "}" : ", ");
      }
   });
}
LogicalResult printOperation(CppEmitter& emitter, util::TupleElementPtrOp op) {
   return printStandardOperation(emitter, op, [&](auto& os) { os << "&" << emitter.getOrCreateName(op.getRef()) << "->t" << op.getIdx(); });
}
LogicalResult printOperation(CppEmitter& emitter, util::GetTupleOp op) {
   return printStandardOperation(emitter, op, [&](auto& os) { os << emitter.getOrCreateName(op.getTuple()) << ".t" << op.getOffset(); });
}
LogicalResult printOperation(CppEmitter& emitter, util::CreateVarLen op) {
   return printStandardOperation(emitter, op, [&](auto& os) { os << "runtime::VarLen32{reinterpret_cast<uint8_t*>(" << emitter.getOrCreateName(op.getRef()) << "), (uint32_t) " << emitter.getOrCreateName(op.getLen()) << "}"; });
}
LogicalResult printOperation(CppEmitter& emitter, util::ToMemrefOp op) {
   return printStandardOperation(emitter, op, [&](auto& os) { os << emitter.getOrCreateName(op.getRef()); });
}
LogicalResult printOperation(CppEmitter& emitter, memref::AtomicRMWOp op) {
   assert(op.getKind() == mlir::arith::AtomicRMWKind::addf || op.getKind() == mlir::arith::AtomicRMWKind::addi || op.getKind() == mlir::arith::AtomicRMWKind::ori);
   return printStandardOperation(emitter, op, [&](auto& os) {
      os << "std::atomic_ref<";
      auto s = emitter.emitType(op.getLoc(), op.getValue().getType()).succeeded();
      assert(s);
      if (op.getKind() == mlir::arith::AtomicRMWKind::addf || op.getKind() == mlir::arith::AtomicRMWKind::addi) {
         os << ">(*" << emitter.getOrCreateName(op.getMemref()) << ").fetch_add(" << emitter.getOrCreateName(op.getValue()) << ")";
      } else {
         os << ">(*" << emitter.getOrCreateName(op.getMemref()) << ").fetch_or(" << emitter.getOrCreateName(op.getValue()) << ")";
      }
   });
}
LogicalResult printOperation(CppEmitter& emitter, util::VarLenCmp op) {
   if (failed(emitter.emitVariableDeclaration(op->getResult(0), false))) {
      return failure();
   }
   raw_ostream& os = emitter.ostream();
   auto left = emitter.getOrCreateName(op.getLeft());
   auto right = emitter.getOrCreateName(op.getRight());
   os << "= *reinterpret_cast<__int128*>(&" << left << ") ==*reinterpret_cast<__int128*>(&" << right << ");\n";

   if (failed(emitter.emitVariableDeclaration(op->getResult(1), false))) {
      return failure();
   }
   os << "= " << left << ".getLen()>12 && " << left << ".getLen()==" << right << ".getLen() &&" << left << ".first4==" << right << ".first4";
   return success();
}
LogicalResult printOperation(CppEmitter& emitter, util::VarLenTryCheapHash op) {
   if (failed(emitter.emitVariableDeclaration(op->getResult(0), false))) {
      return failure();
   }
   raw_ostream& os = emitter.ostream();
   auto input = emitter.getOrCreateName(op.getVarlen());
   os << "= " << input << ".getLen()<13;\n";
   if (failed(emitter.emitVariableDeclaration(op->getResult(1), false))) {
      return failure();
   }
   os << "= hash_combine(hash_64(reinterpret_cast<size_t*>(&" << input << ")[0]), hash_64(reinterpret_cast<size_t*>(&" << input << ")[1]))";

   return success();
}
LogicalResult printOperation(CppEmitter& emitter, util::VarLenGetLen op) {
   return printStandardOperation(emitter, op, [&](auto& os) { os << emitter.getOrCreateName(op.getVarlen()) << ".getLen()"; });
}
LogicalResult printOperation(CppEmitter& emitter, util::UndefOp op) {
   return printStandardOperation(emitter, op, [&](auto& os) { os << "{}"; });
}
LogicalResult printOperation(CppEmitter& emitter, util::Hash64 op) {
   return printStandardOperation(emitter, op, [&](auto& os) { os << "hash_64(" << emitter.getOrCreateName(op.getVal()) << ")"; });
}
LogicalResult printOperation(CppEmitter& emitter, util::HashCombine op) {
   return printStandardOperation(emitter, op, [&](auto& os) { os << "hash_combine(" << emitter.getOrCreateName(op.getH1()) << "," << emitter.getOrCreateName(op.getH2()) << ")"; });
}
LogicalResult printOperation(CppEmitter& emitter, util::HashVarLen op) {
   return printStandardOperation(emitter, op, [&](auto& os) { os << "hashVarLenData(" << emitter.getOrCreateName(op.getVal()) << ")"; });
}
LogicalResult printOperation(CppEmitter& emitter, util::FilterTaggedPtr op) {
   return printStandardOperation(emitter, op, [&](auto& os) { os << "runtime::filterTagged(" << emitter.getOrCreateName(op.getRef()) << "," << emitter.getOrCreateName(op.getHash()) << ")"; });
}
LogicalResult printOperation(CppEmitter& emitter, util::UnTagPtr op) {
   return printStandardOperation(emitter, op, [&](auto& os) { os << "runtime::untag(" << emitter.getOrCreateName(op.getRef()) << ")"; });
}
LogicalResult printOperation(CppEmitter& emitter, util::BufferCastOp op) {
   return printStandardOperation(emitter, op, [&](auto& os) { os << emitter.getOrCreateName(op.getVal()); });
}
LogicalResult printOperation(CppEmitter& emitter, util::BufferCreateOp op) {
   std::string type;
   if (failed(emitter.typeToString(op->getLoc(), mlir::cast<util::RefType>(op.getType().getElementType()).getElementType(), type))) {
      return failure();
   }
   return printStandardOperation(emitter, op, [&](auto& os) { os << "runtime::Buffer{" << emitter.getOrCreateName(op.getLen()) << "*sizeof(" << type << "),(uint8_t*)" << emitter.getOrCreateName(op.getPtr()) << "}"; });
}
LogicalResult printOperation(CppEmitter& emitter, util::BufferGetLen op) {
   std::string type;
   if (failed(emitter.typeToString(op->getLoc(), mlir::cast<util::RefType>(op.getBuffer().getType().getElementType()).getElementType(), type))) {
      return failure();
   }
   return printStandardOperation(emitter, op, [&](auto& os) { os << emitter.getOrCreateName(op.getBuffer()) << ".numElements/std::max(1ul,sizeof(" << type << "))"; });
}
LogicalResult printOperation(CppEmitter& emitter, util::BufferGetRef op) {
   std::string type;
   if (failed(emitter.typeToString(op->getLoc(), op.getBuffer().getType().getElementType(), type))) {
      return failure();
   }
   return printStandardOperation(emitter, op, [&](auto& os) { os << "(" << type << ")" << emitter.getOrCreateName(op.getBuffer()) << ".ptr"; });
}
LogicalResult printOperation(CppEmitter& emitter, util::InvalidRefOp op) {
   return printStandardOperation(emitter, op, [&](auto& os) { os << "nullptr"; });
}
LogicalResult printOperation(CppEmitter& emitter, util::IsRefValidOp op) {
   return printStandardOperation(emitter, op, [&](auto& os) { os << emitter.getOrCreateName(op.getRef()) << "!=nullptr"; });
}
LogicalResult printOperation(CppEmitter& emitter, util::SizeOfOp op) {
   std::string type;
   if (failed(emitter.typeToString(op->getLoc(), op.getType(), type))) {
      return failure();
   }
   return printStandardOperation(emitter, op, [&](auto& os) { os << "sizeof(" << type << ")"; });
}
LogicalResult printOperation(CppEmitter& emitter, util::ArrayElementPtrOp op) {
   return printStandardOperation(emitter, op, [&](auto& os) { os << "&" << emitter.getOrCreateName(op.getRef()) << "[" << emitter.getOrCreateName(op.getIdx()) << "]"; });
}
LogicalResult printOperation(CppEmitter& emitter, util::LoadOp op) {
   if (op.getIdx()) {
      return printStandardOperation(emitter, op, [&](auto& os) { os << emitter.getOrCreateName(op.getRef()) << "[" << emitter.getOrCreateName(op.getIdx()) << "]"; });
   } else {
      return printStandardOperation(emitter, op, [&](auto& os) { os << "*" << emitter.getOrCreateName(op.getRef()); });
   }
}
LogicalResult printOperation(CppEmitter& emitter, util::StoreOp op) {
   raw_ostream& os = emitter.ostream();
   if (op.getIdx()) {
      os << emitter.getOrCreateName(op.getRef()) << "[" << emitter.getOrCreateName(op.getIdx()) << "] = " << emitter.getOrCreateName(op.getVal());
   } else {
      os << "*" << emitter.getOrCreateName(op.getRef()) << " = " << emitter.getOrCreateName(op.getVal());
   }
   return success();
}

LogicalResult printOperation(CppEmitter& emitter,
                             arith::ConstantOp constantOp) {
   Operation* operation = constantOp.getOperation();
   Attribute value = constantOp.getValue();

   return printConstantOp(emitter, operation, value);
}

LogicalResult printOperation(CppEmitter& emitter,
                             func::ConstantOp constantOp) {
   Operation* operation = constantOp.getOperation();
   Attribute value = constantOp.getValueAttr();

   return printConstantOp(emitter, operation, value);
}

LogicalResult printOperation(CppEmitter& emitter, func::CallOp callOp) {
   if (callOp->getNumResults() <= 1) {
      if (failed(emitter.emitAssignPrefix(*callOp.getOperation())))
         return failure();

      raw_ostream& os = emitter.ostream();
      os << callOp.getCallee() << "(";
      if (failed(emitter.emitOperands(*callOp.getOperation())))
         return failure();
      os << ")";
      return success();
   } else {
      raw_ostream& os = emitter.ostream();
      auto tmpVar = emitter.getVariableName();
      os << "auto " << tmpVar << " = " << callOp.getCallee() << "(";
      if (failed(emitter.emitOperands(*callOp.getOperation())))
         return failure();
      os << ");\n";
      size_t offset = 0;
      for (auto r : callOp->getResults()) {
         if (emitter.emitType(callOp->getLoc(), r.getType()).failed()) {
            return failure();
         }
         os << " " << emitter.getOrCreateName(r) << " = " << tmpVar << ".t" << offset++ << ";\n";
      }
      return success();
   }
}

LogicalResult printOperation(CppEmitter& emitter, scf::ForOp forOp) {
   raw_indented_ostream& os = emitter.ostream();

   OperandRange operands = forOp.getInitArgs();
   Block::BlockArgListType iterArgs = forOp.getRegionIterArgs();
   Operation::result_range results = forOp.getResults();

   if (!emitter.shouldDeclareVariablesAtTop()) {
      for (OpResult result : results) {
         if (failed(emitter.emitVariableDeclaration(result,
                                                    /*trailingSemicolon=*/true)))
            return failure();
      }
   }

   for (auto pair : llvm::zip(iterArgs, operands)) {
      if (failed(emitter.emitType(forOp.getLoc(), std::get<0>(pair).getType())))
         return failure();
      os << " " << emitter.getOrCreateName(std::get<0>(pair)) << " = ";
      os << emitter.getOrCreateName(std::get<1>(pair)) << ";";
      os << "\n";
   }

   os << "for (";
   if (failed(
          emitter.emitType(forOp.getLoc(), forOp.getInductionVar().getType())))
      return failure();
   os << " ";
   os << emitter.getOrCreateName(forOp.getInductionVar());
   os << " = ";
   os << emitter.getOrCreateName(forOp.getLowerBound());
   os << "; ";
   os << emitter.getOrCreateName(forOp.getInductionVar());
   os << " < ";
   os << emitter.getOrCreateName(forOp.getUpperBound());
   os << "; ";
   os << emitter.getOrCreateName(forOp.getInductionVar());
   os << " += ";
   os << emitter.getOrCreateName(forOp.getStep());
   os << ") {\n";
   os.indent();

   Region& forRegion = forOp.getRegion();
   auto regionOps = forRegion.getOps();

   // We skip the trailing yield op because this updates the result variables
   // of the for op in the generated code. Instead we update the iterArgs at
   // the end of a loop iteration and set the result variables after the for
   // loop.
   for (auto it = regionOps.begin(); std::next(it) != regionOps.end(); ++it) {
      if (failed(emitter.emitOperation(*it, /*trailingSemicolon=*/true)))
         return failure();
   }

   Operation* yieldOp = forRegion.getBlocks().front().getTerminator();
   // Copy yield operands into iterArgs at the end of a loop iteration.
   for (auto pair : llvm::zip(iterArgs, yieldOp->getOperands())) {
      BlockArgument iterArg = std::get<0>(pair);
      Value operand = std::get<1>(pair);
      os << emitter.getOrCreateName(iterArg) << " = "
         << emitter.getOrCreateName(operand) << ";\n";
   }

   os.unindent() << "}";

   // Copy iterArgs into results after the for loop.
   for (auto pair : llvm::zip(results, iterArgs)) {
      OpResult result = std::get<0>(pair);
      BlockArgument iterArg = std::get<1>(pair);
      os << "\n"
         << emitter.getOrCreateName(result) << " = "
         << emitter.getOrCreateName(iterArg) << ";";
   }

   return success();
}

LogicalResult printOperation(CppEmitter& emitter, scf::WhileOp whileOp) {
   raw_indented_ostream& os = emitter.ostream();

   OperandRange operands = whileOp->getOperands();
   Block::BlockArgListType beforeArgs = whileOp.getBeforeArguments();
   Block::BlockArgListType afterArgs = whileOp.getAfterArguments();
   Operation::result_range results = whileOp.getResults();

   if (!emitter.shouldDeclareVariablesAtTop()) {
      for (OpResult result : results) {
         if (failed(emitter.emitVariableDeclaration(result,
                                                    /*trailingSemicolon=*/true)))
            return failure();
      }
   }

   for (auto pair : llvm::zip(beforeArgs, operands)) {
      if (failed(emitter.emitType(whileOp.getLoc(), std::get<0>(pair).getType())))
         return failure();
      os << " " << emitter.getOrCreateName(std::get<0>(pair)) << " = ";
      os << emitter.getOrCreateName(std::get<1>(pair)) << ";";
      os << "\n";
   }

   os << "while(true){\n";
   os.indent();
   Region& beforeRegion = whileOp.getBefore();
   Region& afterRegion = whileOp.getAfter();
   auto beforeRegionOps = beforeRegion.getOps();
   auto afterRegionOps = afterRegion.getOps();

   // We skip the trailing yield op because this updates the result variables
   // of the for op in the generated code. Instead we update the iterArgs at
   // the end of a loop iteration and set the result variables after the for
   // loop.
   for (auto it = beforeRegionOps.begin();
        std::next(it) != beforeRegionOps.end();
        ++it) {
      if (failed(emitter.emitOperation(*it, /*trailingSemicolon=*/true)))
         return failure();
   }

   auto conditionOp = mlir::cast<mlir::scf::ConditionOp>(
      beforeRegion.getBlocks().front().getTerminator());
   // Copy yield operands into iterArgs at the end of a loop iteration.
   for (auto pair : llvm::zip(afterArgs, conditionOp.getArgs())) {
      BlockArgument iterArg = std::get<0>(pair);
      Value operand = std::get<1>(pair);
      if (failed(emitter.emitType(whileOp.getLoc(), iterArg.getType())))
         return failure();
      os << " " << emitter.getOrCreateName(iterArg) << " = "
         << emitter.getOrCreateName(operand) << ";\n";
   }
   for (auto pair : llvm::zip(results, conditionOp.getArgs())) {
      OpResult result = std::get<0>(pair);
      Value operand = std::get<1>(pair);
      os << "\n"
         << emitter.getOrCreateName(result) << " = "
         << emitter.getOrCreateName(operand) << ";";
   }
   os << " if(!" << emitter.getOrCreateName(conditionOp.getCondition())
      << "){ break; }\n";
   for (auto it = afterRegionOps.begin(); std::next(it) != afterRegionOps.end();
        ++it) {
      if (failed(emitter.emitOperation(*it, /*trailingSemicolon=*/true)))
         return failure();
   }
   auto yieldOp = mlir::cast<mlir::scf::YieldOp>(
      afterRegion.getBlocks().front().getTerminator());

   for (auto pair : llvm::zip(beforeArgs, yieldOp.getResults())) {
      BlockArgument iterArg = std::get<0>(pair);
      Value operand = std::get<1>(pair);
      os << emitter.getOrCreateName(iterArg) << " = "
         << emitter.getOrCreateName(operand) << ";\n";
   }

   os.unindent() << "}";

   return success();
}

LogicalResult printOperation(CppEmitter& emitter, scf::IfOp ifOp) {
   raw_indented_ostream& os = emitter.ostream();

   if (!emitter.shouldDeclareVariablesAtTop()) {
      for (OpResult result : ifOp.getResults()) {
         if (failed(emitter.emitVariableDeclaration(result,
                                                    /*trailingSemicolon=*/true)))
            return failure();
      }
   }

   os << "if (";
   if (failed(emitter.emitOperands(*ifOp.getOperation())))
      return failure();
   os << ") {\n";
   os.indent();

   Region& thenRegion = ifOp.getThenRegion();
   for (Operation& op : thenRegion.getOps()) {
      // Note: This prints a superfluous semicolon if the terminating yield op has
      // zero results.
      if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/true)))
         return failure();
   }

   os.unindent() << "}";

   Region& elseRegion = ifOp.getElseRegion();
   if (!elseRegion.empty()) {
      os << " else {\n";
      os.indent();

      for (Operation& op : elseRegion.getOps()) {
         // Note: This prints a superfluous semicolon if the terminating yield op
         // has zero results.
         if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/true)))
            return failure();
      }

      os.unindent() << "}";
   }

   return success();
}

LogicalResult printOperation(CppEmitter& emitter, scf::YieldOp yieldOp) {
   raw_ostream& os = emitter.ostream();
   Operation& parentOp = *yieldOp.getOperation()->getParentOp();

   if (yieldOp.getNumOperands() != parentOp.getNumResults()) {
      return yieldOp.emitError("number of operands does not to match the number "
                               "of the parent op's results");
   }

   if (failed(interleaveWithError(
          llvm::zip(parentOp.getResults(), yieldOp.getOperands()),
          [&](auto pair) -> LogicalResult {
             auto result = std::get<0>(pair);
             auto operand = std::get<1>(pair);
             os << emitter.getOrCreateName(result) << " = ";

             if (!emitter.hasValueInScope(operand))
                return yieldOp.emitError("operand value not in scope");
             os << emitter.getOrCreateName(operand);
             return success();
          },
          [&]() { os << ";\n"; })))
      return failure();

   return success();
}

LogicalResult printOperation(CppEmitter& emitter,
                             func::ReturnOp returnOp) {
   raw_ostream& os = emitter.ostream();
   os << "return";
   switch (returnOp.getNumOperands()) {
      case 0:
         return success();
      case 1:
         os << " " << emitter.getOrCreateName(returnOp.getOperand(0));
         return success(emitter.hasValueInScope(returnOp.getOperand(0)));
      default:
         os << " {";
         if (failed(emitter.emitOperandsAndAttributes(*returnOp.getOperation())))
            return failure();
         os << "}";
         return success();
   }
}

LogicalResult printOperation(CppEmitter& emitter, ModuleOp moduleOp) {
   CppEmitter::Scope scope(emitter);

   for (Operation& op : moduleOp) {
      if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/false)))
         return failure();
   }
   return success();
}

LogicalResult printOperation(CppEmitter& emitter,
                             func::FuncOp functionOp) {
   if (functionOp.getFunctionBody().empty()) {
      raw_indented_ostream& os = emitter.ostream();

      if (failed(emitter.emitTypes(functionOp.getLoc(),
                                   functionOp.getFunctionType().getResults())))
         return failure();
      os << "( *" << functionOp.getName();
      os << ")(";
      if (failed(interleaveCommaWithError(
             functionOp.getFunctionType().getInputs(), os,
             [&](mlir::Type argType) -> LogicalResult {
                if (failed(emitter.emitType(functionOp.getLoc(), argType)))
                   return failure();
                return success();
             })))
         return failure();
      os << ")=(";
      if (emitter.emitType(functionOp->getLoc(), functionOp.getFunctionType(), os).failed()) {
         return failure();
      }
      os << ")" << emitter.resolveFunction(functionOp.getName().str()) << ";";
      /*// if (functionOp.getName().starts_with("_Z")) {
      os << "asm(\"" << functionOp.getName() << "\")";
      //}
      os << ";";*/
      return success();
   }
   // We need to declare variables at top if the function has multiple blocks.
   if (!emitter.shouldDeclareVariablesAtTop() &&
       functionOp.getBlocks().size() > 1) {
      return functionOp.emitOpError(
         "with multiple blocks needs variables declared at top");
   }

   CppEmitter::Scope scope(emitter);
   raw_indented_ostream& os = emitter.ostream();
   if (failed(emitter.emitTypes(functionOp.getLoc(),
                                functionOp.getFunctionType().getResults())))
      return failure();
   os << " " << functionOp.getName();

   os << "(";
   if (failed(interleaveCommaWithError(
          functionOp.getArguments(), os,
          [&](BlockArgument arg) -> LogicalResult {
             if (failed(emitter.emitType(functionOp.getLoc(), arg.getType())))
                return failure();
             os << " " << emitter.getOrCreateName(arg);
             return success();
          })))
      return failure();
   os << ") {\n";
   os.indent();
   if (emitter.shouldDeclareVariablesAtTop()) {
      // Declare all variables that hold op results including those from nested
      // regions.
      WalkResult result =
         functionOp.walk<WalkOrder::PreOrder>([&](Operation* op) -> WalkResult {
            for (OpResult result : op->getResults()) {
               if (failed(emitter.emitVariableDeclaration(
                      result, /*trailingSemicolon=*/true))) {
                  return WalkResult(
                     op->emitError("unable to declare result variable for op"));
               }
            }
            return WalkResult::advance();
         });
      if (result.wasInterrupted())
         return failure();
   }

   Region::BlockListType& blocks = functionOp.getBlocks();
   // Create label names for basic blocks.
   for (Block& block : blocks) {
      emitter.getOrCreateName(block);
   }

   // Declare variables for basic block arguments.
   for (Block& block : llvm::drop_begin(blocks)) {
      for (BlockArgument& arg : block.getArguments()) {
         if (emitter.hasValueInScope(arg))
            return functionOp.emitOpError(" block argument #")
               << arg.getArgNumber() << " is out of scope";
         if (failed(
                emitter.emitType(block.getParentOp()->getLoc(), arg.getType()))) {
            return failure();
         }
         os << " " << emitter.getOrCreateName(arg) << ";\n";
      }
   }

   for (Block& block : blocks) {
      // Only print a label if the block has predecessors.
      if (!block.hasNoPredecessors()) {
         if (failed(emitter.emitLabel(block)))
            return failure();
      }
      for (Operation& op : block.getOperations()) {
         // When generating code for an scf.if or cf.cond_br op no semicolon needs
         // to be printed after the closing brace.
         // When generating code for an scf.for op, printing a trailing semicolon
         // is handled within the printOperation function.
         bool trailingSemicolon =
            !isa<scf::IfOp, scf::ForOp, cf::CondBranchOp>(op);

         if (failed(emitter.emitOperation(
                op, /*trailingSemicolon=*/trailingSemicolon)))
            return failure();
      }
   }
   os.unindent() << "}\n";
   return success();
}
} // namespace

CppEmitter::CppEmitter(raw_ostream& os, bool declareVariablesAtTop)
   : os(os), declareVariablesAtTop(declareVariablesAtTop) {
   valueInScopeCount.push(0);
   labelInScopeCount.push(0);
   util::FunctionHelper::visitAllFunctions([&](std::string s, void* ptr) {
      symbolMap.insert({s, ptr});
   });
   lingodb::execution::visitBareFunctions([&](std::string s, void* ptr) {
      symbolMap.insert({s, ptr});
   });
}

/// Return the existing or a new name for a Value.
StringRef CppEmitter::getOrCreateName(Value val) {
   if (!valueMapper.count(val))
      valueMapper.insert(val, formatv("v{0}", ++valueInScopeCount.top()));
   return *valueMapper.begin(val);
}
std::string CppEmitter::getVariableName() {
   return formatv("v{0}", ++valueInScopeCount.top());
}

/// Return the existing or a new label for a Block.
StringRef CppEmitter::getOrCreateName(Block& block) {
   if (!blockMapper.count(&block))
      blockMapper.insert(&block, formatv("label{0}", ++labelInScopeCount.top()));
   return *blockMapper.begin(&block);
}

bool CppEmitter::shouldMapToUnsigned(IntegerType::SignednessSemantics val) {
   switch (val) {
      case IntegerType::Signless:
         return false;
      case IntegerType::Signed:
         return false;
      case IntegerType::Unsigned:
         return true;
   }
   llvm_unreachable("Unexpected IntegerType::SignednessSemantics");
}

bool CppEmitter::hasValueInScope(Value val) { return valueMapper.count(val); }

bool CppEmitter::hasBlockLabel(Block& block) {
   return blockMapper.count(&block);
}

LogicalResult CppEmitter::emitAttribute(Location loc, Attribute attr) {
   auto printInt = [&](const APInt& val, bool isUnsigned) {
      if (val.getBitWidth() == 1) {
         if (val.getBoolValue())
            os << "true";
         else
            os << "false";
      } else if (val.getBitWidth() == 128) {
         auto low = val.getLoBits(64).getLimitedValue();
         auto high = val.getHiBits(64).getLimitedValue();
         os << "(static_cast<__int128>(" << high << ")<<64)|static_cast<__int128>("
            << low << "ull)";
      } else {
         SmallString<128> strValue;
         val.toString(strValue, 10, !isUnsigned, true);
         os << strValue;
      }
   };

   auto printFloat = [&](const APFloat& val) {
      if (val.isFinite()) {
         SmallString<128> strValue;
         // Use default values of toString except don't truncate zeros.
         val.toString(strValue, 0, 0, false);
         switch (llvm::APFloatBase::SemanticsToEnum(val.getSemantics())) {
            case llvm::APFloatBase::S_IEEEsingle:
               os << "(float)";
               break;
            case llvm::APFloatBase::S_IEEEdouble:
               os << "(double)";
               break;
            default:
               break;
         };
         os << strValue;
      } else if (val.isNaN()) {
         os << "NAN";
      } else if (val.isInfinity()) {
         if (val.isNegative())
            os << "-";
         os << "INFINITY";
      }
   };

   // Print floating point attributes.
   if (auto fAttr = mlir::dyn_cast<FloatAttr>(attr)) {
      printFloat(fAttr.getValue());
      return success();
   }
   if (auto dense = mlir::dyn_cast<DenseFPElementsAttr>(attr)) {
      os << '{';
      interleaveComma(dense, os, [&](const APFloat& val) { printFloat(val); });
      os << '}';
      return success();
   }

   // Print integer attributes.
   if (auto iAttr = mlir::dyn_cast<IntegerAttr>(attr)) {
      if (auto iType = mlir::dyn_cast<IntegerType>(iAttr.getType())) {
         printInt(iAttr.getValue(), shouldMapToUnsigned(iType.getSignedness()));
         return success();
      }
      if (auto iType = mlir::dyn_cast<IndexType>(iAttr.getType())) {
         printInt(iAttr.getValue(), false);
         return success();
      }
   }
   if (auto dense = mlir::dyn_cast<DenseIntElementsAttr>(attr)) {
      if (auto iType = mlir::dyn_cast<IntegerType>(mlir::cast<TensorType>(dense.getType()).getElementType())) {
         os << '{';
         interleaveComma(dense, os, [&](const APInt& val) {
            printInt(val, shouldMapToUnsigned(iType.getSignedness()));
         });
         os << '}';
         return success();
      }
      if (auto iType = mlir::dyn_cast<IndexType>(mlir::cast<TensorType>(dense.getType()).getElementType())) {
         os << '{';
         interleaveComma(dense, os,
                         [&](const APInt& val) { printInt(val, false); });
         os << '}';
         return success();
      }
   }

   // Print symbolic reference attributes.
   if (auto sAttr = mlir::dyn_cast<SymbolRefAttr>(attr)) {
      if (sAttr.getNestedReferences().size() > 1)
         return emitError(loc, "attribute has more than 1 nested reference");
      os << sAttr.getRootReference().getValue();
      return success();
   }

   // Print type attributes.
   if (auto type = mlir::dyn_cast<TypeAttr>(attr))
      return emitType(loc, type.getValue());

   return emitError(loc, "cannot emit attribute: ") << attr;
}

LogicalResult CppEmitter::emitOperands(Operation& op) {
   auto emitOperandName = [&](Value result) -> LogicalResult {
      if (!hasValueInScope(result))
         return op.emitOpError() << "operand value not in scope";
      os << getOrCreateName(result);
      return success();
   };
   return interleaveCommaWithError(op.getOperands(), os, emitOperandName);
}

LogicalResult
CppEmitter::emitOperandsAndAttributes(Operation& op,
                                      ArrayRef<StringRef>
                                         exclude) {
   if (failed(emitOperands(op)))
      return failure();
   // Insert comma in between operands and non-filtered attributes if needed.
   if (op.getNumOperands() > 0) {
      for (NamedAttribute attr : op.getAttrs()) {
         if (!llvm::is_contained(exclude, attr.getName().strref())) {
            os << ", ";
            break;
         }
      }
   }
   // Emit attributes.
   auto emitNamedAttribute = [&](NamedAttribute attr) -> LogicalResult {
      if (llvm::is_contained(exclude, attr.getName().strref()))
         return success();
      os << "/* " << attr.getName().getValue() << " */";
      if (failed(emitAttribute(op.getLoc(), attr.getValue())))
         return failure();
      return success();
   };
   return interleaveCommaWithError(op.getAttrs(), os, emitNamedAttribute);
}

LogicalResult CppEmitter::emitVariableAssignment(OpResult result) {
   if (!hasValueInScope(result)) {
      return result.getDefiningOp()->emitOpError(
         "result variable for the operation has not been declared");
   }
   os << getOrCreateName(result) << " = ";
   return success();
}

LogicalResult CppEmitter::emitVariableDeclaration(OpResult result,
                                                  bool trailingSemicolon) {
   if (hasValueInScope(result)) {
      return result.getDefiningOp()->emitError(
         "result variable for the operation already declared");
   }
   if (failed(emitType(result.getOwner()->getLoc(), result.getType())))
      return failure();
   os << " " << getOrCreateName(result);
   if (trailingSemicolon)
      os << ";\n";
   return success();
}

LogicalResult CppEmitter::emitAssignPrefix(Operation& op) {
   switch (op.getNumResults()) {
      case 0:
         break;
      case 1: {
         OpResult result = op.getResult(0);
         if (shouldDeclareVariablesAtTop()) {
            if (failed(emitVariableAssignment(result)))
               return failure();
         } else {
            if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/false)))
               return failure();
            os << " = ";
         }
         break;
      }
      default:
         if (!shouldDeclareVariablesAtTop()) {
            for (OpResult result : op.getResults()) {
               if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/true)))
                  return failure();
            }
         }
         os << "std::tie(";
         interleaveComma(op.getResults(), os,
                         [&](Value result) { os << getOrCreateName(result); });
         os << ") = ";
   }
   return success();
}

LogicalResult CppEmitter::emitLabel(Block& block) {
   if (!hasBlockLabel(block))
      return block.getParentOp()->emitError("label for block not found");
   // FIXME: Add feature in `raw_indented_ostream` to ignore indent for block
   // label instead of using `getOStream`.
   os.getOStream() << getOrCreateName(block) << ":\n";
   return success();
}
#define ArithmeticPrinter(Op, Type) .Case<arith::Op>([&](mlir::Operation* op) { return printArithmeticOperation(*this, op, ArithmeticCOperation::Type); })
LogicalResult CppEmitter::emitOperation(Operation& op, bool trailingSemicolon) {
   os << "/*";
   op.getName().print(os);
   os << "    ";
   op.getLoc().print(os);
   os << "*/\n";
   LogicalResult status =
      llvm::TypeSwitch<Operation*, LogicalResult>(&op)
         // Builtin ops.
         .Case<ModuleOp>([&](auto op) { return printOperation(*this, op); })
         // CF ops.
         //.Case<cf::BranchOp, cf::CondBranchOp>(
         //   [&](auto op) { return printOperation(*this, op); })
         // Func ops.
         .Case<func::CallOp, func::ConstantOp, func::FuncOp, func::ReturnOp>(
            [&](auto op) { return printOperation(*this, op); })
         // SCF ops.
         .Case<scf::ForOp, scf::WhileOp, scf::IfOp, scf::YieldOp>(
            [&](auto op) { return printOperation(*this, op); })
         // Arithmetic ops.
         .Case<arith::ConstantOp>(
            [&](auto op) { return printOperation(*this, op); })
      // clang-format off
            ArithmeticPrinter(AddFOp, ADD)
            ArithmeticPrinter(AddIOp, ADD)
            ArithmeticPrinter(AndIOp, AND)
            ArithmeticPrinter(CeilDivSIOp, CEIL_DIV)
            ArithmeticPrinter(CeilDivUIOp, CEIL_DIV)
            ArithmeticPrinter(DivFOp, DIVIDE)
            ArithmeticPrinter(DivSIOp, DIVIDE)
            ArithmeticPrinter(DivUIOp, DIVIDE)
            ArithmeticPrinter(FloorDivSIOp, FLOOR_DIV)
            ArithmeticPrinter(MaximumFOp, MAX)
            ArithmeticPrinter(MaxNumFOp, MAX)
            ArithmeticPrinter(MaxSIOp, MAX)
            ArithmeticPrinter(MaxUIOp, MAX_U)
            ArithmeticPrinter(MinimumFOp, MIN)
            ArithmeticPrinter(MinNumFOp, MIN)
            ArithmeticPrinter(MinSIOp, MIN)
            ArithmeticPrinter(MinUIOp, MIN_U)
            ArithmeticPrinter(MulFOp, MULTIPLY)
            ArithmeticPrinter(MulIOp, MULTIPLY)
            ArithmeticPrinter(NegFOp, NEGATE)
            ArithmeticPrinter(OrIOp, OR)
            ArithmeticPrinter(XOrIOp, XOR)
            ArithmeticPrinter(SelectOp, TERNARY_OP)
            ArithmeticPrinter(RemFOp, FMOD)
            ArithmeticPrinter(RemUIOp, MODULO_U)
            ArithmeticPrinter(ShLIOp, SHIFT_LEFT)
            ArithmeticPrinter(ShRSIOp, SHIFT_RIGHT)
            ArithmeticPrinter(ShRUIOp, SHIFT_RIGHT_U)
            ArithmeticPrinter(SubIOp, SUBTRACT)
            ArithmeticPrinter(SubFOp, SUBTRACT)
         // clang-format on

         .Case<arith::IndexCastOp, arith::IndexCastUIOp, arith::SIToFPOp, arith::UIToFPOp, arith::TruncIOp, arith::TruncFOp, arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp>([&](mlir::Operation* op) { return printSimpleCast(*this, op); })
         .Case<arith::CmpIOp>([&](auto op) {
            switch (op.getPredicate()) {
               case arith::CmpIPredicate::eq:
                  return printArithmeticOperation(*this, op, ArithmeticCOperation::COMPARE_EQUALS);
               case arith::CmpIPredicate::ne:
                  return printArithmeticOperation(*this, op, ArithmeticCOperation::COMPARE_NOT_EQUALS);
               case arith::CmpIPredicate::slt:
                  return printArithmeticOperation(*this, op, ArithmeticCOperation::COMPARE_LESS);
               case arith::CmpIPredicate::ult:
                  return printArithmeticOperation(*this, op, ArithmeticCOperation::COMPARE_LESS_U);
               case arith::CmpIPredicate::sle:
                  return printArithmeticOperation(*this, op, ArithmeticCOperation::COMPARE_LESS_EQ);
               case arith::CmpIPredicate::ule:
                  return printArithmeticOperation(*this, op, ArithmeticCOperation::COMPARE_LESS_EQ_U);
               case arith::CmpIPredicate::sgt:
                  return printArithmeticOperation(*this, op, ArithmeticCOperation::COMPARE_GREATER);
               case arith::CmpIPredicate::ugt:
                  return printArithmeticOperation(*this, op, ArithmeticCOperation::COMPARE_GREATER_U);
               case arith::CmpIPredicate::sge:
                  return printArithmeticOperation(*this, op, ArithmeticCOperation::COMPARE_GREATER_EQ);
               case arith::CmpIPredicate::uge:
                  return printArithmeticOperation(*this, op, ArithmeticCOperation::COMPARE_GREATER_EQ_U);
            }
         })
         .Case<arith::CmpFOp>([&](auto op) {
            switch (op.getPredicate()) {
               case arith::CmpFPredicate::OEQ:
               case arith::CmpFPredicate::UEQ:
                  return printArithmeticOperation(*this, op, ArithmeticCOperation::COMPARE_EQUALS);
               case arith::CmpFPredicate::ONE:
               case arith::CmpFPredicate::UNE:
                  return printArithmeticOperation(*this, op, ArithmeticCOperation::COMPARE_NOT_EQUALS);
               case arith::CmpFPredicate::OLT:
               case arith::CmpFPredicate::ULT:
                  return printArithmeticOperation(*this, op, ArithmeticCOperation::COMPARE_LESS);
               case arith::CmpFPredicate::OLE:
               case arith::CmpFPredicate::ULE:
                  return printArithmeticOperation(*this, op, ArithmeticCOperation::COMPARE_LESS_EQ);
               case arith::CmpFPredicate::OGT:
               case arith::CmpFPredicate::UGT:
                  return printArithmeticOperation(*this, op, ArithmeticCOperation::COMPARE_GREATER);
               case arith::CmpFPredicate::OGE:
               case arith::CmpFPredicate::UGE:
                  return printArithmeticOperation(*this, op, ArithmeticCOperation::COMPARE_GREATER_EQ);
               default:
                  return failure();
            }
         })
         // SCF ops.
         .Case<util::GenericMemrefCastOp, util::TupleElementPtrOp, util::ArrayElementPtrOp, util::LoadOp, util::StoreOp, util::AllocOp, util::AllocaOp, util::CreateConstVarLen, util::UndefOp, util::BufferCastOp, util::BufferCreateOp, util::DeAllocOp, util::InvalidRefOp, util::IsRefValidOp, util::SizeOfOp, util::PackOp, util::CreateVarLen, util::Hash64, util::HashCombine, util::HashVarLen, util::FilterTaggedPtr, util::UnTagPtr, util::BufferGetRef, util::BufferGetLen, util::VarLenCmp, util::VarLenGetLen, util::GetTupleOp, util::VarLenTryCheapHash>(
            [&](auto op) { return printOperation(*this, op); })
         .Case<util::ToMemrefOp, memref::AtomicRMWOp>(
            [&](auto op) { return printOperation(*this, op); })
         .Default([&](Operation*) {
            return op.emitOpError("unable to find printer for op");
         });

   if (failed(status))
      return failure();
   os << (trailingSemicolon ? ";\n" : "\n");
   return success();
}
LogicalResult CppEmitter::typeToString(Location loc, Type type, std::string& string) {
   llvm::raw_string_ostream sstream(string);
   if (emitType(loc, type, sstream).failed())
      return failure();
   return success();
}
LogicalResult CppEmitter::emitType(Location loc, Type type, llvm::raw_ostream& os) {
   if (auto iType = mlir::dyn_cast<IntegerType>(type)) {
      auto width = iType.getWidth();
      if (width == 24) {
         width = 32;
      }
      if (width == 40 || width == 48 || width == 56) {
         width = 64;
      }
      switch (width) {
         case 1:
            return (os << "bool"), success();
         case 8:
         case 16:
         case 32:
         case 64:
            if (shouldMapToUnsigned(iType.getSignedness()))
               return (os << "uint" << width << "_t"), success();
            else
               return (os << "int" << width << "_t"), success();
         case 128:
            return (os << "__int128"), success();
         default:
            return emitError(loc, "cannot emit integer type ") << type;
      }
   }
   if (auto fType = mlir::dyn_cast<FloatType>(type)) {
      switch (fType.getWidth()) {
         case 32:
            return (os << "float"), success();
         case 64:
            return (os << "double"), success();
         default:
            return emitError(loc, "cannot emit float type ") << type;
      }
   }
   if (auto iType = mlir::dyn_cast<IndexType>(type))
      return (os << "size_t"), success();
   if (auto tType = mlir::dyn_cast<TensorType>(type)) {
      if (!tType.hasRank())
         return emitError(loc, "cannot emit unranked tensor type");
      if (!tType.hasStaticShape())
         return emitError(loc, "cannot emit tensor type with non static shape");
      os << "Tensor<";
      if (failed(emitType(loc, tType.getElementType(), os)))
         return failure();
      auto shape = tType.getShape();
      for (auto dimSize : shape) {
         os << ", ";
         os << dimSize;
      }
      os << ">";
      return success();
   }
   if (auto tType = mlir::dyn_cast<TupleType>(type))
      return emitTupleType(loc, tType.getTypes(), os);
   if (auto funcType = mlir::dyn_cast<FunctionType>(type)) {
      os << "std::add_pointer<";
      if (failed(emitTypes(loc, funcType.getResults(), os)))
         return failure();

      os << "(";
      bool first = true;
      for (auto t : funcType.getInputs()) {
         if (first) {
            first = false;
         } else {
            os << ", ";
         }
         if (failed(emitType(loc, t, os)))
            return failure();
      }

      os << ")>::type";
      return success();
   }
   if (auto pType = mlir::dyn_cast<util::RefType>(type)) {
      if (failed(emitType(loc, pType.getElementType(), os)))
         return failure();
      os << "*";
      return success();
   }
   if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
      if (failed(emitType(loc, memrefType.getElementType(), os)))
         return failure();
      os << "*";
      return success();
   }
   if (auto vLType = mlir::dyn_cast<util::VarLen32Type>(type)) {
      os << "runtime::VarLen32";
      return success();
   }
   if (auto bufType = mlir::dyn_cast<util::BufferType>(type)) {
      os << "runtime::Buffer";
      return success();
   }
   return emitError(loc, "cannot emit type ") << type;
}

LogicalResult CppEmitter::emitTypes(Location loc, ArrayRef<Type> types, llvm::raw_ostream& os) {
   switch (types.size()) {
      case 0:
         os << "void";
         return success();
      case 1:
         return emitType(loc, types.front(), os);
      default:
         return emitTupleType(loc, types, os);
   }
}

LogicalResult CppEmitter::emitTupleType(Location loc, ArrayRef<Type> types,
                                        llvm::raw_ostream& os) {
   std::string structName = "tuple";
   for (size_t i = 0; i < types.size(); i++) {
      std::string typeStr;
      llvm::raw_string_ostream sstream(typeStr);
      if (emitType(loc, types[i], sstream).failed())
         return failure();
      structName += "__" + typeStr;
   }
   std::regex toReplace("\\*");
   std::regex toReplace2("::");

   structName = std::regex_replace(structName, toReplace, "ptr");
   structName = std::regex_replace(structName, toReplace2, "_");

   os << structName;
   if (cachedStructs.count(structName) == 0) {
      llvm::raw_string_ostream sstream(structDefs);

      sstream << "struct " << structName << " {";
      for (size_t i = 0; i < types.size(); i++) {
         if (emitType(loc, types[i], sstream).failed())
            return failure();
         sstream << " t" << i << ";";
      }
      sstream << "};\n";
      cachedStructs.insert(structName);
   }

   return success();
}

LogicalResult lingodb::execution::emitC(Operation* op, raw_ostream& os, bool declareVariablesAtTop) {
   std::string resultingCpp;
   llvm::raw_string_ostream sstream(resultingCpp);
   CppEmitter emitter(sstream, declareVariablesAtTop);
   if (emitter.emitOperation(*op, /*trailingSemicolon=*/false).failed()) {
      return failure();
   }
   resultingCpp = emitter.getStructDefs() + resultingCpp;
   os << resultingCpp;
   return success();
}