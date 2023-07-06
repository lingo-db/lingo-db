#include "mlir/Dialect/cranelift/CraneliftOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/cranelift/CraneliftDialect.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

#include <unordered_set>
using namespace mlir;
using namespace mlir::cranelift;

mlir::ParseResult FuncOp::parse(OpAsmParser& parser, OperationState& result) {
   auto buildFuncType =
      [](Builder& builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string&) { return builder.getFunctionType(argTypes, results); };

   return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false, getFunctionTypeAttrName(result.name), buildFuncType, getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(OpAsmPrinter& p) {
   function_interface_impl::printFunctionOp(p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(), getArgAttrsAttrName(), getResAttrsAttrName());
}

#define GET_OP_CLASSES
#include "mlir/Dialect/cranelift/CraneliftOps.cpp.inc"
