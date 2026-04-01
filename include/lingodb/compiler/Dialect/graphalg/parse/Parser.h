#pragma once

#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>

namespace graphalg {

mlir::LogicalResult parse(llvm::StringRef program, mlir::ModuleOp moduleOp);

} // namespace graphalg
