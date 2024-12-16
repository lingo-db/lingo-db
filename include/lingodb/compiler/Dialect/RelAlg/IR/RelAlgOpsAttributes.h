#ifndef LINGODB_COMPILER_DIALECT_RELALG_IR_RELALGOPSATTRIBUTES_H
#define LINGODB_COMPILER_DIALECT_RELALG_IR_RELALGOPSATTRIBUTES_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOpsEnums.h"
#include "lingodb/runtime/metadata.h"

#define GET_ATTRDEF_CLASSES
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOpsAttributes.h.inc"

#endif //LINGODB_COMPILER_DIALECT_RELALG_IR_RELALGOPSATTRIBUTES_H