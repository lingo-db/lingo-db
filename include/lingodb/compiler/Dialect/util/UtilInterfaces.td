#ifndef LINGODB_DIALECT_UTIL_INTERFACES
#define LINGODB_DIALECT_UTIL_INTERFACES
include "mlir/IR/OpBase.td"

def CollectionType : TypeInterface<"CollectionType"> {
  let cppNamespace = "::lingodb::compiler::dialect::util";
  let methods = [
     InterfaceMethod<
       /*description=*/"",
       /*retTy=*/"mlir::Type",
       /*methodName=*/"getElementType">
      ];
}

#endif// LINGODB_DIALECT_UTIL_INTERFACES