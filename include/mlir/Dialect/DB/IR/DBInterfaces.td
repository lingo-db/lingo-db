#ifndef DBInterfaces
#define DBInterfaces
include "mlir/IR/OpBase.td"
def DB_HandleNullInterface : OpInterface<"HandleNullInterface"> {
  let cppNamespace = "::mlir::db";

  let methods = [
    InterfaceMethod<
      /*description=*/"",
      /*retTy=*/"bool",
      /*methodName=*/"canHandleInvalidValues"
    >,    InterfaceMethod<
      /*description=*/"",
      /*retTy=*/"bool",
      /*methodName=*/"canHandleNulls",(ins ),"", /*defaultImplementation=*/[{
    return false;
}]
    >];
}

#endif// DBInterfaces
