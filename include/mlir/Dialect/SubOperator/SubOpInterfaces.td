#ifndef Interfaces
#define Interfaces
include "mlir/IR/OpBase.td"
def SubOperator : OpInterface<"SubOperator"> {
  let cppNamespace = "::mlir::subop";

  let methods = [
    InterfaceMethod<
      /*description=*/"",
      /*retTy=*/"std::vector<std::string>",
      /*methodName=*/"getWrittenMembers",(ins ),"",
       /*defaultImplementation=*/[{
           return {};
       }]
    >,
        InterfaceMethod<
          /*description=*/"",
          /*retTy=*/"std::vector<std::string>",
          /*methodName=*/"getReadMembers",(ins ),"",
           /*defaultImplementation=*/[{
               return {};
           }]
        >,
  ];

}

def StateUsingSubOperator : OpInterface<"StateUsingSubOperator"> {
  let cppNamespace = "::mlir::subop";

  let methods = [
    InterfaceMethod<
      /*description=*/"",
      /*retTy=*/"void",
      /*methodName=*/"updateStateType",(ins "mlir::subop::SubOpStateUsageTransformer&":$transformer,"mlir::Value":$state,"mlir::Type":$newType)
    >,
    InterfaceMethod<
      /*description=*/"",
      /*retTy=*/"void",
      /*methodName=*/"replaceColumns",(ins "mlir::subop::SubOpStateUsageTransformer&":$transformer,"mlir::tuples::Column*":$oldColumn,"mlir::tuples::Column*":$newColumn)
    >,
  ];

}
def LookupAbleState : TypeInterface<"LookupAbleState"> {
  let cppNamespace = "::mlir::subop";
  let methods = [
    InterfaceMethod</*description=*/"", /*retTy=*/"StateMembersAttr", /*methodName=*/"getKeyMembers">,
    InterfaceMethod</*description=*/"", /*retTy=*/"StateMembersAttr", /*methodName=*/"getValueMembers">
  ];
}
def State : TypeInterface<"State"> {
  let cppNamespace = "::mlir::subop";
  let methods = [
    InterfaceMethod</*description=*/"", /*retTy=*/"StateMembersAttr", /*methodName=*/"getMembers">,
    InterfaceMethod</*description=*/"", /*retTy=*/"bool", /*methodName=*/"canPerformMaterialize",(ins ),"", /*defaultImplementation=*/[{
        return false;
    }]>,
    InterfaceMethod</*description=*/"", /*retTy=*/"bool", /*methodName=*/"canPerformScan",(ins ),"", /*defaultImplementation=*/[{
        return true;
    }]>
  ];
}


def StateEntryReference : TypeInterface<"StateEntryReference"> {
  let cppNamespace = "::mlir::subop";
  let methods = [
     InterfaceMethod</*description=*/"", /*retTy=*/"bool", /*methodName=*/"isReadable">,
     InterfaceMethod</*description=*/"", /*retTy=*/"bool", /*methodName=*/"isWriteable">,
     InterfaceMethod</*description=*/"", /*retTy=*/"bool", /*methodName=*/"isStable">,
     InterfaceMethod</*description=*/"", /*retTy=*/"bool", /*methodName=*/"canBeOffset">,
     InterfaceMethod</*description=*/"", /*retTy=*/"StateMembersAttr", /*methodName=*/"getMembers">
  ];
}
def ColumnFoldable : OpInterface<"ColumnFoldable"> {
  let cppNamespace = "::mlir::subop";
    let methods = [
    InterfaceMethod<"","mlir::LogicalResult", "foldColumns",(ins "mlir::subop::ColumnFoldInfo&":$columnInfo)>,
    ];
}

def ReferenceProducer : OpInterface<"ReferenceProducer"> {
  let cppNamespace = "::mlir::subop";
    let methods = [
    InterfaceMethod<"","mlir::tuples::ColumnDefAttr", "getProducedReference",(ins )>,
    ];
}
#endif// Interfaces
