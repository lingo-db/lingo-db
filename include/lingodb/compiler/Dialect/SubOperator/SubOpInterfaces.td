#ifndef Interfaces
#define Interfaces
include "mlir/IR/OpBase.td"
def SubOperator : OpInterface<"SubOperator"> {
  let cppNamespace = "::lingodb::compiler::dialect::subop";

  let methods = [
    InterfaceMethod<
      /*description=*/"",
      /*retTy=*/"std::shared_ptr<::lingodb::compiler::dialect::subop::Members>",
      /*methodName=*/"getWrittenMembers",(ins ),"",
       /*defaultImplementation=*/[{
             return std::make_shared<::lingodb::compiler::dialect::subop::Members>();
       }]
    >,
        InterfaceMethod<
          /*description=*/"",
          /*retTy=*/"std::shared_ptr<::lingodb::compiler::dialect::subop::Members>",
          /*methodName=*/"getReadMembers",(ins ),"",
           /*defaultImplementation=*/[{
             return std::make_shared<::lingodb::compiler::dialect::subop::Members>();
           }]
        >,
		InterfaceMethod<
		  /*description=*/"",
		  /*retTy=*/"mlir::Operation*",
		  /*methodName=*/"cloneSubOp",(ins "mlir::OpBuilder&":$builder, "mlir::IRMapping&" :$mapping, "lingodb::compiler::dialect::subop::ColumnMapping&":$columnMapping),"",
		 /*defaultImplementation=*/[{
			 return builder.clone(*this->getOperation(), mapping);
		 }]
		>,
  ];

}

def StateUsingSubOperator : OpInterface<"StateUsingSubOperator"> {
  let cppNamespace = "::lingodb::compiler::dialect::subop";

  let methods = [
    InterfaceMethod<
      /*description=*/"",
      /*retTy=*/"void",
      /*methodName=*/"updateStateType",(ins "lingodb::compiler::dialect::subop::SubOpStateUsageTransformer&":$transformer,"mlir::Value":$state,"mlir::Type":$newType)
    >,
    InterfaceMethod<
      /*description=*/"",
      /*retTy=*/"void",
      /*methodName=*/"replaceColumns",(ins "lingodb::compiler::dialect::subop::SubOpStateUsageTransformer&":$transformer,"lingodb::compiler::dialect::tuples::Column*":$oldColumn,"lingodb::compiler::dialect::tuples::Column*":$newColumn)
    >,
  ];

}
def StateCreator : OpInterface<"StateCreator"> {
  let cppNamespace = "::lingodb::compiler::dialect::subop";

    let methods = [
      InterfaceMethod<
        /*description=*/"",
        /*retTy=*/"std::shared_ptr<::lingodb::compiler::dialect::subop::Members>",
        /*methodName=*/"getCreatedMembers",(ins ),"",
         /*defaultImplementation=*/[{
             return std::make_shared<::lingodb::compiler::dialect::subop::Members>();
         }]
      >
  ];

}

def LookupAbleState : TypeInterface<"LookupAbleState"> {
  let cppNamespace = "::lingodb::compiler::dialect::subop";
  let methods = [
    InterfaceMethod</*description=*/"", /*retTy=*/"lingodb::compiler::dialect::subop::StateMembersAttr", /*methodName=*/"getKeyMembers">,
    InterfaceMethod</*description=*/"", /*retTy=*/"lingodb::compiler::dialect::subop::StateMembersAttr", /*methodName=*/"getValueMembers">
  ];
}
def State : TypeInterface<"State"> {
  let cppNamespace = "::lingodb::compiler::dialect::subop";
  let methods = [
    InterfaceMethod</*description=*/"", /*retTy=*/"lingodb::compiler::dialect::subop::StateMembersAttr", /*methodName=*/"getMembers">,
    InterfaceMethod</*description=*/"", /*retTy=*/"bool", /*methodName=*/"canPerformMaterialize",(ins ),"", /*defaultImplementation=*/[{
        return false;
    }]>,
    InterfaceMethod</*description=*/"", /*retTy=*/"bool", /*methodName=*/"canPerformScan",(ins ),"", /*defaultImplementation=*/[{
        return true;
    }]>,
    InterfaceMethod</*description=*/"", /*retTy=*/"bool", /*methodName=*/"hasLock",(ins ), "", /*defaultImplementation=*/[{
                                                                                          return false;
    }]>,
  ];
}


def StateEntryReference : TypeInterface<"StateEntryReference"> {
  let cppNamespace = "::lingodb::compiler::dialect::subop";
  let methods = [
     InterfaceMethod</*description=*/"", /*retTy=*/"bool", /*methodName=*/"isReadable">,
     InterfaceMethod</*description=*/"", /*retTy=*/"bool", /*methodName=*/"isWriteable">,
     InterfaceMethod</*description=*/"", /*retTy=*/"bool", /*methodName=*/"isStable">,
     InterfaceMethod</*description=*/"", /*retTy=*/"bool", /*methodName=*/"canBeOffset">,
     InterfaceMethod</*description=*/"", /*retTy=*/"bool", /*methodName=*/"hasLock">,
     InterfaceMethod</*description=*/"", /*retTy=*/"lingodb::compiler::dialect::subop::StateMembersAttr", /*methodName=*/"getMembers">
  ];
}
def ColumnFoldable : OpInterface<"ColumnFoldable"> {
  let cppNamespace = "::lingodb::compiler::dialect::subop";
    let methods = [
    InterfaceMethod<"","mlir::LogicalResult", "foldColumns",(ins "lingodb::compiler::dialect::subop::ColumnMapping&":$columnInfo)>,
    ];
}

def ReferenceProducer : OpInterface<"ReferenceProducer"> {
  let cppNamespace = "::lingodb::compiler::dialect::subop";
    let methods = [
    InterfaceMethod<"","lingodb::compiler::dialect::tuples::ColumnDefAttr", "getProducedReference",(ins )>,
    ];
}

def ContainsNestedSubOps : OpInterface<"ContainsNestedSubOps"> {
  let cppNamespace = "::lingodb::compiler::dialect::subop";
	let methods = [
	InterfaceMethod<"","mlir::Block*", "getBody",(ins )>,
	];
}
#endif// Interfaces
