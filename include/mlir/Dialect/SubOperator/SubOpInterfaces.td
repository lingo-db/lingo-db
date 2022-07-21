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

def State : TypeInterface<"State"> {
  let cppNamespace = "::mlir::subop";
  let methods = [
     InterfaceMethod<
       /*description=*/"",
       /*retTy=*/"StateMembersAttr",
       /*methodName=*/"getMembers">
      ];
}
def StateSupportingLookup : TypeInterface<"StateSupportingLookup"> {
  let cppNamespace = "::mlir::subop";
  let methods = [
     InterfaceMethod<
       /*description=*/"",
       /*retTy=*/"StateMembersAttr",
       /*methodName=*/"getValueMembers">
      ];
}


#endif// Interfaces
