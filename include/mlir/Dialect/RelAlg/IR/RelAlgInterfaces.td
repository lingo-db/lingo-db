#ifndef Interfaces
#define Interfaces
include "mlir/IR/OpBase.td"

def Operator : OpInterface<"Operator"> {
let methods = [
InterfaceMethod<
        "Get Attribute Usage",
        "std::vector<mlir::relalg::RelationalAttributeRefAttr>", "getAttributeUsage",(ins ),"", /*defaultImplementation=*/[{
    return mlir::relalg::detail::getUsages(this->getOperation());
}]
>
];
}
def TupleLamdaOperator : OpInterface<"TupleLamdaOperator"> {
let methods = [
InterfaceMethod<
        "Get Attribute Usage",
        "mlir::Region&", "getLambdaRegion",(ins )
        , /*op=*/[{
return $_op->getRegion(0);
}]
>,
InterfaceMethod<
        "",
        "mlir::Block&", "getLambdaBlock",(ins )
        , /*op=*/[{
return $_op->getRegion(0).front();
}]
>];
}
def AttributeCreating : OpInterface<"AttributeCreating"> {
let methods = [
InterfaceMethod<
        "Get Attribute Usage",
        "mlir::SymbolRefAttr", "getScopeName",(ins )
        , /*op=*/[{
return ::getScopeNameImpl($_op);
}]
>,
InterfaceMethod<
        "get Attribute Creation",
        "std::vector<mlir::relalg::RelationalAttributeDefAttr>", "getCreatedAttributes",(ins ),"", /*defaultImplementation=*/[{
            return mlir::relalg::detail::getCreations(this->getOperation());
}]
>,
];
}

#endif// Interfaces
