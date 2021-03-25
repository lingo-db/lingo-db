#ifndef Interfaces
#define Interfaces
include "mlir/IR/OpBase.td"

def Operator : OpInterface<"Operator"> {
let methods = [
InterfaceMethod<
        "get all attributes that are used directly by this operator",
        "::llvm::SmallPtrSet<::mlir::relalg::RelationalAttribute*,8>", "getUsedAttributes",(ins ),"", /*defaultImplementation=*/[{
    return mlir::relalg::detail::getUsedAttributes(this->getOperation());
}]
>,
InterfaceMethod<
        "get all attributes that are available after this operator",
        "::llvm::SmallPtrSet<::mlir::relalg::RelationalAttribute*,8>", "getAvailableAttributes",(ins ),"", /*defaultImplementation=*/[{
return mlir::relalg::detail::getAvailableAttributes(this->getOperation());
}]
>,
InterfaceMethod<
        "get children of operator",
        "llvm::SmallVector<Operator,4>", "getChildren",(ins ),"", /*defaultImplementation=*/[{
    llvm::SmallVector<Operator,4> children;
    auto op=this->getOperation();
    for (auto operand : op->getOperands()) {
        mlir::Type t=operand.getType();
        bool isreltype=t.isa<mlir::relalg::RelationType>();
        auto childOp=operand.getDefiningOp();
        if (isreltype&&childOp) {
            auto childOperator=mlir::dyn_cast_or_null<Operator>(childOp);
            if(childOperator){
                    children.push_back(childOperator);
            }
        }
    }
    return children;
}]
>,
InterfaceMethod<
        "set children of operator",
        "void ", "setChildren",(ins "llvm::SmallVector<Operator,4>":$children),"", /*defaultImplementation=*/[{
    auto op=this->getOperation();
    size_t i=0;
    size_t j=0;
    for (auto operand : op->getOperands()) {
        mlir::Type t=operand.getType();
        bool isreltype=t.isa<mlir::relalg::RelationType>();
        auto childOp=operand.getDefiningOp();
        if (isreltype&&childOp) {
            op->setOperand(i,children[j].asRelation());
            j++;
        }
        i++;
    }
}]
>,
InterfaceMethod<
        "get resulting relation of operator",
        "mlir::Value", "asRelation",(ins ),"", /*defaultImplementation=*/[{
return this->getOperation()->getResult(0);
}]
>,
InterfaceMethod<
        "get all 'free' attributes",
        "llvm::SmallPtrSet<::mlir::relalg::RelationalAttribute*,8>", "getFreeAttributes",(ins ),"", /*defaultImplementation=*/[{
return mlir::relalg::detail::getFreeAttributes(this->getOperation());
}]
>,
InterfaceMethod<
        "get all attributes that are created by this operator",
        "llvm::SmallPtrSet<mlir::relalg::RelationalAttribute*,8>", "getCreatedAttributes",(ins ),"", /*defaultImplementation=*/[{
return mlir::relalg::detail::getCreatedAttributes(this->getOperation());
}]
>
];
}
def Join : OpInterface<"Join"> {
let methods = [
InterfaceMethod<
        "get all attributes that are used directly by this operator",
        "bool", "isDependentJoin",(ins ),"", /*defaultImplementation=*/[{
return mlir::relalg::detail::isDependentJoin(this->getOperation());
}]
>,
InterfaceMethod<
        "get all attributes that are used directly by this operator",
        "mlir::Operation*", "leftChild",(ins ),"", /*defaultImplementation=*/[{
return this->getOperation()->getOperand(0).getDefiningOp();
}]
>,
InterfaceMethod<
        "get all attributes that are used directly by this operator",
        "mlir::Operation*", "rightChild",(ins ),"", /*defaultImplementation=*/[{
return this->getOperation()->getOperand(1).getDefiningOp();
}]
>
];
}
def TupleLamdaOperator : OpInterface<"TupleLamdaOperator"> {
let methods = [
InterfaceMethod<
        "",
        "mlir::Region&", "getLambdaRegion",(ins ),""
        , /*op=*/[{
return this->getOperation()->getRegion(0);
}]
>,
InterfaceMethod<
        "",
        "mlir::Block&", "getLambdaBlock",(ins ),""
        , /*op=*/[{
return this->getOperation()->getRegion(0).front();
}]>,
InterfaceMethod<
        "",
        "mlir::BlockArgument", "getLambdaArgument",(ins ),""
        , /*op=*/[{
return this->getOperation()->getRegion(0).getArgument(0);
}]
>];
}

#endif// Interfaces
