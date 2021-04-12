#ifndef Interfaces
#define Interfaces
include "mlir/IR/OpBase.td"

def Operator : OpInterface<"Operator"> {
let methods = [
InterfaceMethod<
        "get all attributes that are used directly by this operator",
        "mlir::relalg::Attributes", "getUsedAttributes",(ins ),"", /*defaultImplementation=*/[{
    return mlir::relalg::detail::getUsedAttributes(this->getOperation());
}]
>,
InterfaceMethod<
        "get all attributes that are available after this operator",
        "mlir::relalg::Attributes", "getAvailableAttributes",(ins ),"", /*defaultImplementation=*/[{
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
        "get children of operator",
        "llvm::SmallVector<Operator,4>", "getAllSubOperators",(ins ),"", /*defaultImplementation=*/[{
    llvm::SmallVector<Operator,4> suboperators;
    suboperators.push_back(mlir::dyn_cast_or_null<Operator>(this->getOperation()));
    auto children=this->getChildren();
    for (auto child : children) {
        auto subsub=child.getAllSubOperators();
        suboperators.insert(suboperators.end(),subsub.begin(),subsub.end());
    }
    return suboperators;
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
        "mlir::relalg::Attributes", "getFreeAttributes",(ins ),"", /*defaultImplementation=*/[{
return mlir::relalg::detail::getFreeAttributes(this->getOperation());
}]
>,
InterfaceMethod<
        "get all attributes that are created by this operator",
        "mlir::relalg::Attributes", "getCreatedAttributes",(ins ),"", /*defaultImplementation=*/[{
return mlir::relalg::detail::getCreatedAttributes(this->getOperation());
}]
>
];
}
def BinaryOperator : OpInterface<"BinaryOperator"> {
let methods = [
InterfaceMethod<
        "get all attributes that are used directly by this operator",
        "mlir::Operation*", "leftChild",(ins ),"", /*defaultImplementation=*/[{
return this->getOperation()->getOperand(0).getDefiningOp();
}]
>,
InterfaceMethod<
        "assoc",
        "bool", "isAssoc",(ins "BinaryOperator":$binop),"", /*defaultImplementation=*/[{
return mlir::relalg::detail::binaryOperatorIs(mlir::relalg::detail::assoc,this->getOperation(),binop.getOperation());
}]
>,
InterfaceMethod<
"assoc",
"bool", "isLAsscom",(ins "BinaryOperator":$binop),"", /*defaultImplementation=*/[{
return mlir::relalg::detail::binaryOperatorIs(mlir::relalg::detail::lAsscom,this->getOperation(),binop.getOperation());
}]
>,
InterfaceMethod<
"assoc",
"bool", "isRAsscom",(ins "BinaryOperator":$binop),"", /*defaultImplementation=*/[{
return mlir::relalg::detail::binaryOperatorIs(mlir::relalg::detail::rAsscom,this->getOperation(),binop.getOperation());
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
def PredicateOperator : OpInterface<"PredicateOperator"> {
let methods = [
InterfaceMethod<
        "",
        "mlir::Region&", "getPredicateRegion",(ins ),""
        , /*op=*/[{
return this->getOperation()->getRegion(0);
}]
>,
InterfaceMethod<
        "",
        "mlir::Block&", "getPredicateBlock",(ins ),""
        , /*op=*/[{
return this->getOperation()->getRegion(0).front();
}]>,
InterfaceMethod<
        "",
        "void", "addPredicate",(ins "std::function<mlir::Value(mlir::Value, mlir::OpBuilder& builder)>":$producer),""
        , /*op=*/[{
mlir::relalg::detail::addPredicate(this->getOperation(),producer);
}]>,
InterfaceMethod<
        "",
        "mlir::BlockArgument", "getPredicateArgument",(ins ),""
        , /*op=*/[{
return this->getOperation()->getRegion(0).getArgument(0);
}]
>];
}

#endif// Interfaces
