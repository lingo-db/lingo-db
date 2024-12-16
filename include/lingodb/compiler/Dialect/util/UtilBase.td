#ifndef LINGODB_DIALECT_Util_IR_Base
#define LINGODB_DIALECT_Util_IR_Base
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "lingodb/compiler/Dialect/util/UtilInterfaces.td"

def Util_Dialect : Dialect {
let name = "util";
let cppNamespace = "::lingodb::compiler::dialect::util";
let extraClassDeclaration = [{
private:
void registerTypes();
std::shared_ptr<FunctionHelper> functionHelper;
public:
FunctionHelper& getFunctionHelper(){
    return *functionHelper;
}
}];
    let useDefaultTypePrinterParser=1;
}


class Util_Type<string name, string typeMnemonic, list<Trait> traits = []> : TypeDef<Util_Dialect, name,traits> {
    let mnemonic = typeMnemonic;
}
def BufferType : Util_Type<"Buffer","buffer",[CollectionType]> {
    let summary = "buffer";
    let parameters = (ins "mlir::Type":$t);
    let assemblyFormat = "`<` $t `>`";
        let extraClassDeclaration = [{
            mlir::Type getElementType();
        }];
}

def RefType : Util_Type<"Ref","ref"> {
    let summary = "ref type";
    let parameters = (ins "mlir::Type":$elementType);
    let builders = [
        TypeBuilder<(ins), [{
            return $_get($_ctxt, mlir::IntegerType::get($_ctxt,8));
        }]>,
        TypeBuilderWithInferredContext<(ins "Type":$type), [{
            return $_get(type.getContext(), type);
        }]>,
    ];
    let builderCall = "$_builder.getType<" # dialect.cppNamespace # "::" # cppClassName # ">()";
    let hasCustomAssemblyFormat = 1;
}


def VarLen32Type : Util_Type<"VarLen32","varlen32"> {
    let summary = "type representing variable-length data up to 2^31 bytes";
}

#endif // LINGODB_DIALECT_Util_IR_Base
