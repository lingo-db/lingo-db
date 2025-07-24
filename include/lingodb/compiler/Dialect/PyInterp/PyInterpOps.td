include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"

def PyInterp_Dialect : Dialect {
    let summary = "Dialect for interacting with an embedded Python interpreter";

    let name = "py_interp";
    let cppNamespace = "::lingodb::compiler::dialect::py_interp";
    let extraClassDeclaration = [{
        private:
        void registerTypes();

        public:
    }];
    let useDefaultTypePrinterParser=1;
}

class PyInterp_Op<string mnemonic, list<Trait> traits = []> : Op<PyInterp_Dialect, mnemonic, traits>{
}

class PyInterp_Type<string name, string typeMnemonic> : TypeDef<PyInterp_Dialect, name> {
    let mnemonic = typeMnemonic;
}
def PyObjectType : PyInterp_Type<"PyObject", "py_object"> {
    let summary = "Type representing a Python object";
}

def ImportOp : PyInterp_Op<"import"> {
	let arguments = (ins StrAttr:$name);
	let results = (outs PyObjectType:$result);
	let assemblyFormat = "$name attr-dict";
}
