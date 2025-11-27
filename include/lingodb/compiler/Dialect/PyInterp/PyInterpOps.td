include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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
    let hasConstantMaterializer = 1;
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

def CreateModule : PyInterp_Op<"create_module"> {
    let arguments = (ins StrAttr:$name, StrAttr:$code);
    let results = (outs PyObjectType:$module);
    let assemblyFormat = "$name `,` $code attr-dict";
}
def ConstStrPyObject : PyInterp_Op<"const_str_pyobject", [ConstantLike,Pure]> {
    let arguments = (ins AnyAttr:$value);
    let results = (outs PyObjectType:$obj);
    let assemblyFormat = "$value attr-dict";
     let hasFolder = 1;

}
def CastToPyObject : PyInterp_Op<"cast_to_pyobject"> {
    let arguments = (ins AnyType:$from);
    let results = (outs PyObjectType:$to);
    let assemblyFormat = "$from `:` type($from) attr-dict";
}
def CastFromPyObject : PyInterp_Op<"cast_from_pyobject"> {
    let arguments = (ins PyObjectType:$from);
    let results = (outs AnyType:$to);
    let assemblyFormat = "$from `->` type($to) attr-dict";
}


def Call : PyInterp_Op<"py_call"> {
    let description = [{
        TODO: currently we do a hacky thing: args and kwargs are mixed...
    }];
	let arguments = (ins PyObjectType:$on, Variadic<PyObjectType>:$args, ArrayAttr:$kwnames);
	let results = (outs PyObjectType:$result);
	let assemblyFormat = "$on `(` $args `)` $kwnames attr-dict";
}
def GetAttr : PyInterp_Op<"get_attr"> {
	let arguments = (ins PyObjectType:$on, StrAttr:$name);
	let results = (outs PyObjectType:$result);
	let assemblyFormat = "$on `->` $name attr-dict";
}
def GetAttr2 : PyInterp_Op<"get_attr2"> {
	let arguments = (ins PyObjectType:$on, PyObjectType:$name);
	let results = (outs PyObjectType:$result);
	let assemblyFormat = "$on `->` $name attr-dict";
}

def SetAttr : PyInterp_Op<"set_attr"> {
	let arguments = (ins StrAttr:$name, PyObjectType:$on, PyObjectType:$value);
	let assemblyFormat = "$on `->` $name `=` $value attr-dict";
}
def DecRef : PyInterp_Op<"dec_ref"> {
    let arguments = (ins PyObjectType:$obj);
    let assemblyFormat = "$obj attr-dict";
}
def IncRef : PyInterp_Op<"inc_ref"> {
    let arguments = (ins PyObjectType:$obj);
    let assemblyFormat = "$obj attr-dict";
}