#ifndef LINGODB_DIALECT_TupleStream_IR_Base
#define LINGODB_DIALECT_TupleStream_IR_Base

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"


//===----------------------------------------------------------------------===//
// TupleStream dialect definition.
//===----------------------------------------------------------------------===//

def TupleStream_Dialect : Dialect {
    let name = "tuples";
    let summary = "A general dialect that adds a tuple strem type";
    let cppNamespace = "::lingodb::compiler::dialect::tuples";

    let extraClassDeclaration = [{
        private:
            ColumnManager columnManager;
        public:
            ColumnManager& getColumnManager(){
                return columnManager;
            }
    }];
    let useDefaultTypePrinterParser=1;
    let useDefaultAttributePrinterParser=1;
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

class TupleStream_Type<string name, string typeMnemonic> : TypeDef<TupleStream_Dialect, name> {
    let mnemonic = typeMnemonic;
}

def TupleStream : TupleStream_Type<"TupleStream", "tuplestream"> {
    let summary = "tuple stream type";
}

def TupleStream_Tuple : TupleStream_Type<"Tuple", "tuple"> {
    let summary = "tuple type";
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

class TupleStream_Attr<string name> : AttrDef<TupleStream_Dialect, name>;

def ColumnDefAttr : TupleStream_Attr<"ColumnDef"> {
    let mnemonic = "columndef";
    let parameters = (ins "mlir::SymbolRefAttr":$name,"std::shared_ptr<Column>":$columnPtr,"mlir::Attribute":$fromExisting);
    let extraClassDeclaration = [{
        Column &getColumn() const { return *getColumnPtr();  }
    }];
    let hasCustomAssemblyFormat=1;
}
def ColumnRefAttr : TupleStream_Attr<"ColumnRef"> {
    let mnemonic = "columnref";
    let parameters = (ins "mlir::SymbolRefAttr":$name,"std::shared_ptr<Column>":$columnPtr);
    let extraClassDeclaration = [{
        Column &getColumn() const { return *getColumnPtr();  }
    }];
        let hasCustomAssemblyFormat=1;
}






#endif // LINGODB_DIALECT_TupleStream_IR_Base
