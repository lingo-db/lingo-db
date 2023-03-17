#ifndef OPS
#define OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"


def Cranelift_Dialect : Dialect {
let name = "cranelift";
let cppNamespace = "::mlir::cranelift";
let extraClassDeclaration = [{
private:
void registerTypes();
public:
}];
}

class Cranelift_Op<string mnemonic, list<Trait> traits = []> :
Op<Cranelift_Dialect, mnemonic, traits>{

}

class Cranelift_Type<string name, string typeMnemonic> : TypeDef<Cranelift_Dialect, name> {
    let mnemonic = typeMnemonic;
}

def FuncOp : Cranelift_Op<"func", [
  AffineScope, IsolatedFromAbove, Symbol,FunctionOpInterface
]> {

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                           OptionalAttr<DictArrayAttr>:$arg_attrs,
                           OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);


  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
          /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];
}
def AllocaOp  : Cranelift_Op<"alloc_stack"> {
let arguments = (ins I32Attr:$size);
let results=(outs I64:$ref);
let assemblyFormat = "  $size attr-dict";
}

def ReturnOp : Cranelift_Op<"return", [Pure, HasParent<"FuncOp">,Terminator]> {


  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def CallOp : Cranelift_Op<"call"> {
  let summary = "call operation";
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);
  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def BranchOp : Cranelift_Op<"br", [Pure, Terminator]> {

  let arguments = (ins Variadic<AnyType>:$destOperands);
  let successors = (successor AnySuccessor:$dest);

  let assemblyFormat = [{
    $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// CondBranchOp
//===----------------------------------------------------------------------===//

def CondBranchOp : Cranelift_Op<"cond_br",
    [AttrSizedOperandSegments,Pure, Terminator]> {

  let arguments = (ins I1:$condition,
                       Variadic<AnyType>:$trueDestOperands,
                       Variadic<AnyType>:$falseDestOperands);
  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);

  let assemblyFormat = [{
    $condition `,`
    $trueDest (`(` $trueDestOperands^ `:` type($trueDestOperands) `)`)? `,`
    $falseDest (`(` $falseDestOperands^ `:` type($falseDestOperands) `)`)?
    attr-dict
  }];
}

def IConstOp : Cranelift_Op<"iconst"> {
  let arguments = (ins I64Attr:$value);
  let results = (outs SignlessIntegerLike:$res);
  let assemblyFormat = "attr-dict $value `:` type($res)";
}
def F32ConstOp : Cranelift_Op<"f32const"> {
  let arguments = (ins F32Attr:$value);
  let results = (outs F32:$res);
  let assemblyFormat = "attr-dict $value `:` type($res)";
}
def F64ConstOp : Cranelift_Op<"f64const"> {
  let arguments = (ins F64Attr:$value);
  let results = (outs F64:$res);
  let assemblyFormat = "attr-dict $value `:` type($res)";
}
def GlobalOp : Cranelift_Op<"global"> {
  let arguments = (ins StrAttr:$symbol_name,StrAttr:$value);
  let assemblyFormat = "attr-dict $symbol_name `=` $value";
}
def AddressOfOp : Cranelift_Op<"addressof"> {
  let arguments = (ins StrAttr:$symbol_name);
  let results = (outs I64:$ptr);
  let assemblyFormat = "attr-dict $symbol_name";
}

def SelectOp : Cranelift_Op<"select"> {
  let arguments = (ins I1:$condition,AnyType:$trueVal, AnyType:$falseVal);
  let results = (outs AnyType:$res);
  let assemblyFormat = "attr-dict $condition `?` $trueVal `:` type($trueVal) `,` $falseVal `:` type($falseVal) `->` type($res)";
}

def StoreOp  : Cranelift_Op<"store"> {
let arguments = (ins AnyType:$x,I64:$p);
let assemblyFormat = "$x `:` type($x) `,` $p attr-dict";
}
def LoadOp  : Cranelift_Op<"load"> {
let arguments = (ins I64:$p);
let results = (outs AnyType:$res);
let assemblyFormat = "$p `:` type($res) attr-dict";
}
def BSwap  : Cranelift_Op<"bswap"> {
let arguments = (ins I64:$x);
let results = (outs I64:$res);
let assemblyFormat = "$x attr-dict";
}



def Cranelift_AtomicRmwOpTypeAttr : I64EnumAttr<
    "AtomicRmwOpType", "",
    [
      I64EnumAttrCase<"Add", 0>,
      I64EnumAttrCase<"Sub", 1>,
      I64EnumAttrCase<"And", 2>,
      I64EnumAttrCase<"Nand", 3>,
      I64EnumAttrCase<"Or", 4>,
      I64EnumAttrCase<"Xor", 5>,
      I64EnumAttrCase<"Xchg", 6>,
      I64EnumAttrCase<"Umin", 7>,
      I64EnumAttrCase<"Umax", 8>,
      I64EnumAttrCase<"Smin", 9>,
      I64EnumAttrCase<"Smax", 10>,

    ]> {
  let cppNamespace = "::mlir::cranelift";
}


def Cranelift_ICmpPredicateAttr : I64EnumAttr<
    "ICmpPredicate", "",
    [
      I64EnumAttrCase<"eq", 0>,
      I64EnumAttrCase<"ne", 1>,
      I64EnumAttrCase<"slt", 2>,
      I64EnumAttrCase<"sge", 3>,
      I64EnumAttrCase<"sgt", 4>,
      I64EnumAttrCase<"sle", 5>,
      I64EnumAttrCase<"ult", 6>,
      I64EnumAttrCase<"uge", 7>,
      I64EnumAttrCase<"ugt", 8>,
      I64EnumAttrCase<"ule", 9>,
      I64EnumAttrCase<"of", 10>,
      I64EnumAttrCase<"nof", 11>,

    ]> {
  let cppNamespace = "::mlir::cranelift";
}
def Cranelift_ICmpOp : Cranelift_Op<"icmp"> {
  let arguments = (ins Cranelift_ICmpPredicateAttr:$predicate,
                       SignlessIntegerLike:$lhs,
                       SignlessIntegerLike:$rhs);
  let results = (outs I1:$res);
    let assemblyFormat = "$predicate $lhs `:` type($lhs) `,` $rhs  `:` type($rhs) attr-dict";

}

def Cranelift_FCmpPredicateAttr : I64EnumAttr<
    "FCmpPredicate", "",
    [
      I64EnumAttrCase<"ord", 0>,
      I64EnumAttrCase<"eq", 2>,
      I64EnumAttrCase<"one", 4>,
      I64EnumAttrCase<"lt", 6>,
      I64EnumAttrCase<"le", 7>,
      I64EnumAttrCase<"gt", 8>,
      I64EnumAttrCase<"ge", 9>,
      I64EnumAttrCase<"uno", 1>,
      I64EnumAttrCase<"ueq", 5>,
      I64EnumAttrCase<"ne", 3>,
      I64EnumAttrCase<"ult", 10>,
      I64EnumAttrCase<"ule", 11>,
      I64EnumAttrCase<"ugt", 12>,
      I64EnumAttrCase<"uge", 13>,
    ]> {
  let cppNamespace = "::mlir::cranelift";
}
def Cranelift_FCmpOp : Cranelift_Op<"fcmp"> {
  let arguments = (ins Cranelift_FCmpPredicateAttr:$predicate,
                       FloatLike:$lhs,
                       FloatLike:$rhs);
  let results = (outs I1:$res);
    let assemblyFormat = "$predicate $lhs `:` type($lhs) `,` $rhs  `:` type($rhs) attr-dict";

}
def Cranelift_AtomicRmwOp : Cranelift_Op<"atomic_rmw"> {
  let arguments = (ins Cranelift_AtomicRmwOpTypeAttr:$rmw_op,
                       I64:$p,
                       SignlessIntegerLike:$x);
  let results = (outs SignlessIntegerLike:$res);
    let assemblyFormat = "$rmw_op $p `,` $x  `:` type($x) `->` type($res) attr-dict";

}

class Cranelift_ArithmeticOp<string mnemonic, list<Trait> traits = []> :
    Cranelift_Op<mnemonic, traits # [SameOperandsAndResultType]>;
class Cranelift_Arith_BinaryOp<string mnemonic, list<Trait> traits = []> :
    Cranelift_ArithmeticOp<mnemonic, traits> {
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}
class Cranelift_Arith_IntBinaryOp<string mnemonic, list<Trait> traits = []> :
    Cranelift_Arith_BinaryOp<mnemonic, traits>,
    Arguments<(ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs)>,
    Results<(outs SignlessIntegerLike:$result)>;
class Cranelift_Arith_FloatBinaryOp<string mnemonic, list<Trait> traits = []> :
    Cranelift_Arith_BinaryOp<mnemonic, traits>,
    Arguments<(ins FloatLike:$lhs, FloatLike:$rhs)>,
    Results<(outs FloatLike:$result)>;
def FAddOp : Cranelift_Arith_FloatBinaryOp<"fadd">;
def FSubOp : Cranelift_Arith_FloatBinaryOp<"fsub">;
def FMulOp : Cranelift_Arith_FloatBinaryOp<"fmul">;
def FDivOp : Cranelift_Arith_FloatBinaryOp<"fdiv">;
def IAddOp : Cranelift_Arith_IntBinaryOp<"iadd">;
def ISubOp : Cranelift_Arith_IntBinaryOp<"isub">;
def IMulOp : Cranelift_Arith_IntBinaryOp<"imul">;
def UMulHiOp : Cranelift_Arith_IntBinaryOp<"umulhi">;
def UDivOp : Cranelift_Arith_IntBinaryOp<"udiv">;
def SDivOp : Cranelift_Arith_IntBinaryOp<"sdiv">;
def URemOp : Cranelift_Arith_IntBinaryOp<"urem">;
def SRemOp : Cranelift_Arith_IntBinaryOp<"srem">;
def IShlOp : Cranelift_Arith_IntBinaryOp<"ishl">;
def SShrOp : Cranelift_Arith_IntBinaryOp<"sshr">;
def UShrOp : Cranelift_Arith_IntBinaryOp<"ushr">;
def BOrOp : Cranelift_Arith_IntBinaryOp<"bor">;
def BXOrOp : Cranelift_Arith_IntBinaryOp<"bxor">;
def BAndOp : Cranelift_Arith_IntBinaryOp<"band">;


def UExtendOp : Cranelift_Op<"uextend"> {
  let arguments = (ins SignlessIntegerLike:$value);
  let results = (outs SignlessIntegerLike:$res);
  let assemblyFormat = "attr-dict $value `:` type($value) `->` type($res)";
}
def SExtendOp : Cranelift_Op<"sextend"> {
  let arguments = (ins SignlessIntegerLike:$value);
  let results = (outs SignlessIntegerLike:$res);
  let assemblyFormat = "attr-dict $value `:` type($value) `->` type($res)";
}
def SIToFP : Cranelift_Op<"sitofp"> {
  let arguments = (ins SignlessIntegerLike:$value);
  let results = (outs FloatLike:$res);
  let assemblyFormat = "attr-dict $value `:` type($value) `->` type($res)";
}
def UIToFP : Cranelift_Op<"uitofp"> {
  let arguments = (ins SignlessIntegerLike:$value);
  let results = (outs FloatLike:$res);
  let assemblyFormat = "attr-dict $value `:` type($value) `->` type($res)";
}
def FPromoteOp : Cranelift_Op<"fpromote"> {
  let arguments = (ins FloatLike:$value);
  let results = (outs FloatLike:$res);
  let assemblyFormat = "attr-dict $value `:` type($value) `->` type($res)";
}
def FDemoteOp : Cranelift_Op<"fdemote"> {
  let arguments = (ins FloatLike:$value);
  let results = (outs FloatLike:$res);
  let assemblyFormat = "attr-dict $value `:` type($value) `->` type($res)";
}
def IReduceOp : Cranelift_Op<"ireduce"> {
  let arguments = (ins SignlessIntegerLike:$value);
  let results = (outs SignlessIntegerLike:$res);
  let assemblyFormat = "attr-dict $value `:` type($value) `->` type($res)";
}

def IConcatOp : Cranelift_Op<"iconcat"> {
  let arguments = (ins SignlessIntegerLike:$lower,SignlessIntegerLike:$higher);
  let results = (outs SignlessIntegerLike:$res);
  let assemblyFormat = "attr-dict $lower `,` $higher  `:` type($lower) `,` type($higher) `->` type($res)";
}
def ISplitOp : Cranelift_Op<"isplit"> {
  let arguments = (ins SignlessIntegerLike:$val);
  let results = (outs SignlessIntegerLike:$lower,SignlessIntegerLike:$higher);
  let assemblyFormat = "attr-dict $val `:` type($val) `->` type($lower) `,` type($higher)";
}

def FuncAddrOp : Cranelift_Op<"func_addr"> {
  let arguments = (ins FlatSymbolRefAttr:$callee);
  let results = (outs I64:$res);
  let assemblyFormat = "attr-dict $callee";
}
#endif// OPS
