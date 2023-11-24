import lingodbbridge
from lingodbbridge.mlir import ir
from lingodbbridge.mlir.dialects import func,arith,scf,util,tuples,db,relalg
import lingodbbridge.mlir._mlir_libs.mlir_init as mlir_init

con=lingodbbridge.ext.in_memory()
context=ir.Context()

print()
print(ir.Location.unknown(context=context))
print(ir.Module.parse("""builtin.module {
}""",context=context))



context2=ir.Context()
mlir_init.init_context(context2)
with context2, ir.Location.unknown():
    module = ir.Module.create()
    i32 = ir.IntegerType.get_signless(32)
    with ir.InsertionPoint(module.body), ir.Location.unknown():
        f = func.FuncOp("main", ([], [i32]))
        with ir.InsertionPoint(f.body.blocks.append()):
            c0 = arith.ConstantOp(i32, 1)
            add = arith.AddIOp(c0, c0)
            cond = arith.ConstantOp(ir.IntegerType.get_signless(1), 1)
            if_op = scf.IfOp(cond.result, [i32, i32], hasElse=True)
            with ir.InsertionPoint(if_op.then_block):
                x_true = arith.ConstantOp(i32, 0)
                y_true = arith.ConstantOp(i32, 1)
                scf.YieldOp([x_true, y_true])
            with ir.InsertionPoint(if_op.else_block):
                x_false = arith.ConstantOp(i32, 2)
                y_false = arith.ConstantOp(i32, 3)
                scf.YieldOp([x_false, y_false])
            as_idx=arith.IndexCastOp(ir.IndexType.get(),add)
            undef=util.Hash64(ir.IndexType.get(),as_idx)
            ret = func.ReturnOp([if_op.results_[0]])
    print(module)

context3=ir.Context()
mlir_init.init_context(context3)
with context3, ir.Location.unknown():
    module = ir.Module.create()
    i1type=ir.IntegerType.get_signless(1)
    reftype=util.RefType.get(i1type)
    with ir.InsertionPoint(module.body), ir.Location.unknown():
        f = func.FuncOp("main", ([], []))
        print(tuples.ColumnDefAttr.get(context3,"a","b",i1type))
        with ir.InsertionPoint(f.body.blocks.append()):
            r=util.AllocaOp(reftype)
            util.LoadOp(i1type,r)
            util.CreateConstVarLen(util.VarLen32Type.get(context3),ir.StringAttr.get("abc"))
            val = db.ConstantOp(db.StringType.get(context3),ir.StringAttr.get("somestring"))
            db.CmpOp(db.DBCmpPredicate.lt, val,val)
            curr_tuple=util.UndefOp(tuples.TupleType.get(context3))
            tuples.GetColumnOp(i1type, tuples.ColumnRefAttr.get(context3,"a","b"),curr_tuple)
            ret = func.ReturnOp([])
    print(module)


context4=ir.Context()
mlir_init.init_context(context4)
with context4, ir.Location.unknown():
    module = ir.Module.create()
    i1type=ir.IntegerType.get_signless(1)
    reftype=util.RefType.get(i1type)
    with ir.InsertionPoint(module.body), ir.Location.unknown():
        f = func.FuncOp("main", ([], []))
        with ir.InsertionPoint(f.body.blocks.append()):
            i32type=ir.IntegerType.get_signless(32)
            col1 = tuples.ColumnDefAttr.get(context4,"t1","a",i32type)
            col1_ref = tuples.ColumnRefAttr.get(context4,"t1","a")
            rel1 = relalg.BaseTableOp(ir.StringAttr.get('test_table'), relalg.TableMetaDataAttr.get_empty(context4),ir.DictAttr.get({'a': col1 }))
            rel2 = relalg.SortOp(rel1,ir.ArrayAttr.get([relalg.SortSpecificationAttr.get(col1_ref,relalg.SortSpec.desc)]))
            ret = func.ReturnOp([])
    print(module)
