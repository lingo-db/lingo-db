// RUN: mlir-db-opt %s -subop-memory-mgmt -mlir-print-local-scope | FileCheck %s
//
// Exercises the `ManagedTypeInterface` dispatch in -subop-memory-mgmt:
// each `func.func` is processed as a non-main block, so any value whose
// type returns true from `ManagedType::needsManagement()` should get a
// `db.memory.cleanup_use` before the return, and unmanaged values should not.

// db.string implements ManagedType (returns true).
// CHECK-LABEL: func.func @managed_string
// CHECK: %[[V:.+]] = db.runtime_call "ToLower"
// CHECK: db.memory.cleanup_use %[[V]] : !db.string
func.func @managed_string(%a: !db.string) {
  %0 = db.runtime_call "ToLower"(%a) : (!db.string) -> !db.string
  return
}

// db.list implements ManagedType (returns true).
// CHECK-LABEL: func.func @managed_list
// CHECK: %[[L:.+]] = db.create_list
// CHECK: db.memory.cleanup_use %[[L]] : !db.list<i32>
func.func @managed_list() {
  %0 = db.create_list !db.list<i32>
  return
}

// db.char<N> with N > 1 lowers to a refcounted VarLen32, so it is managed.
// CHECK-LABEL: func.func @managed_char_large
// CHECK: %[[C:.+]] = db.cast
// CHECK: db.memory.cleanup_use %[[C]] : !db.char<5>
func.func @managed_char_large(%a: !db.string) {
  %0 = db.cast %a : !db.string -> !db.char<5>
  return
}

// db.char<1> is inline (single i32) and must NOT be managed.
// CHECK-LABEL: func.func @unmanaged_char_inline
// CHECK-NOT: db.memory.cleanup_use {{.+}} : !db.char<1>
// CHECK: return
func.func @unmanaged_char_inline(%a: !db.string) {
  %0 = db.cast %a : !db.string -> !db.char<1>
  return
}

// db.nullable<db.string> delegates to inner type → managed.
// CHECK-LABEL: func.func @managed_nullable_string
// CHECK: %[[N:.+]] = db.as_nullable
// CHECK: db.memory.cleanup_use %[[N]] : !db.nullable<!db.string>
func.func @managed_nullable_string(%a: !db.string) {
  %0 = db.as_nullable %a : !db.string -> !db.nullable<!db.string>
  return
}

// db.nullable<i32> delegates to inner type → unmanaged.
// CHECK-LABEL: func.func @unmanaged_nullable_i32
// CHECK-NOT: db.memory
// CHECK: return
func.func @unmanaged_nullable_i32(%a: i32) {
  %0 = db.as_nullable %a : i32 -> !db.nullable<i32>
  return
}

// Tuple containing a managed element: the external model on mlir::TupleType
// reports managed; the pass unpacks and emits cleanup_use on each managed leaf.
// CHECK-LABEL: func.func @managed_tuple_with_string
// CHECK: util.pack
// CHECK: db.memory.cleanup_use {{.+}} : !db.string
func.func @managed_tuple_with_string(%s: !db.string, %i: i32) {
  %0 = util.pack %s, %i : !db.string, i32 -> tuple<!db.string, i32>
  return
}

// Tuple of only unmanaged types → external model returns false.
// CHECK-LABEL: func.func @unmanaged_tuple_plain
// CHECK-NOT: db.memory
// CHECK: return
func.func @unmanaged_tuple_plain(%a: i32, %b: i64) {
  %0 = util.pack %a, %b : i32, i64 -> tuple<i32, i64>
  return
}

// ListAppendOp implements RefCountedOp: element is an owned operand
// → add_use is emitted *before* the op.
// CHECK-LABEL: func.func @list_append_owns_element
// CHECK: db.memory.add_use %{{.*}} : !db.string
// CHECK-NEXT: db.list_append
func.func @list_append_owns_element(%l: !db.list<!db.string>, %s: !db.string) {
  db.list_append %l : !db.list<!db.string>, %s : !db.string
  return
}

// ListSetOp implements RefCountedOp: element is an owned operand.
// CHECK-LABEL: func.func @list_set_owns_element
// CHECK: db.memory.add_use %{{.*}} : !db.string
// CHECK-NEXT: db.list_set
func.func @list_set_owns_element(%l: !db.list<!db.string>, %i: index, %s: !db.string) {
  db.list_set %l : !db.list<!db.string>[%i] = %s : !db.string
  return
}

// ListGetOp implements RefCountedOp: the element result is borrowed
// → add_use is emitted *after* the op.
// CHECK-LABEL: func.func @list_get_borrows_element
// CHECK: %[[E:.+]] = db.list_get
// CHECK-NEXT: db.memory.add_use %[[E]] : !db.string
func.func @list_get_borrows_element(%l: !db.list<!db.string>, %i: index) {
  %0 = db.list_get %l : !db.list<!db.string>[%i] : !db.string
  return
}

// NullableGetVal implements RefCountedOp: the unwrapped result is borrowed
// (the underlying buffer is shared with the nullable). Pass emits an
// add_use after so the unwrapped value owns its own reference and the
// matching cleanup_use at the end of the block stays balanced.
// CHECK-LABEL: func.func @nullable_get_val_borrows
// CHECK: %[[V:.+]] = db.nullable_get_val
// CHECK-NEXT: db.memory.add_use %[[V]] : !db.string
func.func @nullable_get_val_borrows(%n: !db.nullable<!db.string>) {
  %0 = db.nullable_get_val %n : !db.nullable<!db.string>
  return
}

// util.unpack implements RefCountedOp via an external model: each
// unpacked value borrows from the tuple.
// CHECK-LABEL: func.func @util_unpack_borrows
// CHECK: util.unpack
// CHECK: db.memory.add_use {{.*}} : !db.string
func.func @util_unpack_borrows(%t: tuple<!db.string, i32>) {
  %s, %i = util.unpack %t : tuple<!db.string, i32> -> !db.string, i32
  return
}

// util.get_tuple implements RefCountedOp via an external model: the
// extracted value borrows from the tuple.
// CHECK-LABEL: func.func @util_get_tuple_borrows
// CHECK: %[[V:.+]] = util.get_tuple
// CHECK-NEXT: db.memory.add_use %[[V]] : !db.string
func.func @util_get_tuple_borrows(%t: tuple<!db.string, i32>) {
  %0 = util.get_tuple %t[0] : (tuple<!db.string, i32>) -> !db.string
  return
}

// scf.for implements RefCountedOp via an external model: each initArg of
// managed type is an owned operand → add_use before the loop.
// CHECK-LABEL: func.func @for_owns_initargs
// CHECK: db.memory.add_use %arg3 : !db.string
// CHECK-NEXT: scf.for
func.func @for_owns_initargs(%lo: index, %hi: index, %step: index, %s: !db.string) -> !db.string {
  %r = scf.for %i = %lo to %hi step %step iter_args(%a = %s) -> (!db.string) {
    scf.yield %a : !db.string
  }
  return %r : !db.string
}

// scf.while implements RefCountedOp via an external model: each init of
// managed type is an owned operand → add_use before the loop.
// CHECK-LABEL: func.func @while_owns_inits
// CHECK: db.memory.add_use %arg0 : !db.string
// CHECK-NEXT: scf.while
func.func @while_owns_inits(%s: !db.string) -> !db.string {
  %r = scf.while (%a = %s) : (!db.string) -> !db.string {
    %c = arith.constant 1 : i1
    scf.condition(%c) %a : !db.string
  } do {
  ^bb0(%a: !db.string):
    scf.yield %a : !db.string
  }
  return %r : !db.string
}

// arith.select on a managed type can't grow refcounts in place — the
// rewriteForRefCount hook replaces it with scf.if and emits add_use in
// each arm.
// CHECK-LABEL: func.func @select_rewrites_managed
// CHECK-NOT: arith.select
// CHECK: scf.if
// CHECK: db.memory.add_use
// CHECK: scf.yield
// CHECK: else
// CHECK: db.memory.add_use
// CHECK: scf.yield
func.func @select_rewrites_managed(%c: i1, %a: !db.string, %b: !db.string) -> !db.string {
  %0 = arith.select %c, %a, %b : !db.string
  return %0 : !db.string
}

// arith.select on an unmanaged type passes through untouched.
// CHECK-LABEL: func.func @select_unmanaged_passes_through
// CHECK: arith.select
// CHECK-NOT: scf.if
// CHECK-NOT: db.memory
// CHECK: return
func.func @select_unmanaged_passes_through(%c: i1, %a: i32, %b: i32) -> i32 {
  %0 = arith.select %c, %a, %b : i32
  return %0 : i32
}

// !py_interp.py_object implements ManagedType via an external model: emitters
// produce `py_interp.{inc_ref, dec_ref}` instead of `db.memory.*`.
//
// In this function: %0 (get_attr) and %arg0 (block arg) are managed locals,
// %1 (py_call result) is returned and bypasses cleanup. Both managed locals
// get an auto-emitted dec_ref before the return.
//
// CHECK-LABEL: func.func @pyobject_auto_decref
// CHECK: py_interp.get_attr
// CHECK: py_interp.py_call
// CHECK-DAG: py_interp.dec_ref %0
// CHECK-DAG: py_interp.dec_ref %arg0
// CHECK: return
func.func @pyobject_auto_decref(%mod: !py_interp.py_object, %arg: !py_interp.py_object) -> !py_interp.py_object {
  %fn = py_interp.get_attr %mod -> "foo"
  %r = py_interp.py_call %fn(%arg) []
  return %r : !py_interp.py_object
}

// py_interp.create_module is excluded from refcount tracking (cached, owned
// by the interpreter). Its result must NOT be dec_ref'd, even though its
// type is managed.
// CHECK-LABEL: func.func @pyobject_create_module_not_counted
// CHECK: %[[M:.+]] = py_interp.create_module
// CHECK: py_interp.get_attr %[[M]]
// CHECK-NOT: py_interp.dec_ref %[[M]]
// CHECK: return
func.func @pyobject_create_module_not_counted(%arg: !py_interp.py_object) -> !py_interp.py_object {
  %m = py_interp.create_module "udf_foo", "def foo(x): return x"
  %fn = py_interp.get_attr %m -> "foo"
  %r = py_interp.py_call %fn(%arg) []
  return %r : !py_interp.py_object
}
