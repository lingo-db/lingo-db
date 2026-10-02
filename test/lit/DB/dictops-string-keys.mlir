// RUN: LINGODB_EXECUTION_MODE=DEFAULT run-mlir %s | FileCheck %s
// RUN: %if baseline-backend %{LINGODB_EXECUTION_MODE=BASELINE run-mlir %s | FileCheck %s %}
// A dict whose key type is changed by the lowering (!db.string ->
// !util.varlen32): the ops cloned from the key comparison function must still
// see the unlowered key type (before: "failed to legalize db.create_dict").

module {
    func.func @cmpKeys(%a: !db.string, %b: !db.string) -> i1 {
        %cmp = db.compare eq %a : !db.string, %b : !db.string
        return %cmp : i1
    }
    func.func @main ()  {
        %dict = db.create_dict @cmpKeys -> !db.dict<!db.string, i32>
        %a = db.constant("a long key that is not inlined") : !db.string
        %b = db.constant("b") : !db.string
        %a2 = db.constant("a long key that is not inlined") : !db.string
        %c1 = arith.constant 1 : i32
        %c2 = arith.constant 2 : i32
        %c3 = arith.constant 3 : i32
        %hash_a = db.hash %a : !db.string
        %hash_b = db.hash %b : !db.string
        %hash_a2 = db.hash %a2 : !db.string
        db.dict_set %dict: !db.dict<!db.string, i32> [%a : !db.string -> %hash_a] = %c1 : i32
        db.dict_set %dict: !db.dict<!db.string, i32> [%b : !db.string -> %hash_b] = %c2 : i32
        db.dict_set %dict: !db.dict<!db.string, i32> [%a2 : !db.string -> %hash_a2] = %c3 : i32
        // CHECK: index(2)
        %len = db.dict_length %dict : !db.dict<!db.string, i32>
        db.runtime_call "DumpValue" (%len) : (index) -> ()
        // CHECK: int(3)
        %get_a = db.dict_get %dict: !db.dict<!db.string, i32> [%a : !db.string -> %hash_a] : i32
        db.runtime_call "DumpValue" (%get_a) : (i32) -> ()
        // CHECK: int(2)
        %get_b = db.dict_get %dict: !db.dict<!db.string, i32> [%b : !db.string -> %hash_b] : i32
        db.runtime_call "DumpValue" (%get_b) : (i32) -> ()
        // CHECK: bool(false)
        %c = db.constant("c") : !db.string
        %hash_c = db.hash %c : !db.string
        %contains_c = db.dict_contains %dict: !db.dict<!db.string, i32> [%c : !db.string -> %hash_c]
        db.runtime_call "DumpValue" (%contains_c) : (i1) -> ()
        return
    }
}
