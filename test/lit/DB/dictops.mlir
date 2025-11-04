// RUN: run-mlir %s | FileCheck %s

module {
    func.func @cmpKeys(%a: i32, %b: i32) -> i1 {
        %cmp = arith.cmpi eq, %a, %b : i32
        return %cmp : i1
    }
    func.func @main ()  {
        %dict = db.create_dict @cmpKeys -> !db.dict<i32, i32>
        %len_empty = db.dict_length %dict : !db.dict<i32, i32>
        // CHECK: index(0)
        db.runtime_call "DumpValue" (%len_empty) : (index) -> ()

        %c1 = arith.constant 1 : i32
        %c2 = arith.constant 2 : i32
        %hash1 = db.hash %c1 : i32
        db.dict_set %dict: !db.dict<i32, i32> [%c1 : i32 -> %hash1] = %c2 : i32
        // CHECK: bool(true)
        %contains_one = db.dict_contains %dict: !db.dict<i32, i32> [%c1 : i32 -> %hash1]
        db.runtime_call "DumpValue" (%contains_one) : (i1) -> ()
        // CHECK: int(2)
        %get_one = db.dict_get %dict: !db.dict<i32, i32> [%c1 : i32 -> %hash1] : i32
        db.runtime_call "DumpValue" (%get_one) : (i32) -> ()
         // CHECK: index(1)
        %len_one = db.dict_length %dict : !db.dict<i32, i32>
        db.runtime_call "DumpValue" (%len_one) : (index) -> ()

        %c3 = arith.constant 3 : i32
        %c4 = arith.constant 4 : i32
        %hash3 = db.hash %c3 : i32
        db.dict_set %dict: !db.dict<i32, i32> [%c3 : i32 -> %hash3] = %c4 : i32
        // CHECK: bool(true)
        %contains_three = db.dict_contains %dict: !db.dict<i32, i32> [%c3 : i32 -> %hash3]
        db.runtime_call "DumpValue" (%contains_three) : (i1) -> ()
        // CHECK: int(4)
        %get_three = db.dict_get %dict: !db.dict<i32, i32> [%c3 : i32 -> %hash3] : i32
        db.runtime_call "DumpValue" (%get_three) : (i32) -> ()
        // CHECK: index(2)
        %len_two = db.dict_length %dict : !db.dict<i32, i32>
        db.runtime_call "DumpValue" (%len_two) : (index) -> ()

        // CHECK: bool(false)
        %c42 = arith.constant 42 : i32
        %hash42 = db.hash %c42 : i32
        %contains_24 = db.dict_contains %dict: !db.dict<i32, i32> [%c42 : i32 -> %hash42]
        db.runtime_call "DumpValue" (%contains_24) : (i1) -> ()


        %it = db.dict_get_iter %dict : !db.dict<i32, i32> -> !db.dict_iter<i32, i32>
        scf.while () : () -> () {
            %has_next = db.dict_iter_valid %it : !db.dict_iter<i32, i32>
            scf.condition(%has_next)
        } do {
            // CHECK: int(1)
            // CHECK: int(2)
            // CHECK: int(3)
            // CHECK: int(4)
            %key = db.dict_iter_get_key %it : !db.dict_iter<i32, i32> -> i32
            db.runtime_call "DumpValue" (%key) : (i32) -> ()
            %value = db.dict_iter_get_value %it : !db.dict_iter<i32, i32> -> i32
            db.runtime_call "DumpValue" (%value) : (i32) -> ()
            db.dict_iter_next %it : !db.dict_iter<i32, i32>
            scf.yield
        }
        return
    }
}