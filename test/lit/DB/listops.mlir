// RUN: run-mlir %s | FileCheck %s
module {
    func.func @main ()  {
        %list = db.create_list !db.list<i32>
        %len_empty = db.list_length %list : !db.list<i32>
        //CHECK: index(0)
        db.runtime_call "DumpValue" (%len_empty) : (index) -> ()
        %c1 = arith.constant 1 : i32
        db.list_append %list : !db.list<i32>, %c1 : i32
        %len_one = db.list_length %list : !db.list<i32>
        //CHECK: index(1)
        db.runtime_call "DumpValue" (%len_one) : (index) -> ()
        %c2 = arith.constant 2 : i32
        db.list_append %list : !db.list<i32>, %c2 : i32
        %len_two = db.list_length %list : !db.list<i32>
        //CHECK: index(2)
        db.runtime_call "DumpValue" (%len_two) : (index) -> ()
        //CHECK: int(1)
        %c0i = arith.constant 0 : index
        %elem1 = db.list_get %list: !db.list<i32> [%c0i]  : i32
        db.runtime_call "DumpValue" (%elem1) : (i32) -> ()
        //CHECK: int(2)
        %c1i = arith.constant 1 : index
        %elem2 = db.list_get %list: !db.list<i32> [%c1i]  : i32
        db.runtime_call "DumpValue" (%elem2) : (i32) -> ()
        //CHECK: int(42)
        %c42 = arith.constant 42 : i32
        db.list_set %list: !db.list<i32> [%c1i] = %c42 : i32
        %elem2_updated = db.list_get %list: !db.list<i32> [%c1i]  : i32
        db.runtime_call "DumpValue" (%elem2_updated) : (i32) -> ()
        // list_pop removes the element and moves the tail: [1, 42, 3] -> pop(0) -> [42, 3]
        %c3 = arith.constant 3 : i32
        db.list_append %list : !db.list<i32>, %c3 : i32
        //CHECK: int(1)
        %popped0 = db.list_pop %list : !db.list<i32> [%c0i] : i32
        db.runtime_call "DumpValue" (%popped0) : (i32) -> ()
        //CHECK: index(2)
        %len_popped = db.list_length %list : !db.list<i32>
        db.runtime_call "DumpValue" (%len_popped) : (index) -> ()
        //CHECK: int(42)
        %first = db.list_get %list: !db.list<i32> [%c0i]  : i32
        db.runtime_call "DumpValue" (%first) : (i32) -> ()
        //CHECK: int(3)
        %popped1 = db.list_pop %list : !db.list<i32> [%c1i] : i32
        db.runtime_call "DumpValue" (%popped1) : (i32) -> ()
        //CHECK: index(1)
        %len_popped2 = db.list_length %list : !db.list<i32>
        db.runtime_call "DumpValue" (%len_popped2) : (index) -> ()
        db.memory.cleanup_use %list : !db.list<i32>


        return
    }
}