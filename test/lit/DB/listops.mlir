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


        return
    }
}