// RUN: env LINGODB_EXECUTION_MODE=DEFAULT run-mlir %s | FileCheck %s
// RUN: %if baseline-backend %{LINGODB_EXECUTION_MODE=BASELINE run-mlir %s | FileCheck %s %}

 module {
    func.func @list_str_cleanup(%arg0: !db.list<!db.string>) {
        %start = arith.constant 0 : index
        %len = db.list_length %arg0 : !db.list<!db.string>
        %step = arith.constant 1 : index
        scf.for %i = %start to %len step %step {
            %elem = db.list_get %arg0 : !db.list<!db.string>[%i] : !db.string
            db.memory.cleanup_use %elem : !db.string
        }
        return
    }
	func.func @main () {
		%0 = db.constant ("This is quite a long string!") : !db.string
	    %lower = db.runtime_call "ToLower"(%0) : (!db.string) -> !db.string
	    db.runtime_call "DumpValue" (%lower) : (!db.string) -> ()
	    db.memory.add_use %lower : !db.string
	    db.memory.cleanup_use %lower : !db.string
	    db.runtime_call "DumpValue" (%lower) : (!db.string) -> ()
	    db.memory.cleanup_use %lower : !db.string

	    %list = db.create_list !db.list<i32>
        %len_empty = db.list_length %list : !db.list<i32>
        //CHECK: index(0)
        db.runtime_call "DumpValue" (%len_empty) : (index) -> ()
        db.memory.add_use %list : !db.list<i32>
        db.memory.cleanup_use %list : !db.list<i32>
        %len_empty2 = db.list_length %list : !db.list<i32>
        //CHECK: index(0)
        db.runtime_call "DumpValue" (%len_empty2) : (index) -> ()
	    db.memory.cleanup_use %list : !db.list<i32>

	    %tstr = db.runtime_call "ToLower"(%0) : (!db.string) -> !db.string

	    %list_str = db.create_list !db.list<!db.string>
	    db.memory.add_use %tstr : !db.string
	    db.list_append %list_str : !db.list<!db.string>, %tstr : !db.string
	    db.memory.add_use %tstr : !db.string
	    db.list_append %list_str : !db.list<!db.string>, %tstr : !db.string
       	db.memory.cleanup_use %list_str : !db.list<!db.string> @list_str_cleanup
	    db.memory.cleanup_use %tstr : !db.string


		return
	}

 }
