module {
 func.func @foo(%arg0 : !py_interp.py_object) -> !py_interp.py_object {
    return %arg0: !py_interp.py_object
 }
 func.func @main() {
    %0 = py_interp.import "math"
    %1 = func.call @foo(%0) : (!py_interp.py_object) -> !py_interp.py_object
    return
  }
}