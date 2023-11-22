import os
def get_mlir_dir():
    return os.path.dirname(os.path.abspath(__file__))+"/llvm/lib/cmake/mlir"

def get_bin_dir():
    return os.path.dirname(os.path.abspath(__file__))+"/llvm/bin"