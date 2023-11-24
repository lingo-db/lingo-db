import sys,os
import ctypes
import pyarrow as pa
import platform
from pathlib import Path
#pa.create_library_symlinks()
if platform.system().lower() == 'linux':
    import ctypes
    def _set_arrow_symbol_resolution(flag):
        for dir in map(Path, pa.get_library_dirs()):
            arrow_path = dir / 'libarrow.so'
            arrow_python_path = dir / 'libarrow_python.so'
            if arrow_path.exists() and arrow_python_path.exists():
                arrow_python = ctypes.CDLL(arrow_path, flag)
                libarrow_python = ctypes.CDLL(arrow_python_path, flag)
                break
    _set_arrow_symbol_resolution(ctypes.RTLD_GLOBAL)
dir_path = os.path.dirname(os.path.realpath(__file__))
ctypes.CDLL(dir_path+'/libs/libpybridge.so',os.RTLD_GLOBAL|os.RTLD_NOW|os.RTLD_DEEPBIND)

from . import ext
