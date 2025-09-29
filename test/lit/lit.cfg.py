# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'mlirdb'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir','.sql','.py']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.mlirdb_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

if platform.system() == "Darwin":  # Check if the system is macOS
    config.environment['LINGODB_COMPILATION_STATIC_LINKER'] = "/opt/homebrew/bin/clang++"  # Use the CMake CXX compiler path
    config.environment['LINGODB_COMPILATION_C_BACKEND_COMPILER_DRIVER'] = "/opt/homebrew/bin/clang++"  # Use the CMake CXX compiler path

llvm_config.with_system_environment(
    ['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP', "LINGODB_COMPILATION_STATICLINKER", "LINGODB_COMPILATION_C_BACKEND_COMPILER_DRIVER", "ENABLE_BASELINE_BACKEND"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'Examples', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt','lit.cfg.py']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.mlirdb_obj_root, 'test')
print(config.test_exec_root)

config.mlirdb_tools_dir = config.mlirdb_obj_root

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)
llvm_config.with_environment('PYTHONPATH', config.mlirdb_obj_root, append_path=True)
llvm_config.with_environment('PYTHONPATH', os.path.join(config.mlirdb_src_root, "arrow/python"), append_path=True)
#llvm_config.with_environment('LINGODB_EXECUTION_MODE', 'C')
#llvm_config.with_environment('DATABASE_DIR', os.path.join(config.mlirdb_src_root,'resources/data/uni'))
tool_dirs = [config.mlirdb_tools_dir, config.llvm_tools_dir]
tools = [
    'mlir-db-opt',
    'run-mlir',
    'sql-to-mlir',
    'run-sql'
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# Query tool features for conditional testing.
def get_features(tool):
    tool_path = f"{config.mlirdb_obj_root}/{tool}"
    try:
        result = subprocess.run([tool_path, '--features'], stdin=subprocess.DEVNULL , stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        features = set(result.stderr.strip().splitlines())
        return features
    except Exception as e:
        print(f"Warning: Could not get features from {tool_path}: {e}")
        return set()

tool_features = set(frozenset(get_features(t)) for t in tools)
if len(tool_features) > 1:
    print("Warning: Tools have differing features, which may lead to inconsistent test behavior.")
else:
    print("Running tests with tool features:\n", "\n".join(next(iter(tool_features))))

if all("baseline-backend" in features for features in tool_features):
    config.available_features.add('baseline-backend')
