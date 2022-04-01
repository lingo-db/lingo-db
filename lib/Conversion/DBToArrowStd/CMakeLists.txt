gen_rt_def(dump-rt-defs "DumpRuntime.h")
gen_rt_def(string-rt-defs "StringRuntime.h")
gen_rt_def(date-rt-defs "DateRuntime.h")
add_mlir_conversion_library(MLIRDBToArrowStd
        LowerToStd.cpp
        ScalarDBToStdPatterns.cpp
        RuntimeSpecificScalarToStdPatterns.cpp
        ArrowTypes.cpp

        ADDITIONAL_HEADER_DIRS
        ../../../mlir/Conversion/DBToArrowStd

        DEPENDS
        dump-rt-defs
        string-rt-defs
        date-rt-defs
        LINK_COMPONENTS
        Core

        LINK_LIBS mlir-support PUBLIC
        MLIRIR
        MLIRDB
        MLIRSCFTransforms
        )
