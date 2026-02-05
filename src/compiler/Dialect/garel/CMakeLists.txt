# --- GARel Dialect Library ---
add_mlir_dialect_library(MLIRGARel
        PARTIAL_SOURCES_INTENDED   # <--- Add this line
        GARelAttr.cpp
        GARelDialect.cpp
        GARelOps.cpp
        GARelTupleOps.cpp
        GARelTypes.cpp

        ADDITIONAL_HEADER_DIRS
        ../../../../include/lingodb/compiler/Dialect/GARel

        DEPENDS
        MLIRGARelOpsIncGen
        MLIRGARelPassesIncGen

        LINK_LIBS PUBLIC
        MLIRInferTypeOpInterface
        MLIRIR
        MLIRSupport
)

# Apply flags
target_compile_options(MLIRGARel PRIVATE -Wno-dangling-assignment-gsl)
if(NOT GRAPHALG_ENABLE_RTTI)
  target_compile_options(MLIRGARel PUBLIC -fno-rtti)
endif()

# --- Conversion Library ---
add_mlir_dialect_library(MLIRGraphAlgToRel
        PARTIAL_SOURCES_INTENDED   # <--- Add this line
        GraphAlgToRel.cpp

        DEPENDS
        MLIRGraphAlgOpsIncGen
        MLIRGARelOpsIncGen

        LINK_LIBS PUBLIC
        MLIRGraphAlg
        MLIRGARel
        MLIRPass
)