configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(MLIRDB_TEST_DEPENDS
        FileCheck count not
        mlir-db-opt
        )

add_lit_testsuite(check-mlirdb "Running the mlirdb regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${MLIRDB_TEST_DEPENDS}
        )
set_target_properties(check-mlirdb PROPERTIES FOLDER "Tests")

add_lit_testsuites(MLIRDB ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${MLIRDB_TEST_DEPENDS})
