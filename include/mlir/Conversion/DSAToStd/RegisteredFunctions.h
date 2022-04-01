#ifndef MLIR_CONVERSION_DSATOSTD_REGISTEREDFUNCTIONS_H
#define MLIR_CONVERSION_DSATOSTD_REGISTEREDFUNCTIONS_H

#define FUNC_LIST(F, OPERANDS, RETURNS)                                                                                                                    \
   F(SetExecutionContext, set_execution_context, OPERANDS(POINTER_TYPE), RETURNS())                                                                        \
   F(GetExecutionContext, get_execution_context, OPERANDS(), RETURNS(POINTER_TYPE))                                                                        \
   F(ExecutionContextGetTable, get_table, OPERANDS(POINTER_TYPE, STRING_TYPE), RETURNS(POINTER_TYPE))                                                      \
   F(TableChunkIteratorInit, table_chunk_iterator_init, OPERANDS(POINTER_TYPE), RETURNS(POINTER_TYPE))                                                     \
   F(TableChunkIteratorNext, table_chunk_iterator_next, OPERANDS(POINTER_TYPE), RETURNS(POINTER_TYPE))                                                     \
   F(TableChunkIteratorCurr, table_chunk_iterator_curr, OPERANDS(POINTER_TYPE), RETURNS(POINTER_TYPE))                                                     \
   F(TableChunkIteratorValid, table_chunk_iterator_valid, OPERANDS(POINTER_TYPE), RETURNS(BOOL_TYPE))                                                      \
   F(TableChunkIteratorFree, table_chunk_iterator_free, OPERANDS(POINTER_TYPE), RETURNS())                                                                 \
   F(TableChunkNumRows, table_chunk_num_rows, OPERANDS(POINTER_TYPE), RETURNS(INDEX_TYPE))                                                                 \
   F(AccessRecordBatch, access_record_batch, OPERANDS(INDEX_TYPE, POINTER_TYPE, POINTER_TYPE,POINTER_TYPE), RETURNS())                                                  \
   F(TableChunkGetColumnBuffer, table_chunk_get_column_buffer, OPERANDS(POINTER_TYPE, INDEX_TYPE, INDEX_TYPE), RETURNS(POINTER_TYPE))                      \
   F(TableChunkGetRawColumnBuffer, table_chunk_get_raw_column_buffer, OPERANDS(POINTER_TYPE, INDEX_TYPE, INDEX_TYPE), RETURNS(POINTER_TYPE, INDEX_TYPE))   \
   F(TableChunkGetColumnOffset, table_chunk_get_column_offset, OPERANDS(POINTER_TYPE, INDEX_TYPE), RETURNS(INDEX_TYPE))                                    \
   F(ArrowGetType2Param, arrow_type2, OPERANDS(INT_TYPE(32), INT_TYPE(32), INT_TYPE(32)), RETURNS(POINTER_TYPE))                                           \
   F(ArrowGetType1Param, arrow_type1, OPERANDS(INT_TYPE(32), INT_TYPE(32)), RETURNS(POINTER_TYPE))                                                         \
   F(ArrowGetType, arrow_type, OPERANDS(INT_TYPE(32)), RETURNS(POINTER_TYPE))                                                                              \
   F(ArrowTableBuilderCreate, arrow_create_table_builder, OPERANDS(STRING_TYPE), RETURNS(POINTER_TYPE))                                                   \
   F(ArrowTableBuilderAddInt8, table_builder_add_int_8, OPERANDS(POINTER_TYPE,  BOOL_TYPE, INT_TYPE(8)), RETURNS())                           \
   F(ArrowTableBuilderAddInt16, table_builder_add_int_16, OPERANDS(POINTER_TYPE,  BOOL_TYPE, INT_TYPE(16)), RETURNS())                        \
   F(ArrowTableBuilderAddInt32, table_builder_add_int_32, OPERANDS(POINTER_TYPE,  BOOL_TYPE, INT_TYPE(32)), RETURNS())                        \
   F(ArrowTableBuilderAddInt64, table_builder_add_int_64, OPERANDS(POINTER_TYPE,  BOOL_TYPE, INT_TYPE(64)), RETURNS())                        \
   F(ArrowTableBuilderAddDecimal, table_builder_add_decimal, OPERANDS(POINTER_TYPE,  BOOL_TYPE, INT_TYPE(128)), RETURNS())                    \
   F(ArrowTableBuilderAddDate32, table_builder_add_date_32, OPERANDS(POINTER_TYPE,  BOOL_TYPE, INT_TYPE(64)), RETURNS())                      \
   F(ArrowTableBuilderAddDate64, table_builder_add_date_64, OPERANDS(POINTER_TYPE,  BOOL_TYPE, INT_TYPE(64)), RETURNS())                      \
   F(ArrowTableBuilderAddFloat32, table_builder_add_float_32, OPERANDS(POINTER_TYPE,  BOOL_TYPE, FLOAT_TYPE), RETURNS())                      \
   F(ArrowTableBuilderAddFloat64, table_builder_add_float_64, OPERANDS(POINTER_TYPE,  BOOL_TYPE, DOUBLE_TYPE), RETURNS())                     \
   F(ArrowTableBuilderAddBool, table_builder_add_bool, OPERANDS(POINTER_TYPE,  BOOL_TYPE, BOOL_TYPE), RETURNS())                              \
   F(ArrowTableBuilderAddBinary, table_builder_add_binary, OPERANDS(POINTER_TYPE,  BOOL_TYPE, STRING_TYPE), RETURNS())                        \
   F(ArrowTableBuilderAddFixedBinary, table_builder_add_fixed_binary, OPERANDS(POINTER_TYPE,  BOOL_TYPE, INT_TYPE(64)), RETURNS())            \
   F(ArrowTableBuilderFinishRow, table_builder_finish_row, OPERANDS(POINTER_TYPE), RETURNS())                                                              \
   F(ArrowTableBuilderBuild, table_builder_build, OPERANDS(POINTER_TYPE), RETURNS(POINTER_TYPE))                                                           \
   F(SortVector, sort, OPERANDS(INDEX_TYPE, POINTER_TYPE, INDEX_TYPE, FUNCTION_TYPE(OPERANDS(POINTER_TYPE, POINTER_TYPE), RETURNS(BOOL_TYPE))), RETURNS()) \
   F(VecResize, resize_vec, OPERANDS(POINTER_TYPE), RETURNS())                                                                                             \
   F(VecCreate, create_vec, OPERANDS(INDEX_TYPE, INDEX_TYPE), RETURNS(POINTER_TYPE))                                                                       \
   F(JoinHtCreate, create_join_ht, OPERANDS(INDEX_TYPE), RETURNS(POINTER_TYPE))                                                                            \
   F(JoinHtResize, join_ht_resize, OPERANDS(POINTER_TYPE), RETURNS())                                                                                      \
   F(AggrHtCreate, create_aggr_ht, OPERANDS(INDEX_TYPE, INDEX_TYPE), RETURNS(POINTER_TYPE))                                                                \
   F(AggrHtResize, resize_aggr_ht, OPERANDS(POINTER_TYPE), RETURNS())                                                                                      \
   F(JoinHtBuild, build_join_ht, OPERANDS(POINTER_TYPE), RETURNS(POINTER_TYPE))                                                                            \
   F(JoinHtFinalize, join_ht_finalize, OPERANDS(POINTER_TYPE), RETURNS())                                                                                  \
   F(ScanSourceInit, scan_source_init, OPERANDS(POINTER_TYPE, STRING_TYPE), RETURNS(POINTER_TYPE))

#endif // MLIR_CONVERSION_DSATOSTD_REGISTEREDFUNCTIONS_H