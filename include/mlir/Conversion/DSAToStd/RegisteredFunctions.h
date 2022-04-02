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