#ifndef MLIR_CONVERSION_DBTOARROWSTD_REGISTEREDFUNCTIONS_H
#define MLIR_CONVERSION_DBTOARROWSTD_REGISTEREDFUNCTIONS_H

#define FUNC_LIST(F, OPERANDS, RETURNS)                                                                                                                    \
   F(SetExecutionContext, set_execution_context, OPERANDS(POINTER_TYPE), RETURNS())                                                                        \
   F(GetExecutionContext, get_execution_context, OPERANDS(), RETURNS(POINTER_TYPE))                                                                        \
   F(ExecutionContextGetTable, get_table, OPERANDS(POINTER_TYPE, STRING_TYPE), RETURNS(POINTER_TYPE))                                                      \
   F(DumpInt, dump_int, OPERANDS(BOOL_TYPE, INT_TYPE(64)), RETURNS())                                                                                      \
   F(DumpIndex, dump_index, OPERANDS(INDEX_TYPE), RETURNS())                                                                                               \
   F(DumpUInt, dump_uint, OPERANDS(BOOL_TYPE, INT_TYPE(64)), RETURNS())                                                                                    \
   F(DumpBool, dump_bool, OPERANDS(BOOL_TYPE, BOOL_TYPE), RETURNS())                                                                                       \
   F(DumpDecimal, dump_decimal, OPERANDS(BOOL_TYPE, INT_TYPE(64), INT_TYPE(64), INT_TYPE(32)), RETURNS())                                                  \
   F(DumpDate, dump_date, OPERANDS(BOOL_TYPE, INT_TYPE(64)), RETURNS())                                                                             \
   F(DumpTimestampSecond, dump_timestamp_second, OPERANDS(BOOL_TYPE, INT_TYPE(64)), RETURNS())                                                             \
   F(DumpTimestampMillisecond, dump_timestamp_millisecond, OPERANDS(BOOL_TYPE, INT_TYPE(64)), RETURNS())                                                   \
   F(DumpTimestampMicrosecond, dump_timestamp_microsecond, OPERANDS(BOOL_TYPE, INT_TYPE(64)), RETURNS())                                                   \
   F(DumpTimestampNanosecond, dump_timestamp_nanosecond, OPERANDS(BOOL_TYPE, INT_TYPE(64)), RETURNS())                                                     \
   F(DumpIntervalMonths, dump_interval_months, OPERANDS(BOOL_TYPE, INT_TYPE(32)), RETURNS())                                                               \
   F(DumpIntervalDayTime, dump_interval_daytime, OPERANDS(BOOL_TYPE, INT_TYPE(64)), RETURNS())                                                             \
   F(DumpFloat, dump_float, OPERANDS(BOOL_TYPE, DOUBLE_TYPE), RETURNS())                                                                                   \
   F(DumpString, dump_string, OPERANDS(BOOL_TYPE, STRING_TYPE), RETURNS())                                                                                 \
   F(DumpChar, dump_char, OPERANDS(BOOL_TYPE, INT_TYPE(64), INT_TYPE(64)), RETURNS())                                                                      \
   F(TableChunkIteratorInit, table_chunk_iterator_init, OPERANDS(POINTER_TYPE), RETURNS(POINTER_TYPE))                                                     \
   F(TableChunkIteratorNext, table_chunk_iterator_next, OPERANDS(POINTER_TYPE), RETURNS(POINTER_TYPE))                                                     \
   F(TableChunkIteratorCurr, table_chunk_iterator_curr, OPERANDS(POINTER_TYPE), RETURNS(POINTER_TYPE))                                                     \
   F(TableChunkIteratorValid, table_chunk_iterator_valid, OPERANDS(POINTER_TYPE), RETURNS(BOOL_TYPE))                                                      \
   F(TableChunkIteratorFree, table_chunk_iterator_free, OPERANDS(POINTER_TYPE), RETURNS())                                                                 \
   F(TableChunkNumRows, table_chunk_num_rows, OPERANDS(POINTER_TYPE), RETURNS(INDEX_TYPE))                                                                 \
   F(TableChunkGetColumnBuffer, table_chunk_get_column_buffer, OPERANDS(POINTER_TYPE, INDEX_TYPE, INDEX_TYPE), RETURNS(POINTER_TYPE))                      \
   F(TableChunkGetRawColumnBuffer, table_chunk_get_raw_column_buffer, OPERANDS(POINTER_TYPE, INDEX_TYPE, INDEX_TYPE), RETURNS(POINTER_TYPE,INDEX_TYPE))                 \
   F(TableChunkGetColumnOffset, table_chunk_get_column_offset, OPERANDS(POINTER_TYPE, INDEX_TYPE), RETURNS(INDEX_TYPE))                                    \
   F(ArrowGetType2Param, arrow_type2, OPERANDS(INT_TYPE(32), INT_TYPE(32), INT_TYPE(32)), RETURNS(POINTER_TYPE))                                           \
   F(ArrowGetType1Param, arrow_type1, OPERANDS(INT_TYPE(32), INT_TYPE(32)), RETURNS(POINTER_TYPE))                                                         \
   F(ArrowGetType, arrow_type, OPERANDS(INT_TYPE(32)), RETURNS(POINTER_TYPE))                                                                              \
   F(ArrowTableSchemaCreate, arrow_schema_create_builder, OPERANDS(), RETURNS(POINTER_TYPE))                                                               \
   F(ArrowTableSchemaAddField, arrow_schema_add_field, OPERANDS(POINTER_TYPE, POINTER_TYPE, BOOL_TYPE, STRING_TYPE), RETURNS())                            \
   F(ArrowTableSchemaBuild, arrow_schema_build, OPERANDS(POINTER_TYPE), RETURNS(POINTER_TYPE))                                                             \
   F(ArrowTableBuilderCreate, arrow_create_table_builder, OPERANDS(POINTER_TYPE), RETURNS(POINTER_TYPE))                                                   \
   F(ArrowTableBuilderAddInt8, table_builder_add_int_8, OPERANDS(POINTER_TYPE, INT_TYPE(32), BOOL_TYPE, INT_TYPE(8)), RETURNS())                           \
   F(ArrowTableBuilderAddInt16, table_builder_add_int_16, OPERANDS(POINTER_TYPE, INT_TYPE(32), BOOL_TYPE, INT_TYPE(16)), RETURNS())                        \
   F(ArrowTableBuilderAddInt32, table_builder_add_int_32, OPERANDS(POINTER_TYPE, INT_TYPE(32), BOOL_TYPE, INT_TYPE(32)), RETURNS())                        \
   F(ArrowTableBuilderAddInt64, table_builder_add_int_64, OPERANDS(POINTER_TYPE, INT_TYPE(32), BOOL_TYPE, INT_TYPE(64)), RETURNS())                        \
   F(ArrowTableBuilderAddDecimal, table_builder_add_decimal, OPERANDS(POINTER_TYPE, INT_TYPE(32), BOOL_TYPE, INT_TYPE(128)), RETURNS())                    \
   F(ArrowTableBuilderAddSmallDecimal, table_builder_add_small_decimal, OPERANDS(POINTER_TYPE, INT_TYPE(32), BOOL_TYPE, INT_TYPE(64)), RETURNS())          \
   F(ArrowTableBuilderAddDate32, table_builder_add_date_32, OPERANDS(POINTER_TYPE, INT_TYPE(32), BOOL_TYPE, INT_TYPE(64)), RETURNS())                      \
   F(ArrowTableBuilderAddDate64, table_builder_add_date_64, OPERANDS(POINTER_TYPE, INT_TYPE(32), BOOL_TYPE, INT_TYPE(64)), RETURNS())                      \
   F(ArrowTableBuilderAddFloat32, table_builder_add_float_32, OPERANDS(POINTER_TYPE, INT_TYPE(32), BOOL_TYPE, FLOAT_TYPE), RETURNS())                      \
   F(ArrowTableBuilderAddFloat64, table_builder_add_float_64, OPERANDS(POINTER_TYPE, INT_TYPE(32), BOOL_TYPE, DOUBLE_TYPE), RETURNS())                     \
   F(ArrowTableBuilderAddBool, table_builder_add_bool, OPERANDS(POINTER_TYPE, INT_TYPE(32), BOOL_TYPE, BOOL_TYPE), RETURNS())                              \
   F(ArrowTableBuilderAddBinary, table_builder_add_binary, OPERANDS(POINTER_TYPE, INT_TYPE(32), BOOL_TYPE, STRING_TYPE), RETURNS())                        \
   F(ArrowTableBuilderAddFixedBinary, table_builder_add_fixed_binary, OPERANDS(POINTER_TYPE, INT_TYPE(32), BOOL_TYPE, INT_TYPE(64)), RETURNS())            \
   F(ArrowTableBuilderFinishRow, table_builder_finish_row, OPERANDS(POINTER_TYPE), RETURNS())                                                              \
   F(ArrowTableBuilderBuild, table_builder_build, OPERANDS(POINTER_TYPE), RETURNS(POINTER_TYPE))                                                           \
   F(CmpStringEQ, cmp_string_eq, OPERANDS(STRING_TYPE, STRING_TYPE), RETURNS(BOOL_TYPE))                                                        \
   F(CmpStringNEQ, cmp_string_neq, OPERANDS(STRING_TYPE, STRING_TYPE), RETURNS(BOOL_TYPE))                                                      \
   F(CmpStringLT, cmp_string_lt, OPERANDS(STRING_TYPE, STRING_TYPE), RETURNS(BOOL_TYPE))                                                        \
   F(CmpStringLTE, cmp_string_lte, OPERANDS(STRING_TYPE, STRING_TYPE), RETURNS(BOOL_TYPE))                                                      \
   F(CmpStringGT, cmp_string_gt, OPERANDS(STRING_TYPE, STRING_TYPE), RETURNS(BOOL_TYPE))                                                        \
   F(CmpStringGTE, cmp_string_gte, OPERANDS(STRING_TYPE, STRING_TYPE), RETURNS(BOOL_TYPE))                                                      \
   F(CmpStringLike, cmp_string_like, OPERANDS(STRING_TYPE, STRING_TYPE), RETURNS(BOOL_TYPE))                                                    \
   F(Substring, substring, OPERANDS(STRING_TYPE, INDEX_TYPE, INDEX_TYPE), RETURNS(STRING_TYPE))                                                    \
   F(CmpStringStartsWith, cmp_string_starts_with, OPERANDS(STRING_TYPE, STRING_TYPE), RETURNS(BOOL_TYPE))                                       \
   F(CmpStringEndsWith, cmp_string_ends_with, OPERANDS(STRING_TYPE, STRING_TYPE), RETURNS(BOOL_TYPE))                                           \
   F(CastStringToInt64, cast_string_int, OPERANDS(STRING_TYPE), RETURNS(INT_TYPE(64)))                                                          \
   F(CastStringToFloat32, cast_string_float32, OPERANDS(STRING_TYPE), RETURNS(FLOAT_TYPE))                                                      \
   F(CastStringToFloat64, cast_string_float64, OPERANDS(STRING_TYPE), RETURNS(DOUBLE_TYPE))                                                     \
   F(CastStringToDecimal, cast_string_decimal, OPERANDS(STRING_TYPE, INT_TYPE(32)), RETURNS(INT_TYPE(128)))                                     \
   F(CastInt64ToString, cast_int_string, OPERANDS(INT_TYPE(64)), RETURNS(STRING_TYPE))                                                          \
   F(CastFloat32ToString, cast_float32_string, OPERANDS(FLOAT_TYPE), RETURNS(STRING_TYPE))                                                      \
   F(CastFloat64ToString, cast_float64_string, OPERANDS(DOUBLE_TYPE), RETURNS(STRING_TYPE))                                                     \
   F(CastDecimalToString, cast_decimal_string, OPERANDS(INT_TYPE(128), INT_TYPE(32)), RETURNS(STRING_TYPE))                                     \
   F(CastCharToString, cast_char_string, OPERANDS(INT_TYPE(64), INT_TYPE(64)), RETURNS(STRING_TYPE))                                            \
   F(SortVector, sort, OPERANDS(INDEX_TYPE, POINTER_TYPE, INDEX_TYPE, FUNCTION_TYPE(OPERANDS(POINTER_TYPE, POINTER_TYPE), RETURNS(BOOL_TYPE))), RETURNS()) \
   F(TimestampAddMonth, timestamp_add_months, OPERANDS(INT_TYPE(64), INT_TYPE(32)), RETURNS(INT_TYPE(64)))                                                 \
   F(TimestampSubtractMonth, timestamp_subtract_months, OPERANDS(INT_TYPE(64), INT_TYPE(32)), RETURNS(INT_TYPE(64)))                                                 \
   F(DateExtractYear, extract_year, OPERANDS(INT_TYPE(64)), RETURNS(INT_TYPE(64)))                                                                         \
   F(DateExtractDoy, extract_doy, OPERANDS(INT_TYPE(64)), RETURNS(INT_TYPE(64)))                                                                           \
   F(DateExtractMonth, extract_month, OPERANDS(INT_TYPE(64)), RETURNS(INT_TYPE(64)))                                                                       \
   F(DateExtractDay, extract_day, OPERANDS(INT_TYPE(64)), RETURNS(INT_TYPE(64)))                                                                           \
   F(DateExtractDow, extractdow, OPERANDS(INT_TYPE(64)), RETURNS(INT_TYPE(64)))                                                                            \
   F(DateExtractHour, extract_hour, OPERANDS(INT_TYPE(64)), RETURNS(INT_TYPE(64)))                                                                         \
   F(DateExtractMinute, extract_minute, OPERANDS(INT_TYPE(64)), RETURNS(INT_TYPE(64)))                                                                     \
   F(DateExtractSecond, extract_minute, OPERANDS(INT_TYPE(64)), RETURNS(INT_TYPE(64)))                                                                     \
   F(VecResize, resize_vec, OPERANDS(POINTER_TYPE), RETURNS())                                                                                             \
   F(VecCreate, create_vec, OPERANDS(INDEX_TYPE, INDEX_TYPE), RETURNS(POINTER_TYPE))                                                                       \
   F(JoinHtCreate, create_join_ht, OPERANDS(INDEX_TYPE), RETURNS(POINTER_TYPE))                                                                       \
   F(JoinHtResize, join_ht_resize, OPERANDS(POINTER_TYPE), RETURNS())                                                                       \
   F(AggrHtCreate, create_aggr_ht, OPERANDS(INDEX_TYPE, INDEX_TYPE), RETURNS(POINTER_TYPE))                                                                 \
   F(AggrHtResize, resize_aggr_ht, OPERANDS(POINTER_TYPE), RETURNS()) \
   F(JoinHtBuild, build_join_ht, OPERANDS(POINTER_TYPE), RETURNS(POINTER_TYPE)) \
   F(JoinHtFinalize, join_ht_finalize, OPERANDS(POINTER_TYPE), RETURNS()) \
   F(ScanSourceInit, scan_source_init, OPERANDS(POINTER_TYPE, STRING_TYPE), RETURNS(POINTER_TYPE))

#endif // MLIR_CONVERSION_DBTOARROWSTD_REGISTEREDFUNCTIONS_H