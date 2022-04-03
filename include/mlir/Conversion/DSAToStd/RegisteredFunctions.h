#ifndef MLIR_CONVERSION_DSATOSTD_REGISTEREDFUNCTIONS_H
#define MLIR_CONVERSION_DSATOSTD_REGISTEREDFUNCTIONS_H

#define FUNC_LIST(F, OPERANDS, RETURNS)                                                                                                                    \
   F(SetExecutionContext, set_execution_context, OPERANDS(POINTER_TYPE), RETURNS())                                                                        \
   F(GetExecutionContext, get_execution_context, OPERANDS(), RETURNS(POINTER_TYPE))                                                                        \
   F(SortVector, sort, OPERANDS(INDEX_TYPE, POINTER_TYPE, INDEX_TYPE, FUNCTION_TYPE(OPERANDS(POINTER_TYPE, POINTER_TYPE), RETURNS(BOOL_TYPE))), RETURNS()) \
   F(VecResize, resize_vec, OPERANDS(POINTER_TYPE), RETURNS())                                                                                             \
   F(VecCreate, create_vec, OPERANDS(INDEX_TYPE, INDEX_TYPE), RETURNS(POINTER_TYPE))                                                                       \
   F(JoinHtCreate, create_join_ht, OPERANDS(INDEX_TYPE), RETURNS(POINTER_TYPE))                                                                            \
   F(JoinHtResize, join_ht_resize, OPERANDS(POINTER_TYPE), RETURNS())                                                                                      \
   F(AggrHtCreate, create_aggr_ht, OPERANDS(INDEX_TYPE, INDEX_TYPE), RETURNS(POINTER_TYPE))                                                                \
   F(AggrHtResize, resize_aggr_ht, OPERANDS(POINTER_TYPE), RETURNS())                                                                                      \
   F(JoinHtBuild, build_join_ht, OPERANDS(POINTER_TYPE), RETURNS(POINTER_TYPE))                                                                            \
   F(JoinHtFinalize, join_ht_finalize, OPERANDS(POINTER_TYPE), RETURNS())
#endif // MLIR_CONVERSION_DSATOSTD_REGISTEREDFUNCTIONS_H