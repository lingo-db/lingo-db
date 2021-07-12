module  {
  func private @_mlir_ciface_dump_int(i1, i64)
  func private @_mlir_ciface_aggr_ht_iterator_free(!util.generic_memref<i8>)
  func private @_mlir_ciface_aggr_ht_iterator_next(!util.generic_memref<i8>) -> !util.generic_memref<i8>
  func private @_mlir_ciface_aggr_ht_iterator_curr(!util.generic_memref<i8>) -> !util.generic_memref<i8>
  func private @_mlir_ciface_aggr_ht_iterator_valid(!util.generic_memref<i8>) -> i1
  func private @_mlir_ciface_aggr_ht_iterator_init(!util.generic_memref<i8>) -> !util.generic_memref<i8>
  func private @_mlir_ciface_aggr_ht_builder_build(!util.generic_memref<i8>) -> !util.generic_memref<i8>
  func private @_mlir_ciface_dump_string(i1, !util.generic_memref<? x i8>)
  func private @_mlir_ciface_hash_int_32(index, i32) -> index
  func private @_mlir_ciface_hash_binary(index, !util.generic_memref<? x i8>) -> index
  func private @_mlir_ciface_aggr_ht_builder_merge(!util.generic_memref<i8>, index, !util.generic_memref<i8>, !util.generic_memref<i8>)
  func private @_mlir_ciface_aggr_ht_builder_add_nullable_var_len(!util.generic_memref<i8>, i1, !util.generic_memref<? x i8>) -> tuple<i1, !util.generic_memref<? x i8>>
  func private @_mlir_ciface_aggr_ht_builder_fast_lookup(!util.generic_memref<i8>, index) -> tuple<i1, !util.generic_memref<i8>>
  func private @_mlir_ciface_cmp_string_eq(i1, !util.generic_memref<? x i8>, !util.generic_memref<? x i8>) -> i1
  func private @_mlir_ciface_aggr_ht_builder_create(index, index, index, index, (!util.generic_memref<i8>, !util.generic_memref<i8>) -> i1, (!util.generic_memref<i8>, !util.generic_memref<i8>) -> (), !util.generic_memref<i8>) -> !util.generic_memref<i8>
  func @db_ht_aggr_builder_update3(%arg0: !util.generic_memref<i8>, %arg1: !util.generic_memref<i8>) {
    %0 = util.generic_memref_cast %arg0 : !util.generic_memref<i8> -> !util.generic_memref<tuple<i32, i32>>
    %1 = util.generic_memref_cast %arg1 : !util.generic_memref<i8> -> !util.generic_memref<tuple<i32, i32>>
    %2 = util.load %0[] : !util.generic_memref<tuple<i32, i32>> -> tuple<i32, i32>
    %3 = util.load %1[] : !util.generic_memref<tuple<i32, i32>> -> tuple<i32, i32>
    %4 = call @db_ht_aggr_builder_raw_update0(%2, %3) : (tuple<i32, i32>, tuple<i32, i32>) -> tuple<i32, i32>
    util.store %4 : tuple<i32, i32>, %0[] : !util.generic_memref<tuple<i32, i32>>
    return
  }
  func @db_ht_aggr_builder_compare2(%arg0: !util.generic_memref<i8>, %arg1: !util.generic_memref<i8>) -> i1 {
    %0 = util.generic_memref_cast %arg0 : !util.generic_memref<i8> -> !util.generic_memref<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>>
    %1 = util.generic_memref_cast %arg1 : !util.generic_memref<i8> -> !util.generic_memref<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>>
    %2 = util.load %0[] : !util.generic_memref<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>> -> tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>
    %3 = util.load %1[] : !util.generic_memref<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>> -> tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>
    %4 = call @db_ht_aggr_builder_raw_compare1(%2, %3) : (tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>) -> i1
    return %4 : i1
  }
  func @db_ht_aggr_builder_raw_compare1(%arg0: tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, %arg1: tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>) -> i1 {
    %true = constant true
    %0:2 = util.unpack %arg0 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32> -> tuple<i1, !util.generic_memref<? x i8>>, i32
    %1:2 = util.unpack %arg1 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32> -> tuple<i1, !util.generic_memref<? x i8>>, i32
    %false = constant false
    %2 = call @_mlir_ciface_cmp_string_eq(%false, %0#0, %1#0) : (i1, tuple<i1, !util.generic_memref<? x i8>>, tuple<i1, !util.generic_memref<? x i8>>) -> i1
    %3:2 = util.unpack %0#0 : tuple<i1, !util.generic_memref<? x i8>> -> i1, !util.generic_memref<? x i8>
    %4:2 = util.unpack %1#0 : tuple<i1, !util.generic_memref<? x i8>> -> i1, !util.generic_memref<? x i8>
    %false_0 = constant false
    %true_1 = constant true
    %5 = select %4#0, %3#0, %false_0 : i1
    %false_2 = constant false
    %6 = util.pack %false_2, %5 : i1, i1 -> tuple<i1, i1>
    %7 = select %5, %6, %2 : tuple<i1, i1>
    %false_3 = constant false
    %true_4 = constant true
    %8:2 = util.unpack %7 : tuple<i1, i1> -> i1, i1
    %9 = select %8#1, %true, %false_3 : i1
    %10 = select %8#0, %true, %9 : i1
    %11 = select %10, %8#0, %false_3 : i1
    %12 = util.pack %11, %10 : i1, i1 -> tuple<i1, i1>
    %13 = cmpi eq, %0#1, %1#1 : i32
    %false_5 = constant false
    %true_6 = constant true
    %14:2 = util.unpack %12 : tuple<i1, i1> -> i1, i1
    %15 = select %14#0, %true_6, %14#1 : i1
    %16 = select %13, %15, %false_5 : i1
    %17 = select %16, %14#0, %false_5 : i1
    %18 = util.pack %17, %16 : i1, i1 -> tuple<i1, i1>
    %19:2 = util.unpack %18 : tuple<i1, i1> -> i1, i1
    %true_7 = constant true
    %20 = xor %19#0, %true_7 : i1
    %21:2 = util.unpack %18 : tuple<i1, i1> -> i1, i1
    %false_8 = constant false
    %true_9 = constant true
    %22 = select %21#1, %20, %false_8 : i1
    return %22 : i1
  }
  func @db_ht_aggr_builder_raw_update0(%arg0: tuple<i32, i32>, %arg1: tuple<i32, i32>) -> tuple<i32, i32> {
    %0:2 = util.unpack %arg0 : tuple<i32, i32> -> i32, i32
    %1:2 = util.unpack %arg1 : tuple<i32, i32> -> i32, i32
    %2 = addi %0#0, %1#0 : i32
    %3 = muli %0#1, %1#1 : i32
    %4 = util.pack %2, %3 : i32, i32 -> tuple<i32, i32>
    return %4 : tuple<i32, i32>
  }
  memref.global "private" constant @db_constant_string4 : memref<4xi8> = dense<[115, 116, 114, 100]>
  memref.global "private" constant @db_constant_string3 : memref<4xi8> = dense<[115, 116, 114, 99]>
  memref.global "private" constant @db_constant_string2 : memref<4xi8> = dense<[115, 116, 114, 98]>
  memref.global "private" constant @db_constant_string1 : memref<4xi8> = dense<[115, 116, 114, 97]>
  memref.global "private" constant @db_constant_string0 : memref<15xi8> = dense<45>
  func @main(%arg0: !util.generic_memref<i8>) {
    %0 = util.alloca() : !util.generic_memref<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>>
    %1 = util.alloca() : !util.generic_memref<tuple<i32, i32>>
    %2 = util.alloca() : !util.generic_memref<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>>
    %3 = util.alloca() : !util.generic_memref<tuple<i32, i32>>
    %4 = util.alloca() : !util.generic_memref<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>>
    %5 = util.alloca() : !util.generic_memref<tuple<i32, i32>>
    %6 = util.alloca() : !util.generic_memref<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>>
    %7 = util.alloca() : !util.generic_memref<tuple<i32, i32>>
    %8 = util.alloca() : !util.generic_memref<tuple<i32, i32>>
    %9 = memref.get_global @db_constant_string0 : memref<15xi8>
    %10 = memref.cast %9 : memref<15xi8> to memref<?xi8>
    %11 = util.to_generic_memref %10 : memref<?xi8> -> !util.generic_memref<? x i8>
    %12 = memref.get_global @db_constant_string1 : memref<4xi8>
    %13 = memref.cast %12 : memref<4xi8> to memref<?xi8>
    %14 = util.to_generic_memref %13 : memref<?xi8> -> !util.generic_memref<? x i8>
    %15 = memref.get_global @db_constant_string2 : memref<4xi8>
    %16 = memref.cast %15 : memref<4xi8> to memref<?xi8>
    %17 = util.to_generic_memref %16 : memref<?xi8> -> !util.generic_memref<? x i8>
    %18 = memref.get_global @db_constant_string3 : memref<4xi8>
    %19 = memref.cast %18 : memref<4xi8> to memref<?xi8>
    %20 = util.to_generic_memref %19 : memref<?xi8> -> !util.generic_memref<? x i8>
    %21 = memref.get_global @db_constant_string4 : memref<4xi8>
    %22 = memref.cast %21 : memref<4xi8> to memref<?xi8>
    %23 = util.to_generic_memref %22 : memref<?xi8> -> !util.generic_memref<? x i8>
    %false = constant false
    %24 = util.pack %false, %14 : i1, !util.generic_memref<? x i8> -> tuple<i1, !util.generic_memref<? x i8>>
    %false_0 = constant false
    %25 = util.pack %false_0, %17 : i1, !util.generic_memref<? x i8> -> tuple<i1, !util.generic_memref<? x i8>>
    %false_1 = constant false
    %26 = util.pack %false_1, %20 : i1, !util.generic_memref<? x i8> -> tuple<i1, !util.generic_memref<? x i8>>
    %false_2 = constant false
    %27 = util.pack %false_2, %23 : i1, !util.generic_memref<? x i8> -> tuple<i1, !util.generic_memref<? x i8>>
    %c4_i32 = constant 4 : i32
    %c2_i32 = constant 2 : i32
    %c3_i32 = constant 3 : i32
    %c1_i32 = constant 1 : i32
    %c0_i32 = constant 0 : i32
    %c1_i32_3 = constant 1 : i32
    %28 = util.pack %24, %c4_i32 : tuple<i1, !util.generic_memref<? x i8>>, i32 -> tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>
    %29 = util.pack %24, %c4_i32 : tuple<i1, !util.generic_memref<? x i8>>, i32 -> tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>
    %30 = util.pack %26, %c3_i32 : tuple<i1, !util.generic_memref<? x i8>>, i32 -> tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>
    %31 = util.pack %27, %c1_i32 : tuple<i1, !util.generic_memref<? x i8>>, i32 -> tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>
    %32 = util.pack %c4_i32, %c4_i32 : i32, i32 -> tuple<i32, i32>
    %33 = util.pack %c2_i32, %c2_i32 : i32, i32 -> tuple<i32, i32>
    %34 = util.pack %c3_i32, %c3_i32 : i32, i32 -> tuple<i32, i32>
    %35 = util.pack %c1_i32, %c1_i32 : i32, i32 -> tuple<i32, i32>
    %36 = util.pack %28, %32 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32> -> tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>
    %37 = util.pack %29, %33 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32> -> tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>
    %38 = util.pack %30, %34 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32> -> tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>
    %39 = util.pack %31, %35 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32> -> tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>
    %40 = util.pack %c0_i32, %c1_i32_3 : i32, i32 -> tuple<i32, i32>
    %f = constant @db_ht_aggr_builder_update3 : (!util.generic_memref<i8>, !util.generic_memref<i8>) -> ()
    %f_4 = constant @db_ht_aggr_builder_compare2 : (!util.generic_memref<i8>, !util.generic_memref<i8>) -> i1
    %41 = util.sizeof tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>
    %42 = util.sizeof tuple<i32, i32>
    %43 = util.sizeof tuple<i32, i32>
    %44 = util.sizeof tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>
    util.store %40 : tuple<i32, i32>, %8[] : !util.generic_memref<tuple<i32, i32>>
    %45 = util.generic_memref_cast %8 : !util.generic_memref<tuple<i32, i32>> -> !util.generic_memref<i8>
    %46 = call @_mlir_ciface_aggr_ht_builder_create(%41, %42, %43, %44, %f_4, %f, %45) : (index, index, index, index, (!util.generic_memref<i8>, !util.generic_memref<i8>) -> i1, (!util.generic_memref<i8>, !util.generic_memref<i8>) -> (), !util.generic_memref<i8>) -> !util.generic_memref<i8>
    %f_5 = constant @db_ht_aggr_builder_raw_compare1 : (tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>) -> i1
    %f_6 = constant @db_ht_aggr_builder_raw_update0 : (tuple<i32, i32>, tuple<i32, i32>) -> tuple<i32, i32>
    %47 = util.pack %46, %f_5, %f_6 : !util.generic_memref<i8>, (tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>) -> i1, (tuple<i32, i32>, tuple<i32, i32>) -> tuple<i32, i32> -> tuple<!util.generic_memref<i8>, (tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>) -> i1, (tuple<i32, i32>, tuple<i32, i32>) -> tuple<i32, i32>>
    %48:3 = util.unpack %47 : tuple<!util.generic_memref<i8>, (tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>) -> i1, (tuple<i32, i32>, tuple<i32, i32>) -> tuple<i32, i32>> -> !util.generic_memref<i8>, (tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>) -> i1, (tuple<i32, i32>, tuple<i32, i32>) -> tuple<i32, i32>
    %49:2 = util.unpack %36 : tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>> -> tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>
    %c0 = constant 0 : index
    %50:2 = util.unpack %49#0 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32> -> tuple<i1, !util.generic_memref<? x i8>>, i32
    %51:2 = util.unpack %50#0 : tuple<i1, !util.generic_memref<? x i8>> -> i1, !util.generic_memref<? x i8>
    %52 = call @_mlir_ciface_hash_binary(%c0, %51#1) : (index, !util.generic_memref<? x i8>) -> index
    %53 = select %51#0, %c0, %52 : index
    %54 = call @_mlir_ciface_hash_int_32(%53, %50#1) : (index, i32) -> index
    %55 = util.generic_memref_cast %6 : !util.generic_memref<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>> -> !util.generic_memref<i8>
    %56 = util.generic_memref_cast %7 : !util.generic_memref<tuple<i32, i32>> -> !util.generic_memref<i8>
    %57 = call @_mlir_ciface_aggr_ht_builder_fast_lookup(%48#0, %54) : (!util.generic_memref<i8>, index) -> tuple<i1, !util.generic_memref<i8>>
    %58:2 = util.unpack %57 : tuple<i1, !util.generic_memref<i8>> -> i1, !util.generic_memref<i8>
    scf.if %58#0 {
      %96 = util.generic_memref_cast %58#1 : !util.generic_memref<i8> -> !util.generic_memref<tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>>
      %97 = util.load %96[] : !util.generic_memref<tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>> -> tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>
      %98:2 = util.unpack %97 : tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>> -> tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>
      %99 = call_indirect %48#1(%98#0, %49#0) : (tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>) -> i1
      scf.if %99 {
        %100 = call_indirect %48#2(%98#1, %49#1) : (tuple<i32, i32>, tuple<i32, i32>) -> tuple<i32, i32>
        %101 = util.pack %98#0, %100 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32> -> tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>
        util.store %101 : tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>, %96[] : !util.generic_memref<tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>>
      } else {
        %100:2 = util.unpack %49#0 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32> -> tuple<i1, !util.generic_memref<? x i8>>, i32
        %101:2 = util.unpack %100#0 : tuple<i1, !util.generic_memref<? x i8>> -> i1, !util.generic_memref<? x i8>
        %102 = call @_mlir_ciface_aggr_ht_builder_add_nullable_var_len(%48#0, %101#0, %101#1) : (!util.generic_memref<i8>, i1, !util.generic_memref<? x i8>) -> tuple<i1, !util.generic_memref<? x i8>>
        %103 = util.pack %102, %100#1 : tuple<i1, !util.generic_memref<? x i8>>, i32 -> tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>
        %104:2 = util.unpack %49#1 : tuple<i32, i32> -> i32, i32
        %105 = util.pack %104#0, %104#1 : i32, i32 -> tuple<i32, i32>
        util.store %103 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, %6[] : !util.generic_memref<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>>
        util.store %105 : tuple<i32, i32>, %7[] : !util.generic_memref<tuple<i32, i32>>
        call @_mlir_ciface_aggr_ht_builder_merge(%48#0, %54, %55, %56) : (!util.generic_memref<i8>, index, !util.generic_memref<i8>, !util.generic_memref<i8>) -> ()
      }
    } else {
      %96:2 = util.unpack %49#0 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32> -> tuple<i1, !util.generic_memref<? x i8>>, i32
      %97:2 = util.unpack %96#0 : tuple<i1, !util.generic_memref<? x i8>> -> i1, !util.generic_memref<? x i8>
      %98 = call @_mlir_ciface_aggr_ht_builder_add_nullable_var_len(%48#0, %97#0, %97#1) : (!util.generic_memref<i8>, i1, !util.generic_memref<? x i8>) -> tuple<i1, !util.generic_memref<? x i8>>
      %99 = util.pack %98, %96#1 : tuple<i1, !util.generic_memref<? x i8>>, i32 -> tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>
      %100:2 = util.unpack %49#1 : tuple<i32, i32> -> i32, i32
      %101 = util.pack %100#0, %100#1 : i32, i32 -> tuple<i32, i32>
      util.store %99 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, %6[] : !util.generic_memref<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>>
      util.store %101 : tuple<i32, i32>, %7[] : !util.generic_memref<tuple<i32, i32>>
      call @_mlir_ciface_aggr_ht_builder_merge(%48#0, %54, %55, %56) : (!util.generic_memref<i8>, index, !util.generic_memref<i8>, !util.generic_memref<i8>) -> ()
    }
    %false_7 = constant false
    call @_mlir_ciface_dump_string(%false_7, %11) : (i1, !util.generic_memref<? x i8>) -> ()
    %59:3 = util.unpack %47 : tuple<!util.generic_memref<i8>, (tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>) -> i1, (tuple<i32, i32>, tuple<i32, i32>) -> tuple<i32, i32>> -> !util.generic_memref<i8>, (tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>) -> i1, (tuple<i32, i32>, tuple<i32, i32>) -> tuple<i32, i32>
    %60:2 = util.unpack %37 : tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>> -> tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>
    %c0_8 = constant 0 : index
    %61:2 = util.unpack %60#0 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32> -> tuple<i1, !util.generic_memref<? x i8>>, i32
    %62:2 = util.unpack %61#0 : tuple<i1, !util.generic_memref<? x i8>> -> i1, !util.generic_memref<? x i8>
    %63 = call @_mlir_ciface_hash_binary(%c0_8, %62#1) : (index, !util.generic_memref<? x i8>) -> index
    %64 = select %62#0, %c0_8, %63 : index
    %65 = call @_mlir_ciface_hash_int_32(%64, %61#1) : (index, i32) -> index
    %66 = util.generic_memref_cast %4 : !util.generic_memref<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>> -> !util.generic_memref<i8>
    %67 = util.generic_memref_cast %5 : !util.generic_memref<tuple<i32, i32>> -> !util.generic_memref<i8>
    %68 = call @_mlir_ciface_aggr_ht_builder_fast_lookup(%59#0, %65) : (!util.generic_memref<i8>, index) -> tuple<i1, !util.generic_memref<i8>>
    %69:2 = util.unpack %68 : tuple<i1, !util.generic_memref<i8>> -> i1, !util.generic_memref<i8>
    scf.if %69#0 {
      %96 = util.generic_memref_cast %69#1 : !util.generic_memref<i8> -> !util.generic_memref<tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>>
      %97 = util.load %96[] : !util.generic_memref<tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>> -> tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>
      %98:2 = util.unpack %97 : tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>> -> tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>
      %99 = call_indirect %59#1(%98#0, %60#0) : (tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>) -> i1
      scf.if %99 {
        %100 = call_indirect %59#2(%98#1, %60#1) : (tuple<i32, i32>, tuple<i32, i32>) -> tuple<i32, i32>
        %101 = util.pack %98#0, %100 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32> -> tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>
        util.store %101 : tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>, %96[] : !util.generic_memref<tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>>
      } else {
        %100:2 = util.unpack %60#0 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32> -> tuple<i1, !util.generic_memref<? x i8>>, i32
        %101:2 = util.unpack %100#0 : tuple<i1, !util.generic_memref<? x i8>> -> i1, !util.generic_memref<? x i8>
        %102 = call @_mlir_ciface_aggr_ht_builder_add_nullable_var_len(%59#0, %101#0, %101#1) : (!util.generic_memref<i8>, i1, !util.generic_memref<? x i8>) -> tuple<i1, !util.generic_memref<? x i8>>
        %103 = util.pack %102, %100#1 : tuple<i1, !util.generic_memref<? x i8>>, i32 -> tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>
        %104:2 = util.unpack %60#1 : tuple<i32, i32> -> i32, i32
        %105 = util.pack %104#0, %104#1 : i32, i32 -> tuple<i32, i32>
        util.store %103 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, %4[] : !util.generic_memref<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>>
        util.store %105 : tuple<i32, i32>, %5[] : !util.generic_memref<tuple<i32, i32>>
        call @_mlir_ciface_aggr_ht_builder_merge(%59#0, %65, %66, %67) : (!util.generic_memref<i8>, index, !util.generic_memref<i8>, !util.generic_memref<i8>) -> ()
      }
    } else {
      %96:2 = util.unpack %60#0 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32> -> tuple<i1, !util.generic_memref<? x i8>>, i32
      %97:2 = util.unpack %96#0 : tuple<i1, !util.generic_memref<? x i8>> -> i1, !util.generic_memref<? x i8>
      %98 = call @_mlir_ciface_aggr_ht_builder_add_nullable_var_len(%59#0, %97#0, %97#1) : (!util.generic_memref<i8>, i1, !util.generic_memref<? x i8>) -> tuple<i1, !util.generic_memref<? x i8>>
      %99 = util.pack %98, %96#1 : tuple<i1, !util.generic_memref<? x i8>>, i32 -> tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>
      %100:2 = util.unpack %60#1 : tuple<i32, i32> -> i32, i32
      %101 = util.pack %100#0, %100#1 : i32, i32 -> tuple<i32, i32>
      util.store %99 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, %4[] : !util.generic_memref<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>>
      util.store %101 : tuple<i32, i32>, %5[] : !util.generic_memref<tuple<i32, i32>>
      call @_mlir_ciface_aggr_ht_builder_merge(%59#0, %65, %66, %67) : (!util.generic_memref<i8>, index, !util.generic_memref<i8>, !util.generic_memref<i8>) -> ()
    }
    %false_9 = constant false
    call @_mlir_ciface_dump_string(%false_9, %11) : (i1, !util.generic_memref<? x i8>) -> ()
    %70:3 = util.unpack %47 : tuple<!util.generic_memref<i8>, (tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>) -> i1, (tuple<i32, i32>, tuple<i32, i32>) -> tuple<i32, i32>> -> !util.generic_memref<i8>, (tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>) -> i1, (tuple<i32, i32>, tuple<i32, i32>) -> tuple<i32, i32>
    %71:2 = util.unpack %38 : tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>> -> tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>
    %c0_10 = constant 0 : index
    %72:2 = util.unpack %71#0 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32> -> tuple<i1, !util.generic_memref<? x i8>>, i32
    %73:2 = util.unpack %72#0 : tuple<i1, !util.generic_memref<? x i8>> -> i1, !util.generic_memref<? x i8>
    %74 = call @_mlir_ciface_hash_binary(%c0_10, %73#1) : (index, !util.generic_memref<? x i8>) -> index
    %75 = select %73#0, %c0_10, %74 : index
    %76 = call @_mlir_ciface_hash_int_32(%75, %72#1) : (index, i32) -> index
    %77 = util.generic_memref_cast %2 : !util.generic_memref<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>> -> !util.generic_memref<i8>
    %78 = util.generic_memref_cast %3 : !util.generic_memref<tuple<i32, i32>> -> !util.generic_memref<i8>
    %79 = call @_mlir_ciface_aggr_ht_builder_fast_lookup(%70#0, %76) : (!util.generic_memref<i8>, index) -> tuple<i1, !util.generic_memref<i8>>
    %80:2 = util.unpack %79 : tuple<i1, !util.generic_memref<i8>> -> i1, !util.generic_memref<i8>
    scf.if %80#0 {
      %96 = util.generic_memref_cast %80#1 : !util.generic_memref<i8> -> !util.generic_memref<tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>>
      %97 = util.load %96[] : !util.generic_memref<tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>> -> tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>
      %98:2 = util.unpack %97 : tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>> -> tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>
      %99 = call_indirect %70#1(%98#0, %71#0) : (tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>) -> i1
      scf.if %99 {
        %100 = call_indirect %70#2(%98#1, %71#1) : (tuple<i32, i32>, tuple<i32, i32>) -> tuple<i32, i32>
        %101 = util.pack %98#0, %100 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32> -> tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>
        util.store %101 : tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>, %96[] : !util.generic_memref<tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>>
      } else {
        %100:2 = util.unpack %71#0 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32> -> tuple<i1, !util.generic_memref<? x i8>>, i32
        %101:2 = util.unpack %100#0 : tuple<i1, !util.generic_memref<? x i8>> -> i1, !util.generic_memref<? x i8>
        %102 = call @_mlir_ciface_aggr_ht_builder_add_nullable_var_len(%70#0, %101#0, %101#1) : (!util.generic_memref<i8>, i1, !util.generic_memref<? x i8>) -> tuple<i1, !util.generic_memref<? x i8>>
        %103 = util.pack %102, %100#1 : tuple<i1, !util.generic_memref<? x i8>>, i32 -> tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>
        %104:2 = util.unpack %71#1 : tuple<i32, i32> -> i32, i32
        %105 = util.pack %104#0, %104#1 : i32, i32 -> tuple<i32, i32>
        util.store %103 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, %2[] : !util.generic_memref<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>>
        util.store %105 : tuple<i32, i32>, %3[] : !util.generic_memref<tuple<i32, i32>>
        call @_mlir_ciface_aggr_ht_builder_merge(%70#0, %76, %77, %78) : (!util.generic_memref<i8>, index, !util.generic_memref<i8>, !util.generic_memref<i8>) -> ()
      }
    } else {
      %96:2 = util.unpack %71#0 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32> -> tuple<i1, !util.generic_memref<? x i8>>, i32
      %97:2 = util.unpack %96#0 : tuple<i1, !util.generic_memref<? x i8>> -> i1, !util.generic_memref<? x i8>
      %98 = call @_mlir_ciface_aggr_ht_builder_add_nullable_var_len(%70#0, %97#0, %97#1) : (!util.generic_memref<i8>, i1, !util.generic_memref<? x i8>) -> tuple<i1, !util.generic_memref<? x i8>>
      %99 = util.pack %98, %96#1 : tuple<i1, !util.generic_memref<? x i8>>, i32 -> tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>
      %100:2 = util.unpack %71#1 : tuple<i32, i32> -> i32, i32
      %101 = util.pack %100#0, %100#1 : i32, i32 -> tuple<i32, i32>
      util.store %99 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, %2[] : !util.generic_memref<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>>
      util.store %101 : tuple<i32, i32>, %3[] : !util.generic_memref<tuple<i32, i32>>
      call @_mlir_ciface_aggr_ht_builder_merge(%70#0, %76, %77, %78) : (!util.generic_memref<i8>, index, !util.generic_memref<i8>, !util.generic_memref<i8>) -> ()
    }
    %false_11 = constant false
    call @_mlir_ciface_dump_string(%false_11, %11) : (i1, !util.generic_memref<? x i8>) -> ()
    %81:3 = util.unpack %47 : tuple<!util.generic_memref<i8>, (tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>) -> i1, (tuple<i32, i32>, tuple<i32, i32>) -> tuple<i32, i32>> -> !util.generic_memref<i8>, (tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>) -> i1, (tuple<i32, i32>, tuple<i32, i32>) -> tuple<i32, i32>
    %82:2 = util.unpack %39 : tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>> -> tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>
    %c0_12 = constant 0 : index
    %83:2 = util.unpack %82#0 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32> -> tuple<i1, !util.generic_memref<? x i8>>, i32
    %84:2 = util.unpack %83#0 : tuple<i1, !util.generic_memref<? x i8>> -> i1, !util.generic_memref<? x i8>
    %85 = call @_mlir_ciface_hash_binary(%c0_12, %84#1) : (index, !util.generic_memref<? x i8>) -> index
    %86 = select %84#0, %c0_12, %85 : index
    %87 = call @_mlir_ciface_hash_int_32(%86, %83#1) : (index, i32) -> index
    %88 = util.generic_memref_cast %0 : !util.generic_memref<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>> -> !util.generic_memref<i8>
    %89 = util.generic_memref_cast %1 : !util.generic_memref<tuple<i32, i32>> -> !util.generic_memref<i8>
    %90 = call @_mlir_ciface_aggr_ht_builder_fast_lookup(%81#0, %87) : (!util.generic_memref<i8>, index) -> tuple<i1, !util.generic_memref<i8>>
    %91:2 = util.unpack %90 : tuple<i1, !util.generic_memref<i8>> -> i1, !util.generic_memref<i8>
    scf.if %91#0 {
      %96 = util.generic_memref_cast %91#1 : !util.generic_memref<i8> -> !util.generic_memref<tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>>
      %97 = util.load %96[] : !util.generic_memref<tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>> -> tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>
      %98:2 = util.unpack %97 : tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>> -> tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>
      %99 = call_indirect %81#1(%98#0, %82#0) : (tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>) -> i1
      scf.if %99 {
        %100 = call_indirect %81#2(%98#1, %82#1) : (tuple<i32, i32>, tuple<i32, i32>) -> tuple<i32, i32>
        %101 = util.pack %98#0, %100 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32> -> tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>
        util.store %101 : tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>, %96[] : !util.generic_memref<tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>>
      } else {
        %100:2 = util.unpack %82#0 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32> -> tuple<i1, !util.generic_memref<? x i8>>, i32
        %101:2 = util.unpack %100#0 : tuple<i1, !util.generic_memref<? x i8>> -> i1, !util.generic_memref<? x i8>
        %102 = call @_mlir_ciface_aggr_ht_builder_add_nullable_var_len(%81#0, %101#0, %101#1) : (!util.generic_memref<i8>, i1, !util.generic_memref<? x i8>) -> tuple<i1, !util.generic_memref<? x i8>>
        %103 = util.pack %102, %100#1 : tuple<i1, !util.generic_memref<? x i8>>, i32 -> tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>
        %104:2 = util.unpack %82#1 : tuple<i32, i32> -> i32, i32
        %105 = util.pack %104#0, %104#1 : i32, i32 -> tuple<i32, i32>
        util.store %103 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, %0[] : !util.generic_memref<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>>
        util.store %105 : tuple<i32, i32>, %1[] : !util.generic_memref<tuple<i32, i32>>
        call @_mlir_ciface_aggr_ht_builder_merge(%81#0, %87, %88, %89) : (!util.generic_memref<i8>, index, !util.generic_memref<i8>, !util.generic_memref<i8>) -> ()
      }
    } else {
      %96:2 = util.unpack %82#0 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32> -> tuple<i1, !util.generic_memref<? x i8>>, i32
      %97:2 = util.unpack %96#0 : tuple<i1, !util.generic_memref<? x i8>> -> i1, !util.generic_memref<? x i8>
      %98 = call @_mlir_ciface_aggr_ht_builder_add_nullable_var_len(%81#0, %97#0, %97#1) : (!util.generic_memref<i8>, i1, !util.generic_memref<? x i8>) -> tuple<i1, !util.generic_memref<? x i8>>
      %99 = util.pack %98, %96#1 : tuple<i1, !util.generic_memref<? x i8>>, i32 -> tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>
      %100:2 = util.unpack %82#1 : tuple<i32, i32> -> i32, i32
      %101 = util.pack %100#0, %100#1 : i32, i32 -> tuple<i32, i32>
      util.store %99 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, %0[] : !util.generic_memref<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>>
      util.store %101 : tuple<i32, i32>, %1[] : !util.generic_memref<tuple<i32, i32>>
      call @_mlir_ciface_aggr_ht_builder_merge(%81#0, %87, %88, %89) : (!util.generic_memref<i8>, index, !util.generic_memref<i8>, !util.generic_memref<i8>) -> ()
    }
    %false_13 = constant false
    call @_mlir_ciface_dump_string(%false_13, %11) : (i1, !util.generic_memref<? x i8>) -> ()
    %92:3 = util.unpack %47 : tuple<!util.generic_memref<i8>, (tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>) -> i1, (tuple<i32, i32>, tuple<i32, i32>) -> tuple<i32, i32>> -> !util.generic_memref<i8>, (tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>) -> i1, (tuple<i32, i32>, tuple<i32, i32>) -> tuple<i32, i32>
    %93 = call @_mlir_ciface_aggr_ht_builder_build(%92#0) : (!util.generic_memref<i8>) -> !util.generic_memref<i8>
    %94 = call @_mlir_ciface_aggr_ht_iterator_init(%93) : (!util.generic_memref<i8>) -> !util.generic_memref<i8>
    %95 = scf.while (%arg1 = %94) : (!util.generic_memref<i8>) -> !util.generic_memref<i8> {
      %96 = call @_mlir_ciface_aggr_ht_iterator_valid(%arg1) : (!util.generic_memref<i8>) -> i1
      scf.condition(%96) %arg1 : !util.generic_memref<i8>
    } do {
    ^bb0(%arg1: !util.generic_memref<i8>):  // no predecessors
      %96 = call @_mlir_ciface_aggr_ht_iterator_curr(%arg1) : (!util.generic_memref<i8>) -> !util.generic_memref<i8>
      %97 = util.generic_memref_cast %96 : !util.generic_memref<i8> -> !util.generic_memref<tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>>
      %98 = util.load %97[] : !util.generic_memref<tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>> -> tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>>
      %99:2 = util.unpack %98 : tuple<tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>> -> tuple<tuple<i1, !util.generic_memref<? x i8>>, i32>, tuple<i32, i32>
      %100:2 = util.unpack %99#0 : tuple<tuple<i1, !util.generic_memref<? x i8>>, i32> -> tuple<i1, !util.generic_memref<? x i8>>, i32
      %101:2 = util.unpack %99#1 : tuple<i32, i32> -> i32, i32
      %102:2 = util.unpack %100#0 : tuple<i1, !util.generic_memref<? x i8>> -> i1, !util.generic_memref<? x i8>
      call @_mlir_ciface_dump_string(%102#0, %102#1) : (i1, !util.generic_memref<? x i8>) -> ()
      %false_14 = constant false
      %103 = sexti %100#1 : i32 to i64
      call @_mlir_ciface_dump_int(%false_14, %103) : (i1, i64) -> ()
      %false_15 = constant false
      %104 = sexti %101#0 : i32 to i64
      call @_mlir_ciface_dump_int(%false_15, %104) : (i1, i64) -> ()
      %false_16 = constant false
      %105 = sexti %101#1 : i32 to i64
      call @_mlir_ciface_dump_int(%false_16, %105) : (i1, i64) -> ()
      %false_17 = constant false
      call @_mlir_ciface_dump_string(%false_17, %11) : (i1, !util.generic_memref<? x i8>) -> ()
      %106 = call @_mlir_ciface_aggr_ht_iterator_next(%arg1) : (!util.generic_memref<i8>) -> !util.generic_memref<i8>
      scf.yield %106 : !util.generic_memref<i8>
    }
    call @_mlir_ciface_aggr_ht_iterator_free(%95) : (!util.generic_memref<i8>) -> ()
    return
  }
}