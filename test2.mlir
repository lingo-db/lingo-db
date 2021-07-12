module @querymodule  {
  func @main(%arg0: !util.generic_memref<i8>) -> !db.table {
    %5 = db.create_table_builder ["s.name", "v.titel"] : !db.table_builder<tuple<!db.string, !db.string>>
    %6 = db.create_join_ht_builder : !db.join_ht_builder<tuple<!db.int<64>>,tuple<!db.string>>
    %7 = db.create_join_ht_builder : !db.join_ht_builder<tuple<!db.int<64>>,tuple<!db.int<64>>>
    %8 = db.get_table "hoeren" %arg0 : !util.generic_memref<i8>
    %9 = db.tablescan %8 ["matrnr", "vorlnr"] : !db.iterable<!db.iterable<tuple<!db.int<64>, !db.int<64>>,table_row_iterator>,table_chunk_iterator>
    %10 = db.for %arg1 in %9 : !db.iterable<!db.iterable<tuple<!db.int<64>, !db.int<64>>,table_row_iterator>,table_chunk_iterator>  iter_args(%arg2 = %7) -> (!db.join_ht_builder<tuple<!db.int<64>>,tuple<!db.int<64>>>) {
      %21 = db.for %arg3 in %arg1 : !db.iterable<tuple<!db.int<64>, !db.int<64>>,table_row_iterator>  iter_args(%arg4 = %arg2) -> (!db.join_ht_builder<tuple<!db.int<64>>,tuple<!db.int<64>>>) {
        %22:2 = util.unpack %arg3 : tuple<!db.int<64>, !db.int<64>> -> !db.int<64>, !db.int<64>
        %23 = util.pack %22#0 : !db.int<64> -> tuple<!db.int<64>>
        %24 = util.pack %22#1 : !db.int<64> -> tuple<!db.int<64>>
        %25 = util.pack %23, %24 : tuple<!db.int<64>>, tuple<!db.int<64>> -> tuple<tuple<!db.int<64>>, tuple<!db.int<64>>>
        %26 = db.builder_merge %arg4 : !db.join_ht_builder<tuple<!db.int<64>>,tuple<!db.int<64>>>, %25 : tuple<tuple<!db.int<64>>, tuple<!db.int<64>>> -> !db.join_ht_builder<tuple<!db.int<64>>,tuple<!db.int<64>>>
        db.yield %26 : !db.join_ht_builder<tuple<!db.int<64>>,tuple<!db.int<64>>>
      }
      db.yield %21 : !db.join_ht_builder<tuple<!db.int<64>>,tuple<!db.int<64>>>
    }
    %11 = db.builder_build %7 : !db.join_ht_builder<tuple<!db.int<64>>,tuple<!db.int<64>>> -> !db.join_ht<tuple<!db.int<64>>,tuple<!db.int<64>>>
    %12 = db.get_table "studenten" %arg0 : !util.generic_memref<i8>
    %13 = db.tablescan %12 ["matrnr", "name"] : !db.iterable<!db.iterable<tuple<!db.int<64>, !db.string>,table_row_iterator>,table_chunk_iterator>
    %14 = db.for %arg1 in %13 : !db.iterable<!db.iterable<tuple<!db.int<64>, !db.string>,table_row_iterator>,table_chunk_iterator>  iter_args(%arg2 = %6) -> (!db.join_ht_builder<tuple<!db.int<64>>,tuple<!db.string>>) {
      %21 = db.for %arg3 in %arg1 : !db.iterable<tuple<!db.int<64>, !db.string>,table_row_iterator>  iter_args(%arg4 = %arg2) -> (!db.join_ht_builder<tuple<!db.int<64>>,tuple<!db.string>>) {
        %22:2 = util.unpack %arg3 : tuple<!db.int<64>, !db.string> -> !db.int<64>, !db.string
        %23 = util.pack %22#0 : !db.int<64> -> tuple<!db.int<64>>
        %24 = db.lookup %11 : !db.join_ht<tuple<!db.int<64>>,tuple<!db.int<64>>>, %23 : tuple<!db.int<64>> -> !db.iterable<tuple<tuple<!db.int<64>>, tuple<!db.int<64>>>,join_ht_iterator>
        %25 = db.for %arg5 in %24 : !db.iterable<tuple<tuple<!db.int<64>>, tuple<!db.int<64>>>,join_ht_iterator>  iter_args(%arg6 = %arg4) -> (!db.join_ht_builder<tuple<!db.int<64>>,tuple<!db.string>>) {
          %26:2 = util.unpack %arg5 : tuple<tuple<!db.int<64>>, tuple<!db.int<64>>> -> tuple<!db.int<64>>, tuple<!db.int<64>>
          %27 = util.unpack %26#0 : tuple<!db.int<64>> -> !db.int<64>
          %28 = util.unpack %26#1 : tuple<!db.int<64>> -> !db.int<64>
          %29 = db.compare eq %27 : !db.int<64>, %22#0 : !db.int<64>
          %30 = db.if %29 : !db.bool -> (!db.join_ht_builder<tuple<!db.int<64>>,tuple<!db.string>>) {
            %31 = util.pack %28 : !db.int<64> -> tuple<!db.int<64>>
            %32 = util.pack %22#1 : !db.string -> tuple<!db.string>
            %33 = util.pack %31, %32 : tuple<!db.int<64>>, tuple<!db.string> -> tuple<tuple<!db.int<64>>, tuple<!db.string>>
            %34 = db.builder_merge %arg6 : !db.join_ht_builder<tuple<!db.int<64>>,tuple<!db.string>>, %33 : tuple<tuple<!db.int<64>>, tuple<!db.string>> -> !db.join_ht_builder<tuple<!db.int<64>>,tuple<!db.string>>
            db.yield %34 : !db.join_ht_builder<tuple<!db.int<64>>,tuple<!db.string>>
          } else {
            db.yield %arg6 : !db.join_ht_builder<tuple<!db.int<64>>,tuple<!db.string>>
          }
          db.yield %30 : !db.join_ht_builder<tuple<!db.int<64>>,tuple<!db.string>>
        }
        db.yield %25 : !db.join_ht_builder<tuple<!db.int<64>>,tuple<!db.string>>
      }
      db.yield %21 : !db.join_ht_builder<tuple<!db.int<64>>,tuple<!db.string>>
    }
    %15 = db.builder_build %6 : !db.join_ht_builder<tuple<!db.int<64>>,tuple<!db.string>> -> !db.join_ht<tuple<!db.int<64>>,tuple<!db.string>>
    %16 = db.get_table "vorlesungen" %arg0 : !util.generic_memref<i8>
    %17 = db.tablescan %16 ["titel", "vorlnr"] : !db.iterable<!db.iterable<tuple<!db.string, !db.int<64>>,table_row_iterator>,table_chunk_iterator>
    %18 = db.for %arg1 in %17 : !db.iterable<!db.iterable<tuple<!db.string, !db.int<64>>,table_row_iterator>,table_chunk_iterator>  iter_args(%arg2 = %5) -> (!db.table_builder<tuple<!db.string, !db.string>>) {
      %21 = db.for %arg3 in %arg1 : !db.iterable<tuple<!db.string, !db.int<64>>,table_row_iterator>  iter_args(%arg4 = %arg2) -> (!db.table_builder<tuple<!db.string, !db.string>>) {
        %22:2 = util.unpack %arg3 : tuple<!db.string, !db.int<64>> -> !db.string, !db.int<64>
        %23 = util.pack %22#1 : !db.int<64> -> tuple<!db.int<64>>
        %24 = db.lookup %15 : !db.join_ht<tuple<!db.int<64>>,tuple<!db.string>>, %23 : tuple<!db.int<64>> -> !db.iterable<tuple<tuple<!db.int<64>>, tuple<!db.string>>,join_ht_iterator>
        %25 = db.for %arg5 in %24 : !db.iterable<tuple<tuple<!db.int<64>>, tuple<!db.string>>,join_ht_iterator>  iter_args(%arg6 = %arg4) -> (!db.table_builder<tuple<!db.string, !db.string>>) {
          %26:2 = util.unpack %arg5 : tuple<tuple<!db.int<64>>, tuple<!db.string>> -> tuple<!db.int<64>>, tuple<!db.string>
          %27 = util.unpack %26#0 : tuple<!db.int<64>> -> !db.int<64>
          %28 = util.unpack %26#1 : tuple<!db.string> -> !db.string
          %29 = db.compare eq %27 : !db.int<64>, %22#1 : !db.int<64>
          %30 = db.if %29 : !db.bool -> (!db.table_builder<tuple<!db.string, !db.string>>) {
            %31 = util.pack %28, %22#0 : !db.string, !db.string -> tuple<!db.string, !db.string>
            %32 = db.builder_merge %arg6 : !db.table_builder<tuple<!db.string, !db.string>>, %31 : tuple<!db.string, !db.string> -> !db.table_builder<tuple<!db.string, !db.string>>
            db.yield %32 : !db.table_builder<tuple<!db.string, !db.string>>
          } else {
            db.yield %arg6 : !db.table_builder<tuple<!db.string, !db.string>>
          }
          db.yield %30 : !db.table_builder<tuple<!db.string, !db.string>>
        }
        db.yield %25 : !db.table_builder<tuple<!db.string, !db.string>>
      }
      db.yield %21 : !db.table_builder<tuple<!db.string, !db.string>>
    }
    %19 = db.builder_build %18 : !db.table_builder<tuple<!db.string, !db.string>> -> !db.table
    return %19 : !db.table
  }
}

