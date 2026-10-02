--// A query that throws (here: constant folding of an invalid cast) must make
--// sqlite-tester fail. It used to print the error and carry on (exit code 0),
--// reading past the end of the file for the error message.
--// RUN: split-file %s %t
--// RUN: not sqlite-tester %t/error.test 2>&1 | FileCheck %s
--// CHECK: ERROR: stoll
--// CHECK-NEXT: while executing query: query tsv rowsort 1
--// CHECK-NOT: executing:query tsv rowsort 2

//--- error.test
query tsv rowsort 1
select cast('abc' as int);
----
1

query tsv rowsort 2
select 2;
----
2
