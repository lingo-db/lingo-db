shopt -s extglob
set -e
set -x
#./build/lingodb-debug/run-sql resources/sql/tpcds/10.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/11.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/12.sql resources/data/tpcds-1 (fix by solving #30)
#./build/lingodb-debug/run-sql resources/sql/tpcds/13.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/14a.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/14b.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/15.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/16.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/17.sql resources/data/tpcds-1 (fix by solving #27)
#./build/lingodb-debug/run-sql resources/sql/tpcds/18.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/19.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/1.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/20.sql resources/data/tpcds-1 (fix by solving #30)
#./build/lingodb-debug/run-sql resources/sql/tpcds/21.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/22.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/23a.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/23b.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/24a.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/24b.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/25.sql resources/data/tpcds-1 (fix by solving #27)
#./build/lingodb-debug/run-sql resources/sql/tpcds/26.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/27.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/28.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/29.sql resources/data/tpcds-1 (fix by solving #27)
#./build/lingodb-debug/run-sql resources/sql/tpcds/2.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/30.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/31.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/32.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/33.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/34.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/35.sql resources/data/tpcds-1 (fix by solving #27)
#./build/lingodb-debug/run-sql resources/sql/tpcds/36.sql resources/data/tpcds-1 (fix by solving #29) -> rank()
#./build/lingodb-debug/run-sql resources/sql/tpcds/37.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/38.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/39a.sql resources/data/tpcds-1 (fix by solving #27)
#./build/lingodb-debug/run-sql resources/sql/tpcds/39b.sql resources/data/tpcds-1 (fix by solving #27)
#./build/lingodb-debug/run-sql resources/sql/tpcds/3.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/40.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/41.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/42.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/43.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/44.sql resources/data/tpcds-1 (fix by solving #30)
#./build/lingodb-debug/run-sql resources/sql/tpcds/45.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/46.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/47.sql resources/data/tpcds-1 (fix by solving #30)
#./build/lingodb-debug/run-sql resources/sql/tpcds/48.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/49.sql resources/data/tpcds-1 (fix by solving #30)
#./build/lingodb-debug/run-sql resources/sql/tpcds/4.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/50.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/51.sql resources/data/tpcds-1 (fix by solving #30)
#./build/lingodb-debug/run-sql resources/sql/tpcds/52.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/53.sql resources/data/tpcds-1 (fix by solving #30)
#./build/lingodb-debug/run-sql resources/sql/tpcds/54.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/55.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/56.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/57.sql resources/data/tpcds-1 (fix by solving #30)
#./build/lingodb-debug/run-sql resources/sql/tpcds/58.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/59.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/5.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/60.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/61.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/62.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/63.sql resources/data/tpcds-1 (fix by solving #30)
#./build/lingodb-debug/run-sql resources/sql/tpcds/64.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/65.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/66.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/67.sql resources/data/tpcds-1 (fix by solving #29) -> rank()
#./build/lingodb-debug/run-sql resources/sql/tpcds/68.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/69.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/6.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/70.sql resources/data/tpcds-1 (fix by solving #30)
#./build/lingodb-debug/run-sql resources/sql/tpcds/71.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/72.sql resources/data/tpcds-1 (fix by solving #28)
#./build/lingodb-debug/run-sql resources/sql/tpcds/73.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/74.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/75.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/76.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/77.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/78.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/79.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/7.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/80.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/81.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/82.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/83.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/84.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/85.sql resources/data/tpcds-1 timeout
#./build/lingodb-debug/run-sql resources/sql/tpcds/86.sql resources/data/tpcds-1 (fix by solving #29) -> rank(), grouping
#./build/lingodb-debug/run-sql resources/sql/tpcds/87.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/88.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/89.sql resources/data/tpcds-1 (fix by solving #30)
#./build/lingodb-debug/run-sql resources/sql/tpcds/8.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/90.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/91.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/92.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/93.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/94.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/95.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/96.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/97.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/98.sql resources/data/tpcds-1 (fix by solving #30)
#./build/lingodb-debug/run-sql resources/sql/tpcds/99.sql resources/data/tpcds-1
#./build/lingodb-debug/run-sql resources/sql/tpcds/9.sql resources/data/tpcds-1

