set persist=1;
create table test(
    str varchar(20),
    float32 float(2),
    float64 float(4),
    decimal decimal(5, 2),
    int32 int,
    int64 bigint,
    bool bool,
    date32 date,
    date64 timestamp,
    char1 char(1),
    char20 char(20),
    primary key(float64)
);

INSERT into test(str, float32, float64, decimal, int32, int64, bool, date32, date64, char1, char20) values ('str', 1.1, 1.1, 1.10, 1, 1, 1, '1996-01-02', '1996-01-02 13:37','a','abcdefghijklmnopqrst'), (null, null, null, null, null, null, null, null, null, null, null);
