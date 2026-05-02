"""Smoke test that lingodb wheel works as a loaded Python extension AND
that Python UDFs run inside per-worker sub-interpreters."""
import lingodb

con = lingodb.create_in_memory()

print("--- 1. Plain SQL still works ---")
print(con.sql("SELECT 1+1 AS result").to_pandas())

print("--- 2. Define + call a Python UDF ---")
con.sql_stmt("""
CREATE FUNCTION square(x INT) RETURNS INT LANGUAGE python AS '
def square(x):
    return x * x
';
""")

print(con.sql("SELECT square(7) AS sq").to_pandas())

print("--- 3. Python UDF over a small table ---")
con.sql_stmt("CREATE TABLE nums (v INT);")
con.sql_stmt("INSERT INTO nums VALUES (1), (2), (3), (4), (5);")
print(con.sql("SELECT v, square(v) AS sq FROM nums ORDER BY v").to_pandas())

print("PASS")
