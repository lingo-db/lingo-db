"""Heavier test: enough rows to exercise multiple workers/sub-interpreters."""
import lingodb
import pandas as pd
import pyarrow as pa

con = lingodb.create_in_memory()

con.sql_stmt("""
CREATE FUNCTION square(x INT) RETURNS BIGINT LANGUAGE python AS '
def square(x):
    return x * x
';
""")
con.sql_stmt("""
CREATE FUNCTION add_one(x INT) RETURNS BIGINT LANGUAGE python AS '
def add_one(x):
    return x + 1
';
""")

# Insert 100000 rows via Arrow ingestion
df = pd.DataFrame({"v": list(range(1, 100001))})
schema = pa.schema([("v", pa.int32())])
con.add_table("big", pa.Table.from_pandas(df, schema=schema))

result = con.sql("SELECT SUM(square(v)) AS s, SUM(add_one(v)) AS s2 FROM big").to_pandas()
print(result)

expected_sq = sum(i*i for i in range(1, 100001))
expected_add = sum(i+1 for i in range(1, 100001))
got_sq = int(result.iloc[0]['s'])
got_add = int(result.iloc[0]['s2'])
assert got_sq == expected_sq, f"sq: got {got_sq}, expected {expected_sq}"
assert got_add == expected_add, f"add: got {got_add}, expected {expected_add}"
print("HEAVY PASS")

# Run a second query in the same connection — exercises sub-interpreter reuse
result2 = con.sql("SELECT COUNT(*) AS c FROM big WHERE square(v) > 100").to_pandas()
print(result2)
print("REUSE PASS")
