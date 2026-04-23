"""Prepared-statement test for the LingoDB Flight SQL server.

Usage:
    python test_prepared.py [--host 127.0.0.1] [--port 31337]
"""

from __future__ import annotations

import argparse
import sys

import pyarrow as pa
import adbc_driver_flightsql.dbapi as flight_sql


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=31337, type=int)
    args = parser.parse_args()

    uri = f"grpc://{args.host}:{args.port}"
    print(f"[client] connecting to {uri}")

    with flight_sql.connect(uri) as conn:
        with conn.cursor() as cur:
            # --- 1. No-parameter prepared statement -------------------------
            param_schema = cur.adbc_prepare("select count(*) from studenten")
            print(f"[client] prepare() reported param_schema = {param_schema}")
            assert len(param_schema) == 0

            cur.execute_partial_result_set = None  # make sure we're on the normal path
            cur.execute("select count(*) from studenten")
            tbl = cur.fetch_arrow_table()
            assert tbl.num_rows == 1
            assert tbl.column(0)[0].as_py() == 8
            print(f"[client] no-parameter prepared: count = {tbl.column(0)[0].as_py()}")

            # --- 2. Single integer parameter --------------------------------
            sql = "select name, semester from studenten where matrnr = ?"
            param_schema = cur.adbc_prepare(sql)
            print(f"[client] prepare({sql!r}) -> params: {param_schema}")
            assert len(param_schema) == 1

            cur.execute(sql, parameters=[26120])
            tbl = cur.fetch_arrow_table()
            print(f"[client] integer-param result: {tbl.to_pydict()}")
            assert tbl.num_rows == 1
            assert tbl.column("name")[0].as_py() == "Fichte"
            assert tbl.column("semester")[0].as_py() == 10

            # Re-execute with a different value to verify rebinding.
            cur.execute(sql, parameters=[29555])
            tbl = cur.fetch_arrow_table()
            print(f"[client] rebind result: {tbl.to_pydict()}")
            assert tbl.num_rows == 1
            assert tbl.column("name")[0].as_py() == "Feuerbach"

            # --- 3. Multiple parameters of mixed types ----------------------
            sql = "select name from studenten where semester >= ? and name <> ?"
            cur.execute(sql, parameters=[8, "Fichte"])
            tbl = cur.fetch_arrow_table()
            names = sorted(tbl.column("name").to_pylist())
            print(f"[client] mixed-param result: {names}")
            expected = sorted(["Xenokrates", "Jonas", "Aristoxenos"])
            assert names == expected, f"{names} != {expected}"

            # --- 4. String parameter with an embedded single-quote ----------
            cur.execute(
                "select ? as greeting, ? as escaped",
                parameters=["hello", "it's a test"],
            )
            tbl = cur.fetch_arrow_table()
            assert tbl.column("greeting")[0].as_py() == "hello"
            assert tbl.column("escaped")[0].as_py() == "it's a test"
            print("[client] single-quote escaping OK")

            # --- 5. NULL parameter ------------------------------------------
            cur.execute(
                "select ? is null as was_null, coalesce(?, 42) as fallback",
                parameters=[None, None],
            )
            tbl = cur.fetch_arrow_table()
            print(f"[client] null-param result: {tbl.to_pydict()}")
            assert tbl.column("was_null")[0].as_py() is True
            assert tbl.column("fallback")[0].as_py() == 42

            # --- 6. Floating-point parameter --------------------------------
            cur.execute("select ? * 2 as doubled", parameters=[3.5])
            tbl = cur.fetch_arrow_table()
            assert tbl.column("doubled")[0].as_py() == 7.0
            print("[client] float-param OK")

            # --- 7. ClosePreparedStatement happens implicitly on cursor close
    print("[client] all prepared-statement assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
