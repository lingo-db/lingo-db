"""End-to-end test for the LingoDB Flight SQL server.

Usage:
    python test_client.py [--host 127.0.0.1] [--port 31337]

Expects the server to be running against the 'uni' dataset.
"""

from __future__ import annotations

import argparse
import sys

import adbc_driver_flightsql.dbapi as flight_sql


def row_count(cursor) -> int:
    cursor.execute("select count(*) from studenten")
    (rows,) = cursor.fetchone()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=31337, type=int)
    args = parser.parse_args()

    uri = f"grpc://{args.host}:{args.port}"
    print(f"[client] connecting to {uri}")

    with flight_sql.connect(uri) as conn:
        with conn.cursor() as cur:
            # 1) Trivial literal query (no catalog access)
            cur.execute("select 1 + 2 as x, 'hello' as greeting")
            tbl = cur.fetch_arrow_table()
            print("[client] literal query result:")
            print(tbl)
            assert tbl.num_rows == 1
            assert tbl.column("x")[0].as_py() == 3
            assert tbl.column("greeting")[0].as_py() == "hello"

            # 2) Catalog query against the uni dataset
            n = row_count(cur)
            print(f"[client] studenten count: {n}")
            assert n == 8, f"expected 8 rows, got {n}"

            # 3) Ordering + projection
            cur.execute(
                "select name, semester from studenten order by semester desc limit 3"
            )
            tbl = cur.fetch_arrow_table()
            print("[client] top 3 by semester:")
            print(tbl)
            assert tbl.num_rows == 3
            semesters = tbl.column("semester").to_pylist()
            assert semesters == sorted(semesters, reverse=True)

            # 4) Join
            cur.execute(
                """select p.name as professor, v.titel as lecture
                   from professoren p join vorlesungen v on v.gelesenvon = p.persnr
                   order by p.name, v.titel"""
            )
            tbl = cur.fetch_arrow_table()
            print(f"[client] join returned {tbl.num_rows} rows:")
            print(tbl.to_pandas().head(10).to_string(index=False))
            assert tbl.num_rows == 10  # from the uni seed data

            # 5) Expected failure: syntax error should surface as a client-side error
            try:
                cur.execute("select * from no_such_table")
                cur.fetch_arrow_table()
            except Exception as exc:
                print(f"[client] expected error propagated: {type(exc).__name__}: {exc}")
            else:
                raise AssertionError("expected query against missing table to fail")

    print("[client] all assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
