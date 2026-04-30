#!/usr/bin/env python3
"""A/B-compare cardinality estimation quality across SQL queries under two
different runtime configurations of the lingodb compiler.

Both runs invoke the same `cardinality-errors` binary on the same query, only
the LINGODB_* environment variables differ. This makes the script a generic
A/B harness — point it at any setting toggled via env vars (any optimizer
pass, execution mode, etc.).

Examples:
    # Compare default behavior vs an experimental optimizer toggle
    cardinality-diff.py --b-env LINGODB_OPT_FOO=false --b-label no-foo

    # Two non-default configurations on a single TPC-DS query
    cardinality-diff.py --suite tpcds --query 43.sql \\
        --a-env LINGODB_OPT_FOO=true  --a-label foo \\
        --b-env LINGODB_OPT_FOO=false --b-label nofoo

    # Arbitrary suite (sql dir + database dir)
    cardinality-diff.py --sql-dir queries/ --db /path/to/db \\
        --b-env LINGODB_OPT_BAR=baz
"""

import argparse
import math
import os
import re
import subprocess
import sys
from glob import glob

DEFAULT_BIN = "build/lingodb-release/cardinality-errors"
DEFAULT_TPCH = "../lingo-db/resources/data/tpch-1"
DEFAULT_TPCDS = "../lingo-db/resources/data/tpcds-1"

# Matches summary lines like:
#   [overall]        n=  7  geomean=  21.10  max= 304.33
#   aggregation      n=  1  geomean= 304.33  max= 304.33
SUMMARY_RE = re.compile(
    r"^\s*(\S+)\s+n=\s*(\d+)\s+geomean=\s*([\d.naninf-]+)\s+max=\s*([\d.naninf-]+)\s*$"
)


def parse_env_args(items):
    """Parse a list of KEY=VALUE strings into a dict."""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--*-env expects KEY=VALUE, got: {item!r}")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def parse_summary(stdout):
    """Return {operator_kind -> (n, geomean, max)} from the summary block."""
    out = {}
    in_summary = False
    for line in stdout.splitlines():
        if line.strip() == "=== summary ===":
            in_summary = True
            continue
        if not in_summary:
            continue
        m = SUMMARY_RE.match(line)
        if not m:
            continue
        kind = m.group(1)
        n = int(m.group(2))
        try:
            geo = float(m.group(3))
        except ValueError:
            geo = math.nan
        try:
            mx = float(m.group(4))
        except ValueError:
            mx = math.nan
        out[kind] = (n, geo, mx)
    return out


def run_one(binary, sql_file, db_dir, extra_env, timeout):
    env = os.environ.copy()
    env.update(extra_env)
    try:
        proc = subprocess.run(
            [binary, sql_file, db_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        return None, msg[-1] if msg else f"exit {proc.returncode}"
    summary = parse_summary(proc.stdout.decode("utf-8", "replace"))
    if not summary:
        return None, "no summary"
    return summary, None


def fmt(v, spec="{:7.2f}"):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return f"{'-':>7}"
    return spec.format(v)


def pct(a, b):
    """Relative change b vs a, in %."""
    if a is None or b is None or a == 0 or math.isnan(a) or math.isnan(b):
        return "    -  "
    return f"{(b - a) / a * 100.0:+6.1f}%"


def run_suite(binary, suite, sql_files, db_dir, a_env, b_env, a_label, b_label, timeout):
    print(f"\n========== {suite}  (db={db_dir})  ==========")
    header = (
        f"{'query':<10}  "
        f"{a_label[:7]+' geo':>11}  {b_label[:7]+' geo':>11}  {'Δgeo':>7}  "
        f"{a_label[:7]+' max':>11}  {b_label[:7]+' max':>11}  {'Δmax':>7}  "
        f"note"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for sql in sql_files:
        qname = os.path.basename(sql).removesuffix(".sql")
        a, a_err = run_one(binary, sql, db_dir, a_env, timeout)
        b, b_err = run_one(binary, sql, db_dir, b_env, timeout)

        if a is None or b is None:
            note = a_err or b_err or "?"
            print(f"{qname:<10}  {'-':>11}  {'-':>11}  {'-':>7}  {'-':>11}  {'-':>11}  {'-':>7}  {note}")
            continue

        a_n, a_geo, a_max = a.get("[overall]", (0, math.nan, math.nan))
        b_n, b_geo, b_max = b.get("[overall]", (0, math.nan, math.nan))

        marker = ""
        if not math.isnan(a_geo) and not math.isnan(b_geo):
            if a_geo + 1e-9 < b_geo:
                marker = f"{a_label} better"
            elif b_geo + 1e-9 < a_geo:
                marker = f"{b_label} better"

        print(
            f"{qname:<10}  "
            f"{fmt(a_geo, '{:11.2f}')}  {fmt(b_geo, '{:11.2f}')}  {pct(a_geo, b_geo)}  "
            f"{fmt(a_max, '{:11.2f}')}  {fmt(b_max, '{:11.2f}')}  {pct(a_max, b_max)}  "
            f"{marker}"
        )
        rows.append((qname, a_geo, b_geo, a_max, b_max, a, b))

    if not rows:
        return rows

    print()
    valid = [r for r in rows if not math.isnan(r[1]) and not math.isnan(r[2])]
    n = len(valid)
    if n:
        a_geo_g = math.exp(sum(math.log(r[1]) for r in valid if r[1] > 0) / n)
        b_geo_g = math.exp(sum(math.log(r[2]) for r in valid if r[2] > 0) / n)
        a_max_max = max(r[3] for r in valid)
        b_max_max = max(r[4] for r in valid)
        a_better = sum(1 for r in valid if r[1] + 1e-9 < r[2])
        b_better = sum(1 for r in valid if r[2] + 1e-9 < r[1])
        same = n - a_better - b_better
        print(
            f"{suite} summary: queries={n}  "
            f"geomean(overall geo): {a_label}={a_geo_g:.3f}  {b_label}={b_geo_g:.3f}  "
            f"max(overall max): {a_label}={a_max_max:.2f}  {b_label}={b_max_max:.2f}  "
            f"{a_label}/{b_label}/same better = {a_better}/{b_better}/{same}"
        )
    return rows


def per_kind_diff(suite_name, rows, a_label, b_label):
    """Aggregate per-operator-kind geomean q-error across all queries in the
    suite."""
    kinds = {}  # kind -> (a_list, b_list)
    for _, _, _, _, _, a, b in rows:
        for k, (n, geo, _mx) in a.items():
            if k == "[overall]" or n == 0 or math.isnan(geo) or geo <= 0:
                continue
            kinds.setdefault(k, ([], []))[0].append(geo)
        for k, (n, geo, _mx) in b.items():
            if k == "[overall]" or n == 0 or math.isnan(geo) or geo <= 0:
                continue
            kinds.setdefault(k, ([], []))[1].append(geo)
    if not kinds:
        return
    print(f"\n{suite_name} per-operator-kind geomean (across queries that exercise it):")
    print(f"  {'kind':<14}  {a_label[:9]+' geo':>13}  {b_label[:9]+' geo':>13}  {'Δ':>7}")
    for kind in sorted(kinds):
        a_list, b_list = kinds[kind]
        a_g = math.exp(sum(math.log(x) for x in a_list) / len(a_list)) if a_list else math.nan
        b_g = math.exp(sum(math.log(x) for x in b_list) / len(b_list)) if b_list else math.nan
        print(f"  {kind:<14}  {fmt(a_g, '{:13.3f}')}  {fmt(b_g, '{:13.3f}')}  {pct(a_g, b_g)}")


def query_sortkey(path):
    stem = os.path.basename(path).removesuffix(".sql")
    m = re.match(r"^(\d+)([a-z]*)$", stem)
    return (int(m.group(1)), m.group(2)) if m else (10**9, stem)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--bin", default=DEFAULT_BIN, help="cardinality-errors binary")
    p.add_argument("--tpch", default=DEFAULT_TPCH, help="TPC-H database directory")
    p.add_argument("--tpcds", default=DEFAULT_TPCDS, help="TPC-DS database directory")
    p.add_argument("--suite", choices=["tpch", "tpcds", "all"], default="all",
                   help="built-in suite shortcut (default: all)")
    p.add_argument("--query", default=None,
                   help="restrict to query files matching this glob (e.g. '5.sql', '4*.sql')")
    p.add_argument("--sql-dir", default=None,
                   help="custom SQL directory (overrides --suite)")
    p.add_argument("--db", default=None,
                   help="custom database directory (used with --sql-dir)")
    p.add_argument("--timeout", type=int, default=300, help="per-run timeout in seconds")
    p.add_argument("--a-label", default="A", help="label for run A (default: A)")
    p.add_argument("--b-label", default="B", help="label for run B (default: B)")
    p.add_argument("--a-env", action="append", default=[], metavar="KEY=VAL",
                   help="env var for run A (repeatable)")
    p.add_argument("--b-env", action="append", default=[], metavar="KEY=VAL",
                   help="env var for run B (repeatable)")
    args = p.parse_args()

    if not os.path.isfile(args.bin):
        print(f"binary not found: {args.bin}", file=sys.stderr)
        sys.exit(1)

    a_env = parse_env_args(args.a_env)
    b_env = parse_env_args(args.b_env)
    if not a_env and not b_env:
        print("warning: no --a-env / --b-env given; both runs will be identical",
              file=sys.stderr)

    suites = []
    if args.sql_dir:
        if not args.db:
            print("--sql-dir requires --db", file=sys.stderr)
            sys.exit(1)
        suites.append(("custom", args.sql_dir, args.db))
    else:
        if args.suite in ("tpch", "all"):
            suites.append(("tpch", "resources/sql/tpch", args.tpch))
        if args.suite in ("tpcds", "all"):
            suites.append(("tpcds", "resources/sql/tpcds", args.tpcds))

    for name, sql_dir, db_dir in suites:
        if not os.path.isdir(db_dir):
            print(f"skipping {name}: db dir missing ({db_dir})", file=sys.stderr)
            continue
        pattern = args.query if args.query else "*.sql"
        files = sorted(
            (f for f in glob(os.path.join(sql_dir, pattern))
             if os.path.basename(f) != "initialize.sql"),
            key=query_sortkey,
        )
        if not files:
            print(f"skipping {name}: no SQL files found in {sql_dir}", file=sys.stderr)
            continue
        rows = run_suite(args.bin, name, files, db_dir, a_env, b_env,
                         args.a_label, args.b_label, args.timeout)
        per_kind_diff(name, rows, args.a_label, args.b_label)


if __name__ == "__main__":
    main()
