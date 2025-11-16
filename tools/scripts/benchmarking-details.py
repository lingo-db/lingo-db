#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import pandas as pd

output_file = sys.argv[3]
# ---- CONFIG ----
DATASET = sys.argv[1]
BASE_DATASET = DATASET.split("-")[0]
print("base dataset:", BASE_DATASET)
QUERIES_DIR = Path(f"resources/sql/{BASE_DATASET}")
BINARY = Path("./build/lingodb-release/sql")
DB_BASE_DIR = Path("resources/data/")
BASE_ENV = {
    "LINGODB_EXECUTION_MODE": sys.argv[2],
    "LINGODB_SQL_PROMPT": "0",
    "LINGODB_SQL_REPORT_TIMES": "1",
    "LINGODB_SQL_REPORT_FORMAT": "csv",
}
WARMUP_RUNS = 3
MEASURED_RUNS = 10
# Max seconds to wait for a CSV result after sending a query
RESULT_TIMEOUT_S = 60.0

# Scenario configurations
SCENARIOS = {
    "Baseline": ({
        "LINGODB_OPT_PATTERNS_EXTRA_OPT": "true",
        "LINGODB_OPT_PUSHDOWN_RESTRICTIONS": "false",
        "LINGODB_OPT_ELIMINATE_NULLABLE": "false",
    },BINARY),
    "CheaperPatterns": ({
        "LINGODB_OPT_PATTERNS_EXTRA_OPT": "false",
        "LINGODB_OPT_PUSHDOWN_RESTRICTIONS": "false",
        "LINGODB_OPT_ELIMINATE_NULLABLE": "false",
    }, BINARY),
    "Restrictions":({
        "LINGODB_OPT_PATTERNS_EXTRA_OPT": "false",
        "LINGODB_OPT_PUSHDOWN_RESTRICTIONS": "true",
        "LINGODB_OPT_ELIMINATE_NULLABLE": "false",
    }, BINARY),
    "EliminateNulls":({
        "LINGODB_OPT_PATTERNS_EXTRA_OPT": "false",
        "LINGODB_OPT_PUSHDOWN_RESTRICTIONS": "true",
        "LINGODB_OPT_ELIMINATE_NULLABLE": "true",
    }, BINARY)
}
# ----------------


class LinePump(threading.Thread):
    """Continuously read lines from a file-like and push into a Queue."""
    def __init__(self, stream, out_q: queue.Queue[str], name: str):
        super().__init__(name=name, daemon=True)
        self._stream = stream
        self._q = out_q
        self._stop = threading.Event()

    def run(self):
        try:
            for line in iter(self._stream.readline, ''):
                if self._stop.is_set():
                    break
                self._q.put(line.rstrip('\n'))
        finally:
            # Signal EOF with None
            self._q.put(None)

    def stop(self):
        self._stop.set()


@dataclass
class Timings:
    execution_ms: float
    compilation_ms: float
    client_total_ms: Optional[float] = None


class Runner:
    """
    Start the sql binary once, send queries via stdin, get timings from stderr.
    Expects a CSV header line containing both 'execution' and 'QOpt', followed by one data row.
    """
    def __init__(self, binary: Path, db_base_dir: Path, dataset: str, extra_env: Dict[str, str]):
        self._cmd = [str(binary), str(db_base_dir / dataset)]
        self._env = {**os.environ, **extra_env}
        self._p: Optional[subprocess.Popen] = None
        self._stderr_q: queue.Queue[Optional[str]] = queue.Queue()
        self._stdout_q: queue.Queue[Optional[str]] = queue.Queue()
        self._stderr_pump: Optional[LinePump] = None
        self._stdout_pump: Optional[LinePump] = None
        self._lock = threading.Lock()  # serialize .execute calls

    def __enter__(self):
        # Use text mode and line buffering to make readline non-blocking-friendly
        self._p = subprocess.Popen(
            self._cmd,
            env=self._env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )
        assert self._p.stdin and self._p.stdout and self._p.stderr

        self._stderr_pump = LinePump(self._p.stderr, self._stderr_q, name="stderr-pump")
        self._stdout_pump = LinePump(self._p.stdout, self._stdout_q, name="stdout-pump")
        self._stderr_pump.start()
        self._stdout_pump.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._p and self._p.stdin:
                try:
                    self._p.stdin.close()
                except Exception:
                    pass
            if self._stderr_pump:
                self._stderr_pump.stop()
            if self._stdout_pump:
                self._stdout_pump.stop()
            if self._p:
                # Give the process a moment to exit gracefully
                try:
                    self._p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        self._p.terminate()
                        self._p.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self._p.kill()
        finally:
            pass

    def _send(self, sql: str):
        assert self._p and self._p.stdin
        # Ensure each statement is terminated and a newline is sent
        text = sql if sql.rstrip().endswith(";") else (sql.rstrip() + ";\n")
        if not text.endswith("\n"):
            text += "\n"
        self._p.stdin.write(text)
        self._p.stdin.flush()

    @staticmethod
    def _is_header(line: str) -> bool:
        # Heuristic from your previous code: header contains both "execution" and "QOpt"
        l = line.lower()
        return ("execution" in l) and ("qopt" in l) and ("," in l)

    def _read_result_from_stderr(self, timeout_s: float) -> Timings:
        """
        Wait on stderr queue for a CSV header + one data row.
        Return parsed timings. Raises TimeoutError / RuntimeError on issues.
        """
        deadline = time.time() + timeout_s
        header: Optional[str] = None

        # 1) Find the header line
        while time.time() < deadline:
            try:
                item = self._stderr_q.get(timeout=0.1)
            except queue.Empty:
                # check if process died
                if self._p and (self._p.poll() is not None):
                    raise RuntimeError("Process exited before producing results.")
                continue

            if item is None:
                # EOF on stderr
                raise RuntimeError("stderr closed before producing results.")

            line = item.strip()
            if not line:
                continue
            # Fail fast on explicit errors your engine might print
            if line.startswith("ERROR:") or line.startswith("what():"):
                raise RuntimeError(f"Engine error: {line}")

            if self._is_header(line):
                header = line
                break

        if header is None:
            raise TimeoutError("Timed out waiting for CSV header on stderr.")

        # 2) Next non-empty line is the data row
        data_row: Optional[str] = None
        while time.time() < deadline:
            try:
                item = self._stderr_q.get(timeout=0.1)
            except queue.Empty:
                if self._p and (self._p.poll() is not None):
                    raise RuntimeError("Process exited before producing data row.")
                continue

            if item is None:
                raise RuntimeError("stderr closed before producing data row.")

            line = item.strip()
            if not line:
                continue
            data_row = line
            break

        if data_row is None:
            raise TimeoutError("Timed out waiting for CSV data row on stderr.")

        # 3) Parse CSV for fields we care about
        # We keep it flexible: find the columns by name
        header_cols = next(csv.reader([header]))
        data_cols = next(csv.reader([data_row]))

        return {k:v for k,v in zip(header_cols,data_cols)}



    def execute(self, sql: str, timeout_s: float = RESULT_TIMEOUT_S) -> Timings:
        with self._lock:
            self._send(sql)
            return self._read_result_from_stderr(timeout_s)


def read_sql_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return text if text.rstrip().endswith(";") else text.rstrip() + ";"


def main():
    # Basic arg override: allow queries dir or a single file via CLI, optional
    # Usage examples:
    #   python run_tpch.py                     -> use QUERIES_DIR
    #   python run_tpch.py path/to/query.sql   -> run only that file
    target = QUERIES_DIR

    if BINARY.is_dir():
        print(f"ERROR: BINARY path looks like a directory: {BINARY}", file=sys.stderr)
        sys.exit(2)
    if not BINARY.exists():
        print(f"ERROR: BINARY not found: {BINARY}", file=sys.stderr)
        sys.exit(2)

    results=[]

    for scenario_name, (scenario_env, binary) in SCENARIOS.items():
        print(f"\nRunning scenario: {scenario_name}")
        
        # Combine base environment with scenario-specific environment
        full_env = {**BASE_ENV, **scenario_env}
        
        with Runner(binary, DB_BASE_DIR, DATASET, full_env) as runner:
            sql_files: List[Path]
            if target.is_file():
                sql_files = [target]
            else:
                sql_files = sorted([p for p in target.iterdir() if p.suffix.lower() == ".sql"])

            for sql_path in sql_files:
                name = sql_path.name
                if "initialize" in name.lower():
                    continue

                sql = read_sql_file(sql_path)

                # Warmups
                for _ in range(WARMUP_RUNS):
                    try:
                        runner.execute(sql)
                    except Exception as e:
                        print(f"[warmup] {scenario_name}/{name}: {e}", file=sys.stderr)

                for i in range(MEASURED_RUNS):
                    try:
                        t = runner.execute(sql)
                        print(t)
                        t["scenario"] = scenario_name
                        t["query"] = name
                        t["run"] = i
                        results.append(t)

                    except Exception as e:
                        print(f"[run {i+1}] {scenario_name}/{name}: {e}", file=sys.stderr)

    df=pd.DataFrame(results)
    df.to_csv(output_file,index=False)
    print(f"Wrote {output_file}")


if __name__ == "__main__":
    # Make Ctrl+C kill child quickly
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()
