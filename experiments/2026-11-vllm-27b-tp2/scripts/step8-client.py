#!/usr/bin/env python3
"""Step 8's client, run by scripts/bench-run.sh as BENCH_CLIENT: concurrency and one long prompt on one server start.

    BENCH_CLIENT=experiments/2026-11-vllm-27b-tp2/scripts/step8-client.py \
        scripts/bench-run.sh OUTDIR LABEL 3 -- experiments/2026-11-vllm-27b-tp2/scripts/vllm-serve.sh <config>.args

1. scripts/oai-bench.py --concurrency 1,2,4: prose and structured, thinking off, 256 tokens, 3 reps after a warm-up,
   each level as LABEL-cN (per stream TTFT and decode, and the group's total t/s; with --max-num-seqs below N the
   extra requests queue, so their TTFT includes the wait)
2. scripts/long-bench.py --mode ttft at about 32K tokens, 2 reps, into OUTDIR/long.jsonl
Every request starts with its own random line (--unique-prefix).
"""
import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
ap = argparse.ArgumentParser()
for k in ("--url", "--label", "--out"):
    ap.add_argument(k, required=True)
ap.add_argument("--reps", default="3")
ap.add_argument("--load-s", default="")
a = ap.parse_args()
common = ["--url", a.url, "--unique-prefix"] + (["--load-s", a.load_s] if a.load_s else [])
py = sys.executable
for cmd in (
    [py, str(SCRIPTS / "oai-bench.py"), *common, "--label", a.label, "--out", a.out, "--reps", a.reps,
     "--kinds", "prose,structured", "--max-tokens", "256", "--concurrency", "1,2,4"],
    [py, str(SCRIPTS / "long-bench.py"), *common, "--label", a.label, "--mode", "ttft",
     "--out", str(Path(a.out).parent / "long.jsonl"), "--targets", "34700", "--reps", "2"],
):
    print("== " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
