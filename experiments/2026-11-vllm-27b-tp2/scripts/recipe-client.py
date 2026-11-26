#!/usr/bin/env python3
"""Step 7's client: everything measured on one server start, run by scripts/bench-run.sh as BENCH_CLIENT.

    BENCH_CLIENT=experiments/2026-11-vllm-27b-tp2/scripts/recipe-client.py \
        scripts/bench-run.sh OUTDIR LABEL 5 -- experiments/2026-11-vllm-27b-tp2/scripts/vllm-serve.sh <config>.args

1. scripts/baseline-client.py (step 4: shared decode, reproduction mode, TTFT at about 0.5K / 4K / 32K)
2. scripts/long-bench.py --mode depth at about 1K / 32K / 131K (step 5's depth pass), into OUTDIR/depth.jsonl
3. the needle: scripts/long-bench.py --mode ttft with a fixed 835,000-character document (252,872 tokens with this
   tokenizer, as in step 6), 3 reps, into OUTDIR/needle.jsonl
Every request starts with its own random line (--unique-prefix).
RECIPE_PASSES (default "baseline,depth,needle") picks the passes: the first run of step 7 lost 2 and 3 to a wrong path
to long-bench.py (fixed), and they were run on a second server start with RECIPE_PASSES=depth,needle.
"""
import os
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[2] / "scripts"


def run(cmd):
    print("== " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


ap = argparse.ArgumentParser()
for k in ("--url", "--label", "--out"):
    ap.add_argument(k, required=True)
ap.add_argument("--reps", default="5")
ap.add_argument("--load-s", default="")
a = ap.parse_args()
py = sys.executable
out = Path(a.out).parent
load = ["--load-s", a.load_s] if a.load_s else []
passes = os.environ.get("RECIPE_PASSES", "baseline,depth,needle").split(",")
if "baseline" in passes:
    run([py, str(HERE / "baseline-client.py"), "--url", a.url, "--label", a.label, "--out", a.out, "--reps", a.reps]
        + load)
if "depth" in passes:
    run([py, str(SCRIPTS / "long-bench.py"), "--url", a.url, "--label", a.label, "--mode", "depth", "--out",
         str(out / "depth.jsonl"), "--targets", "1000,32768,131072", "--reps", "2", "--max-tokens", "512",
         "--unique-prefix"] + load)
if "needle" in passes:
    run([py, str(SCRIPTS / "long-bench.py"), "--url", a.url, "--label", a.label, "--mode", "ttft", "--out",
         str(out / "needle.jsonl"), "--targets", "249450", "--chars", "835000", "--reps", "3", "--unique-prefix"]
        + load)
