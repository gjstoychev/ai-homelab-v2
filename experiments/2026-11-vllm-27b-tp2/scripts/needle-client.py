#!/usr/bin/env python3
"""Step 6's client: one long needle prompt, run by scripts/bench-run.sh as BENCH_CLIENT.

    BENCH_CLIENT=experiments/2026-11-vllm-27b-tp2/scripts/needle-client.py [NEEDLE_TARGET=262000] \
        scripts/bench-run.sh OUTDIR LABEL REPS -- experiments/2026-11-vllm-27b-tp2/scripts/vllm-serve.sh <config>.args

scripts/long-bench.py --mode ttft with one target and REPS reps (max_tokens 64, thinking off, a random first line per
request so no rep is served from the prefix cache), into OUTDIR/needle.jsonl. long-bench.py sizes the document 5 %
below its target by a noisy calibration (262,000 overshot the 262,144 context; 250,000 gave 225K tokens). NEEDLE_CHARS=n
fixes the document size in characters instead (the source's needle: 249,450 tokens in 246.7 s).
"""
import argparse
import os
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
cmd = [sys.executable, str(SCRIPTS / "long-bench.py"), "--url", a.url, "--label", a.label, "--mode", "ttft",
       "--out", str(Path(a.out).parent / "needle.jsonl"), "--targets", os.environ.get("NEEDLE_TARGET", "262000"),
       "--reps", a.reps, "--unique-prefix"] + (["--load-s", a.load_s] if a.load_s else []) \
      + (["--chars", os.environ["NEEDLE_CHARS"]] if os.environ.get("NEEDLE_CHARS") else [])
print("== " + " ".join(cmd), flush=True)
sys.exit(subprocess.run(cmd).returncode)
