#!/usr/bin/env python3
"""Step 5's client: decode for one k, and decode at depth where asked; run by scripts/bench-run.sh as BENCH_CLIENT.

    BENCH_CLIENT=experiments/2026-11-vllm-27b-tp2/scripts/sweep-client.py [SWEEP_DEPTH=1] \
        scripts/bench-run.sh OUTDIR LABEL REPS -- experiments/2026-11-vllm-27b-tp2/scripts/vllm-serve.sh <config>.args

1. decode: scripts/oai-bench.py, prose, structured and mixed, thinking off, 640 tokens (enough for the mixed answer to
   reach its JSON), REPS reps after a warm-up, MTP counters per request (label LABEL)
2. with SWEEP_DEPTH=1: scripts/long-bench.py --mode depth at about 1K, 32K and 131K tokens of context, a prose and a
   JSON task on the same document, 2 reps each, 512 tokens, into OUTDIR/depth.jsonl
Every request starts with its own random line (--unique-prefix), so vLLM's prefix cache serves nothing.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"


def run(cmd):
    print("== " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    for k in ("--url", "--label", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--reps", default="3")
    ap.add_argument("--load-s", default="")
    a = ap.parse_args()
    common = ["--url", a.url, "--unique-prefix"] + (["--load-s", a.load_s] if a.load_s else [])
    py = sys.executable
    run([py, str(SCRIPTS / "oai-bench.py"), *common, "--label", a.label, "--out", a.out, "--reps", a.reps,
         "--kinds", "prose,structured,mixed", "--max-tokens", "640"])
    if os.environ.get("SWEEP_DEPTH") == "1":
        run([py, str(SCRIPTS / "long-bench.py"), *common, "--label", a.label, "--mode", "depth",
             "--out", str(Path(a.out).parent / "depth.jsonl"),
             "--targets", os.environ.get("SWEEP_DEPTH_TARGETS", "1000,32768,131072"), "--reps", "2",
             "--max-tokens", "512"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
