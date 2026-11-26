#!/usr/bin/env python3
"""Step 4's client: three passes on one server start, run by scripts/bench-run.sh as BENCH_CLIENT.

    BENCH_CLIENT=experiments/2026-11-vllm-27b-tp2/scripts/baseline-client.py \
        scripts/bench-run.sh OUTDIR LABEL REPS -- experiments/2026-11-vllm-27b-tp2/scripts/vllm-serve.sh <config>.args

1. shared:       scripts/oai-bench.py, prose and structured, thinking off, 256 tokens, REPS reps (label LABEL)
2. reproduction: scripts/oai-bench.py, prose, structured and mixed, thinking on, 800 tokens, 3 reps (LABEL-repro),
                 the source's way of counting: reasoning tokens are decode tokens
3. TTFT:         scripts/long-bench.py --mode ttft at targets that give about the source's 523 / 4,123 / 32,812 prompt
                 tokens, 3 reps each, into OUTDIR/long.jsonl
Every request starts with its own random line (--unique-prefix): vLLM's prefix cache would otherwise serve a repeated
prompt from cache and the TTFT would not be a prompt read. Passes 1 and 2 record vLLM's MTP counters per request.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
TTFT_TARGETS = os.environ.get("BASELINE_TTFT_TARGETS", "700,4500,34700")


def run(cmd):
    print("== " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    for k in ("--url", "--label", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--reps", default="5")
    ap.add_argument("--load-s", default="")
    a = ap.parse_args()
    common = ["--url", a.url, "--unique-prefix"] + (["--load-s", a.load_s] if a.load_s else [])
    py = sys.executable
    run([py, str(SCRIPTS / "oai-bench.py"), *common, "--label", a.label, "--out", a.out, "--reps", a.reps,
         "--kinds", "prose,structured", "--max-tokens", "256"])
    run([py, str(SCRIPTS / "oai-bench.py"), *common, "--label", a.label + "-repro", "--out", a.out, "--reps", "3",
         "--kinds", "prose,structured,mixed", "--max-tokens", "800", "--thinking"])
    run([py, str(SCRIPTS / "long-bench.py"), *common, "--label", a.label, "--mode", "ttft",
         "--out", str(Path(a.out).parent / "long.jsonl"), "--targets", TTFT_TARGETS, "--reps", "3"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
