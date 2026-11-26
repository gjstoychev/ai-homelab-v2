#!/usr/bin/env python3
"""Step 9g's client: profile one prefill of about 4.2K tokens with vLLM's torch profiler, run by scripts/bench-run.sh
as BENCH_CLIENT against configs/fp8-nomtp-profile.args.

    BENCH_CLIENT=experiments/2026-11-vllm-27b-tp2/scripts/profile-client.py \
        scripts/bench-run.sh OUTDIR LABEL 1 -- experiments/2026-11-vllm-27b-tp2/scripts/vllm-serve.sh <config>.args

Two unprofiled warm-up requests (unique prefix each, so the prefix cache serves none of them), then /start_profile, one
request of the same size (max_tokens 1: the profile holds the prefill and one sampled token), /stop_profile. The
profiler adds overhead: its TTFT is not a measurement of the server, the kernel times in the trace are.
Writes OUTDIR/profile.jsonl; the traces land in the cache directory mounted into the container.
"""
import argparse
import importlib.util
import json
import sys
import urllib.request
import uuid
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
spec = importlib.util.spec_from_file_location("long_bench", SCRIPTS / "long-bench.py")
lb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lb)

ap = argparse.ArgumentParser()
for k in ("--url", "--label", "--out"):
    ap.add_argument(k, required=True)
ap.add_argument("--reps", default="1")
ap.add_argument("--load-s", default="")
ap.add_argument("--target", type=int, default=4500)
a = ap.parse_args()


def post_empty(path):
    req = urllib.request.Request(a.url + path, b"", {"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.status


text = lb.corpus(Path.home() / "strata/v0.2.21/src/docs")
n, cpt = lb.sized(a.url, text, a.target, 600)
doc = lb.document(text, n, "The secret code is obsidian-2824.")


def request():
    msgs = [{"role": "user", "content": f"[{uuid.uuid4()}]\n{doc}\n\nWhat is the secret code? Answer in one line."}]
    return lb.chat(a.url, msgs, 1, 600, False)[1]


rows = []
for i in range(2):
    m = request()
    rows.append({"phase": f"warmup{i}", **m})
    print(f"warm-up {i}: prompt {m['prompt_tokens']} tok, ttft {m['ttft_s']} s", flush=True)
print("start_profile", post_empty("/start_profile"), flush=True)
m = request()
rows.append({"phase": "profiled", **m})
print(f"profiled: prompt {m['prompt_tokens']} tok, ttft {m['ttft_s']} s", flush=True)
print("stop_profile", post_empty("/stop_profile"), flush=True)
with open(Path(a.out).parent / "profile.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps({"label": a.label, "chars": n, "chars_per_token": round(cpt, 3), **r}) + "\n")
