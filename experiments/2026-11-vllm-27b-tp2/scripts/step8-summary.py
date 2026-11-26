#!/usr/bin/env python3
"""Summarize step 8 (run-step8.sh) as Markdown tables.

    experiments/2026-11-vllm-27b-tp2/scripts/step8-summary.py runs/8-parallel > runs/8-parallel/summary.md

Concurrency (results.jsonl, label NAME-cN, N requests at once): one row per configuration, kind and N, over the
measured repetitions (rep 0, the warm-up, left out).
  total t/s   generated tokens of the group over its wall time (first request sent -> last stream closed); the mean
              of the groups in which every stream ran to its token limit ("full groups"). A group where a request
              ended early (a short answer; the tag line of a concurrent request changes what the model answers) is
              counted in "short groups" and left out, as its token count is not the work asked for.
  TTFT s      per stream, mean (max): the wait in the queue is part of it
  decode t/s  per stream, mean of (tokens - 1) / (last - first) over the streams
Long prompt (long.jsonl): TTFT of about 32K tokens, per configuration.
KV pool from each server log.
"""
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def mean(xs, nd=1):
    xs = [x for x in xs if x is not None]
    return f"{statistics.mean(xs):.{nd}f}" if xs else "–"


def main(d):
    d = Path(d)
    groups = defaultdict(lambda: defaultdict(list))   # (config, n, kind) -> rep -> [stream records]
    for line in open(d / "results.jsonl"):
        r = json.loads(line)
        if r.get("warmup") or "concurrency" not in r:
            continue
        name, n = r["label"].rsplit("-c", 1)
        groups[(name, int(n), r["kind"])][r["rep"]].append(r)
    print("| configuration | requests at once | kind | full groups | total t/s | short groups | TTFT s mean (max) "
          "| decode t/s per stream | KV pool |")
    print("|---|--:|---|--:|--:|--:|---|--:|--:|")
    for (name, n, kind) in sorted(groups):
        reps = groups[(name, n, kind)]
        full = [rs for rs in reps.values() if not any(x["short"] for x in rs)]
        short = len(reps) - len(full)
        tot = [rs[0]["group_tokens"] / rs[0]["group_wall_s"] for rs in full]
        allr = [x for rs in reps.values() for x in rs]
        ttft = [x["ttft_s"] for x in allr]
        out = d / f"{name}.out"
        pool = re.findall(r"GPU KV cache size: ([\d,]+) tokens", out.read_text(errors="replace")) if out.exists() else []
        print(f"| {name} | {n} | {kind} | {len(full)} | {mean(tot)} | {short} | {mean(ttft, 2)} ({max(ttft):.2f})"
              f" | {mean([x['decode_tps'] for x in allr if not x['short']])} | {pool[-1] if pool else '–'} |")
    lp = d / "long.jsonl"
    if lp.exists():
        print()
        print("| configuration | prompt tokens | TTFT s | needle |")
        print("|---|--:|--:|--:|")
        by = defaultdict(list)
        for line in open(lp):
            r = json.loads(line)
            if "error" not in r:
                by[r["label"]].append(r)
        for name, rs in by.items():
            print(f"| {name} | {rs[0]['prompt_tokens']} | {mean([r['ttft_s'] for r in rs], 2)} "
                  f"| {sum(r['needle_found'] for r in rs)}/{len(rs)} |")


if __name__ == "__main__":
    main(sys.argv[1])
