#!/usr/bin/env python3
"""Step 9g: what the GPUs did during one profiled prefill. Reads vLLM's torch-profiler traces (rank*.pt.trace.json.gz).

    scripts/trace-summary.py DIR   > summary.md

Per rank: the span from the first to the last kernel, the busy time (union of kernel intervals, so overlapping streams
count once), the time inside NCCL kernels (an all-reduce kernel also counts the time it waits for the other rank), the
count of NCCL kernels, and the top kernels by total time. Cat "kernel" (and "gpu_memcpy" / "gpu_memset") events only.
"""
import collections
import gzip
import json
import re
import sys
from pathlib import Path


def union(iv):
    iv.sort()
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return tot + (cur_e - cur_s if cur_e is not None else 0)


def short(n):
    n = re.sub(r"<.*", "", n)
    return n[:70]


def main(d):
    print("| rank | span ms | busy ms | NCCL kernels | NCCL ms | NCCL / span | NCCL mean µs | max µs |")
    print("|---|---|---|---|---|---|---|---|")
    tops = {}
    for p in sorted(Path(d).glob("rank*.pt.trace.json.gz")):
        rank = p.name.split(".")[0]
        ev = [e for e in json.load(gzip.open(p))["traceEvents"]
              if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("dur", 0) > 0]
        t0 = min(e["ts"] for e in ev)
        t1 = max(e["ts"] + e["dur"] for e in ev)
        nccl = [e for e in ev if "nccl" in e["name"].lower()]
        nd = sum(e["dur"] for e in nccl)
        busy = union([(e["ts"], e["ts"] + e["dur"]) for e in ev])
        print(f"| {rank} | {(t1 - t0) / 1e3:.1f} | {busy / 1e3:.1f} | {len(nccl)} | {nd / 1e3:.1f} | "
              f"{100 * nd / (t1 - t0):.1f} % | {nd / max(1, len(nccl)):.0f} | {max((e['dur'] for e in nccl), default=0):.0f} |")
        agg = collections.defaultdict(lambda: [0, 0.0])
        for e in ev:
            a = agg[short(e["name"])]
            a[0] += 1
            a[1] += e["dur"]
        tops[rank] = (sorted(agg.items(), key=lambda kv: -kv[1][1])[:14], t1 - t0)
    for rank, (top, span) in tops.items():
        print(f"\n**{rank}: top kernels by total time**\n\n| kernel | count | ms | share of span |\n|---|---|---|---|")
        for n, (c, t) in top:
            print(f"| `{n}` | {c} | {t / 1e3:.1f} | {100 * t / span:.1f} % |")


main(sys.argv[1])
