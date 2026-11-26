#!/usr/bin/env python3
"""Summarize step 5 (run-sweep.sh) as Markdown tables, in the source's terms.

    experiments/2026-11-vllm-27b-tp2/scripts/sweep-summary.py runs/5-sweep > runs/5-sweep/summary.md

Measured requests only (warm-ups left out); means over them. Per request: steps/s = decode t/s / acceptance length
(one target-model step per draft; without MTP one step per token), acceptance length = 1 + accepted / drafts.
KV pool from each label's server log ("GPU KV cache size").
Table 1, from results.jsonl: per k, tok/s and acceptance by kind, steps/s over all kinds. A request that ended before
its token limit (finish_reason other than "length": a short answer, mostly JSON, so a high acceptance and speed) is
left out of the means and counted in the last column; its numbers stay in results.jsonl.
Table 2, from depth.jsonl: per k and depth, tok/s, steps/s and acceptance for the prose and the JSON task.
"""
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def mean(xs, nd=2):
    xs = [x for x in xs if x is not None]
    return f"{statistics.mean(xs):.{nd}f}" if xs else "–"


def acc(rs):
    sp = [r["spec"] for r in rs if r.get("spec")]
    drafts = sum(s["drafts"] for s in sp)
    return 1 + sum(s["accepted_tokens"] for s in sp) / drafts if drafts else None


def steps(r):
    s = r.get("spec")
    if not r.get("decode_tps"):
        return None
    return r["decode_tps"] / s["acceptance_length"] if s and s.get("acceptance_length") else r["decode_tps"]


def kv_pool(d, label):
    p = d / f"{label}.out"
    m = re.findall(r"GPU KV cache size: ([\d,]+) tokens", p.read_text(errors="replace")) if p.exists() else []
    return m[-1] if m else "–"


def k_of(label):
    return int(label.lstrip("k"))


def main(d):
    d = Path(d)
    rows, cut = defaultdict(list), defaultdict(list)
    for line in open(d / "results.jsonl"):
        r = json.loads(line)
        if not r.get("warmup"):
            (rows if not r["short"] else cut)[(r["label"], r["kind"])].append(r)
    labels = sorted({l for l, _ in rows}, key=k_of)
    kinds = ["prose", "structured", "mixed"]
    print("| k | steps/s | prose tok/s | structured tok/s | mixed tok/s | acceptance, prose / structured / mixed "
          "| KV pool | short |")
    print("|--:|--:|--:|--:|--:|---|--:|--:|")
    for l in labels:
        allr = [r for kd in kinds for r in rows[(l, kd)]]
        a = [acc(rows[(l, kd)]) for kd in kinds]
        print(f"| {k_of(l)} | {mean([steps(r) for r in allr])} | "
              + " | ".join(mean([r["decode_tps"] for r in rows[(l, kd)]]) for kd in kinds)
              + " | " + (" / ".join(f"{x:.2f}" for x in a) if all(a) else "—")
              + f" | {kv_pool(d, l)} | {sum(len(cut[(l, kd)]) for kd in kinds)} |")
    print()
    print("Per-position acceptance (share of drafts whose n-th token was accepted):")
    print()
    print("| k | prose | structured | mixed |")
    print("|--:|---|---|---|")
    for l in labels:
        cells = []
        for kd in kinds:
            sp = [r["spec"] for r in rows[(l, kd)] if r.get("spec")]
            drafts = sum(s["drafts"] for s in sp)
            pos = defaultdict(int)
            for s in sp:
                for p, v in (s.get("accepted_per_position") or {}).items():
                    pos[int(p)] += v
            cells.append(" ".join(f"{pos[p] / drafts:.2f}" for p in sorted(pos)) if drafts else "—")
        print(f"| {k_of(l)} | " + " | ".join(cells) + " |")
    dp = d / "depth.jsonl"
    if not dp.exists():
        return
    depth = defaultdict(list)
    for line in open(dp):
        r = json.loads(line)
        depth[(r["label"], r["target"], r["kind"])].append(r)
    print()
    print("| k | context tokens | prose tok/s | structured tok/s | steps/s | acceptance, prose / structured "
          "| TTFT s | short |")
    print("|--:|--:|--:|--:|--:|---|--:|--:|")
    for (l, t) in sorted({(l, t) for l, t, _ in depth}, key=lambda x: (k_of(x[0]), x[1])):
        p, s = depth[(l, t, "prose")], depth[(l, t, "structured")]
        a = [acc(p), acc(s)]
        short = sum(r["finish_reason"] != "length" for r in p + s)
        print(f"| {k_of(l)} | {mean([r['prompt_tokens'] for r in p + s], 0)} | {mean([r['decode_tps'] for r in p])}"
              f" | {mean([r['decode_tps'] for r in s])} | {mean([steps(r) for r in p + s])}"
              f" | {' / '.join(f'{x:.2f}' for x in a) if all(a) else '—'} | {mean([r['ttft_s'] for r in p + s], 1)}"
              f" | {short}/{len(p + s)} |")


if __name__ == "__main__":
    main(sys.argv[1])
