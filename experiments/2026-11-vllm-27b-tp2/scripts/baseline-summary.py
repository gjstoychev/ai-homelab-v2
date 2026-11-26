#!/usr/bin/env python3
"""Summarize a step-4-style run directory (baseline-client.py) as Markdown tables.

    experiments/2026-11-vllm-27b-tp2/scripts/baseline-summary.py runs/4-baseline > runs/4-baseline/summary.md

Measured requests only (rep 0, the warm-up, left out); mean ± standard deviation, min–max where useful.
Decode and acceptance table from results.jsonl: one row per label and kind; "acc. len" is 1 + accepted / drafts over
the measured requests together, "per position" the share of drafts whose k-th token was accepted.
TTFT table from long.jsonl: one row per label and target (no decode column: the needle answer is about 8 tokens).
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def ms(xs, nd=2):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "–"
    m = statistics.mean(xs)
    return f"{m:.{nd}f} ± {statistics.stdev(xs):.{nd}f}" if len(xs) > 1 else f"{m:.{nd}f}"


def rng(xs, nd=1):
    xs = [x for x in xs if x is not None]
    return f"{min(xs):.{nd}f}–{max(xs):.{nd}f}" if xs else "–"


def main(d):
    d = Path(d)
    rows = defaultdict(list)
    for line in open(d / "results.jsonl"):
        r = json.loads(line)
        if not r.get("warmup"):
            rows[(r["label"], r["kind"])].append(r)
    print("| label | kind | n | thinking | tokens (reasoning) | TTFT s | decode t/s | range | acc. len | per position |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (label, kind), rs in rows.items():
        sp = [r["spec"] for r in rs if r.get("spec")]
        drafts = sum(s["drafts"] for s in sp)
        acc = sum(s["accepted_tokens"] for s in sp)
        pos = defaultdict(int)
        for s in sp:
            for p, v in (s.get("accepted_per_position") or {}).items():
                pos[int(p)] += v
        per = " / ".join(f"{pos[p] / drafts:.2f}" for p in sorted(pos)) if drafts else "–"
        toks = statistics.mean(r["completion_tokens"] for r in rs)
        reas = [r.get("reasoning_tokens") for r in rs if r.get("reasoning_tokens") is not None]
        tok = f"{toks:.0f}" + (f" ({statistics.mean(reas):.0f})" if reas else "")
        dec = [r["decode_tps"] for r in rs]
        print(f"| {label} | {kind} | {len(rs)} | {'on' if rs[0].get('thinking') else 'off'} | {tok}"
              f" | {ms([r['ttft_s'] for r in rs], 3)} | {ms(dec)} | {rng(dec)}"
              f" | {f'{1 + acc / drafts:.2f}' if drafts else '–'} | {per} |")
    lp = d / "long.jsonl"
    if lp.exists():
        long = defaultdict(list)
        for line in open(lp):
            r = json.loads(line)
            if "error" not in r:
                long[(r["label"], r["target"])].append(r)
        print()
        print("| label | prompt tokens | n | TTFT s | prompt t/s (tokens / TTFT) | needle |")
        print("|---|---|---|---|---|---|")
        for (label, _), rs in long.items():
            pt = statistics.mean(r["prompt_tokens"] for r in rs)
            print(f"| {label} | {pt:.0f} | {len(rs)} | {ms([r['ttft_s'] for r in rs], 3)}"
                  f" | {ms([r['prompt_tokens'] / r['ttft_s'] for r in rs], 0)}"
                  f" | {sum(r['needle_found'] for r in rs)}/{len(rs)} |")


if __name__ == "__main__":
    main(sys.argv[1])
