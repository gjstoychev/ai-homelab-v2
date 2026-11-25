#!/usr/bin/env python3
"""Summarize scripts/bench-run.sh runs as a Markdown table: one row per run label and prompt kind.

    scripts/bench-summary.py RUNDIR [RUNDIR ...] > summary.md

Measured requests only (the warm-up, rep 0, is left out); mean ± standard deviation. Columns:
  load s        server start -> ready (bench-run.sh)
  TTFT s        client: request -> first content
  prompt t/s    server timings prompt_per_second (llama.cpp-style "timings"; Strata sends the same field)
  decode t/s    client: (completion tokens - 1) / (last content - first content)
  drafts        accepted / proposed tokens, summed over the measured requests (server timings)
  VRAM MiB      peak per GPU over the whole run (.gpu.csv)
  RSS GiB       peak resident memory of the server's process group (.ram.csv)
  short         measured requests that ended before max_tokens (finish_reason other than "length")
"""
import csv
import json
import statistics
import sys
from pathlib import Path


def ms(xs, nd=1):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "–"
    m = statistics.mean(xs)
    return f"{m:.{nd}f} ± {statistics.stdev(xs):.{nd}f}" if len(xs) > 1 else f"{m:.{nd}f}"


def peaks(run_dir, label):
    vram, rss = {}, None
    g = run_dir / f"{label}.gpu.csv"
    if g.exists():
        with open(g) as f:
            for row in csv.reader(f):
                if not row or row[0].startswith("timestamp"):
                    continue
                try:
                    i, mem = row[1].strip(), int(row[5].split()[0])
                except (IndexError, ValueError):
                    continue
                vram[i] = max(vram.get(i, 0), mem)
    r = run_dir / f"{label}.ram.csv"
    if r.exists():
        with open(r) as f:
            vals = [int(row["server_rss_mib"]) for row in csv.DictReader(f) if row.get("server_rss_mib", "").isdigit()]
            rss = max(vals) if vals else None
    return vram, rss


def main():
    rows = []
    for d in map(Path, sys.argv[1:]):
        res = d / "results.jsonl"
        if not res.exists():
            continue
        recs = [json.loads(l) for l in open(res)]
        for label in dict.fromkeys(r["label"] for r in recs):
            vram, rss = peaks(d, label)
            for kind in dict.fromkeys(r["kind"] for r in recs if r["label"] == label):
                m = [r for r in recs if r["label"] == label and r["kind"] == kind and not r["warmup"]]
                t = [r.get("timings") or {} for r in m]
                dn = sum(x.get("draft_n") or 0 for x in t)
                da = sum(x.get("draft_n_accepted") or 0 for x in t)
                rows.append([
                    f"{d.name}/{label}", kind, str(len(m)),
                    f"{m[0]['load_s']:.1f}" if m and m[0].get("load_s") is not None else "–",
                    ms([r["ttft_s"] for r in m], 2),
                    ms([x.get("prompt_per_second") for x in t]),
                    ms([r["decode_tps"] for r in m]),
                    f"{da}/{dn}" + (f" ({100 * da / dn:.0f} %)" if dn else ""),
                    " / ".join(f"{vram[k]}" for k in sorted(vram)) or "–",
                    f"{rss / 1024:.1f}" if rss else "–",
                    str(sum(1 for r in m if r["short"])),
                ])
    head = ["run", "kind", "n", "load s", "TTFT s", "prompt t/s", "decode t/s", "drafts", "VRAM MiB", "RSS GiB", "short"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        print("| " + " | ".join(r) + " |")


if __name__ == "__main__":
    main()
