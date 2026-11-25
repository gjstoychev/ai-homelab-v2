#!/usr/bin/env python3
"""Compare scripts/quality-check.py runs: one row per prompt, one column per run label.

    scripts/quality-compare.py RUNDIR/results.jsonl [--ref LABEL] > quality.md

A cell is "=" when every repetition gave the reference's text (the reference's first repetition), "≠" when it differs,
followed by "✓" or "✗" from the prompt's check; "~" marks a label whose repetitions disagree with each other.
Below the table, the first point where each differing text leaves the reference, with some context.
"""
import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results")
    ap.add_argument("--ref", default=None, help="the reference label (default: the first in the file)")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.results)]
    labels = list(dict.fromkeys(r["label"] for r in recs))
    prompts = list(dict.fromkeys(r["prompt_id"] for r in recs))
    ref = a.ref or labels[0]
    by = defaultdict(list)
    for r in recs:
        by[(r["label"], r["prompt_id"])].append(r)
    reftext = {p: by[(ref, p)][0]["text"] for p in prompts}

    print("| prompt | " + " | ".join(labels) + " |")
    print("|---|" + "---|" * len(labels))
    diffs = []
    for p in prompts:
        cells = []
        for l in labels:
            rs = by[(l, p)]
            texts = {r["text"] for r in rs}
            same = all(t == reftext[p] for t in texts)
            cell = ("=" if same else "≠") + (" ✓" if all(r["ok"] for r in rs) else " ✗") + (" ~" if len(texts) > 1 else "")
            cells.append(cell)
            for t in sorted(texts):
                if t != reftext[p]:
                    diffs.append((p, l, t))
        print(f"| {p} | " + " | ".join(cells) + " |")
    print(f"\nReference: `{ref}`. = same text as the reference, ≠ different, ✓/✗ the prompt's check, ~ repetitions differ.\n")
    seen = set()
    for p, l, t in diffs:
        if (p, t) in seen:
            continue
        seen.add((p, t))
        r = reftext[p]
        i = next((k for k in range(min(len(r), len(t))) if r[k] != t[k]), min(len(r), len(t)))
        lo = max(0, i - 60)
        same_as = sorted({l2 for (p2, l2, t2) in diffs if p2 == p and t2 == t})
        print(f"- **{p}**, {', '.join(f'`{x}`' for x in same_as)}: leaves the reference at character {i} of {len(r)}")
        print(f"  - reference: `…{r[lo:i + 60]!r}…`")
        print(f"  - this:      `…{t[lo:i + 60]!r}…`")


if __name__ == "__main__":
    main()
