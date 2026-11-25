#!/usr/bin/env python3
"""Long prompts and a four-turn job against an OpenAI-compatible server (llama.cpp, Strata, Basalt).

    scripts/long-bench.py --url URL --label L --out results.jsonl --mode ttft [--targets 32768,131072] [--reps 3]
    scripts/long-bench.py --url URL --label L --out results.jsonl --mode job  [--job-tokens 32768] [--reps 3]

The document: every *.md file of --corpus-dir (default: Strata's docs at the pinned tag) in sorted order, each under a
"# FILE: name" line, cut to the size that gives the target prompt length, with one invented sentence (the needle) in
the middle: "The passphrase for the archive is <word>-<number>." The last question asks for it, so a long prompt that
is read badly shows as a wrong answer, not only as a time.

ttft: for each target, one calibration request on a short slice gives the characters per token; then REPS requests
      of the same prompt (streamed, temperature 0, thinking off, max_tokens 64, no prompt reuse). Measured: TTFT,
      prompt t/s (server timings), decode t/s after the prompt, and whether the needle came back.
job:  REPS runs of a four-turn conversation over a document of --job-tokens: summary, a list from the text, a script,
      the needle. Prompt reuse ON (the server's own caches, as an agent would have them); a random session tag at the
      start of every run, so a run does not reuse an earlier run's prompt. Measured: the wall time of the whole job,
      and per turn TTFT, prompt tokens and how many of them the server reused.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

NEEDLE_WORDS = ["cormorant", "tamarisk", "basilisk", "meridian", "quillwort", "obsidian"]
QUESTION = "What is the passphrase for the archive mentioned in the document? Answer with the passphrase only."
JOB_TURNS = [
    ("summary", "Summarize the document above in five bullet points.", 400),
    ("list", "Which environment variables starting with STRATA_ does the document mention? List up to ten, one line "
             "each, with what it does.", 400),
    ("script", "Write a short bash script that starts the server on two GPUs the way the document describes. Only "
               "the script.", 400),
    ("needle", QUESTION, 32),
]


def corpus(d):
    parts = []
    for p in sorted(Path(d).glob("*.md")):
        parts.append(f"# FILE: {p.name}\n\n" + p.read_text(encoding="utf-8", errors="replace"))
    return "\n\n".join(parts)


def document(text, n_chars, needle):
    body = text[:n_chars]
    mid = body.rfind("\n\n", 0, len(body) // 2)
    mid = mid if mid > 0 else len(body) // 2
    return body[:mid] + f"\n\n{needle}\n\n" + body[mid:]


def post(url, body, timeout, stream):
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=timeout)
    if not stream:
        with r:
            return json.load(r)
    return r


def chat(url, messages, max_tokens, timeout, reuse):
    """One streamed request; returns the text and the measurements."""
    body = {"model": "long", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "seed": 42,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": reuse}
    t0 = time.perf_counter()
    first = last = None
    text, usage, timings, finish = [], None, None, None
    with post(url, body, timeout, True) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            t = time.perf_counter()
            usage = ev.get("usage") or usage
            timings = ev.get("timings") or timings
            for ch in ev.get("choices") or []:
                c = (ch.get("delta") or {}).get("content")
                if c:
                    text.append(c)
                    first = first if first is not None else t
                    last = t
                finish = ch.get("finish_reason") or finish
    n = (usage or {}).get("completion_tokens")
    return "".join(text), {
        "ttft_s": round(first - t0, 3) if first else None,
        "total_s": round(time.perf_counter() - t0, 3),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
        "completion_tokens": n,
        "decode_tps": round((n - 1) / (last - first), 2) if n and n > 1 and last and last > first else None,
        "finish_reason": finish,
        "timings": timings,
    }


def calibrate(url, text, timeout):
    """Characters per token of `text`: 16 slices of 1,000 characters spread evenly over it, in one request. The
    corpus is not uniform (prose first, more code and tables later), so the beginning alone underestimates tokens."""
    step = max(1000, len(text) // 16)
    sample = "\n".join(text[i:i + 1000] for i in range(0, len(text) - 1000, step))[:16000 + 15]
    _, m = chat(url, [{"role": "user", "content": sample + "\n\nReply with OK."}], 1, timeout, False)
    return len(sample) / max(1, (m["prompt_tokens"] or 1) - 20)


def sized(url, text, target, timeout):
    """The number of characters of `text` for a prompt of about `target` tokens: calibrated on the prefix itself,
    twice (the second time on the prefix the first estimate gave), 5 % below the target."""
    n = len(text)
    for _ in range(2):
        cpt = calibrate(url, text[:n], timeout)
        n = min(len(text), int((target - 200) * cpt * 0.95))
    return n, cpt


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["ttft", "job"], default=os.environ.get("LONG_MODE", "ttft"))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--targets", default=os.environ.get("LONG_TARGETS", "32768,131072"))
    ap.add_argument("--job-tokens", type=int, default=int(os.environ.get("LONG_JOB_TOKENS", "32768")))
    ap.add_argument("--corpus-dir", default=os.environ.get("LONG_CORPUS", str(Path.home() / "strata/v0.2.21/src/docs")))
    ap.add_argument("--load-s", type=float, default=None)
    ap.add_argument("--timeout", type=float, default=3600)
    a = ap.parse_args()

    text = corpus(a.corpus_dir)
    meta = {"label": a.label, "mode": a.mode, "load_s": a.load_s, "corpus_dir": a.corpus_dir,
            "corpus_chars": len(text), "corpus_sha256": hashlib.sha256(text.encode()).hexdigest()[:16]}
    rnd = random.Random(42)
    needle_word = f"{rnd.choice(NEEDLE_WORDS)}-{rnd.randint(1000, 9999)}"
    needle = f"The passphrase for the archive is {needle_word}."
    print(f"{a.label}: corpus {len(text)} chars, needle {needle_word}", flush=True)

    with open(a.out, "a") as f:
        if a.mode == "ttft":
            for target in (int(x) for x in a.targets.split(",")):
                n_chars, cpt = sized(a.url, text, target, a.timeout)
                doc = document(text, n_chars, needle)
                print(f"{a.label} {target}: {n_chars} chars at {cpt:.2f} chars/token", flush=True)
                msgs = [{"role": "user", "content": doc + "\n\n" + QUESTION}]
                for rep in range(1, a.reps + 1):
                    try:
                        out, m = chat(a.url, msgs, 64, a.timeout, False)
                    except urllib.error.HTTPError as e:       # e.g. 400: the prompt does not fit the context
                        err = e.read().decode("utf-8", "replace")[:300]
                        f.write(json.dumps({**meta, "target": target, "rep": rep, "error": f"HTTP {e.code}: {err}"}) + "\n")
                        print(f"{a.label} {target} rep {rep}: HTTP {e.code}: {err}", flush=True)
                        break
                    rec = {**meta, "target": target, "rep": rep, "chars_per_token": round(cpt, 3),
                           "needle": needle_word, "needle_found": needle_word in out, "text": out[:200], **m,
                           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    s = m["timings"] or {}
                    print(f"{a.label} {target} rep {rep}: prompt {m['prompt_tokens']} tok, ttft {m['ttft_s']} s "
                          f"(server {s.get('prompt_per_second', '-')} t/s), decode {m['decode_tps']} t/s, "
                          f"needle {'found' if rec['needle_found'] else 'MISSING'}: {out[:40]!r}", flush=True)
        else:
            n_chars, cpt = sized(a.url, text, a.job_tokens, a.timeout)
            doc = document(text, n_chars, needle)
            print(f"{a.label} job: {n_chars} chars at {cpt:.2f} chars/token", flush=True)
            for rep in range(1, a.reps + 1):
                tag = uuid.uuid4().hex[:12]
                msgs = [{"role": "user", "content": f"Session {tag}.\n\n" + doc + "\n\n" + JOB_TURNS[0][1]}]
                turns, t0 = [], time.perf_counter()
                for i, (name, q, max_tokens) in enumerate(JOB_TURNS):
                    if i:
                        msgs.append({"role": "user", "content": q})
                    out, m = chat(a.url, msgs, max_tokens, a.timeout, True)
                    msgs.append({"role": "assistant", "content": out})
                    turns.append({"turn": name, **{k: m[k] for k in ("ttft_s", "total_s", "prompt_tokens",
                                                                     "cached_tokens", "completion_tokens",
                                                                     "decode_tps", "finish_reason")},
                                  "reused": (m["timings"] or {}).get("cache_n"), "text": out[:300]})
                wall = time.perf_counter() - t0
                rec = {**meta, "rep": rep, "session": tag, "job_tokens": a.job_tokens, "wall_s": round(wall, 2),
                       "needle": needle_word, "needle_found": needle_word in turns[-1]["text"], "turns": turns,
                       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"{a.label} job rep {rep}: {wall:.1f} s; " + "; ".join(
                    f"{t['turn']} ttft {t['ttft_s']} s, {t['prompt_tokens']} tok ({t['reused']} reused), "
                    f"{t['completion_tokens']} out" for t in turns)
                    + f"; needle {'found' if rec['needle_found'] else 'MISSING'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
