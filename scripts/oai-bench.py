#!/usr/bin/env python3
"""Measure one configuration of any OpenAI-compatible server (llama.cpp, Strata, Basalt) the same way.

    scripts/oai-bench.py --url http://127.0.0.1:8080 --label 1gpu-nodraft --out runs/x/results.jsonl \
        [--reps 5] [--kinds prose,structured] [--max-tokens 256] [--load-s 20.3]

For each kind of prompt: one warm-up request (rep 0) and REPS measured ones, streamed from /v1/chat/completions.
Every request: the same long document (the lighthouse text of 2026-11-llamacpp-flags, about 4,300 tokens) and an
instruction that asks for far more than --max-tokens, temperature 0, seed 42, thinking off
(chat_template_kwargs.enable_thinking = false: both engines' templates read it), cache_prompt false (llama-server),
no ignore_eos (Strata has none).

Measured on the client, the same for every engine:
  ttft_s        request sent -> first content in the stream (prompt reading plus the first token)
  decode_tps    (completion_tokens - 1) / (last content - first content)
  total_s       request sent -> stream closed
The server's own numbers are kept as they come (usage, and llama.cpp-style "timings" where the server sends them:
prompt_per_second, predicted_per_second, draft_n, draft_n_accepted). A request whose finish_reason is not "length"
produced fewer tokens than asked; it stays in the file and is flagged, not dropped.
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.request

PARAGRAPH = (
    "The lighthouse keeper wrote down the weather every morning: wind "
    "direction, the height of the swell, the colour of the sky over the "
    "eastern headland and the number of ships that passed before noon. "
    "Over forty years the notebooks filled three shelves, and nobody but "
    "the keeper had ever read them. "
)
DOCUMENT = PARAGRAPH * 72 + "When the station was automated, the notebooks"

INSTRUCTIONS = {
    # prose: hard for a drafter to guess ahead
    "prose": "Continue this story in the same voice. Write at least 1,500 words: what happened to the notebooks, "
             "who read them first, and what they found in them. Plain prose, no headings, no lists.",
    # structured: repetitive keys and syntax, easy to guess ahead
    "structured": "Turn the keeper's notebooks into data. Output only a JSON array of 60 objects, one per notebook, "
                  "each with the keys \"year\", \"wind_direction\", \"swell_m\", \"sky_colour\", \"ships_before_noon\" "
                  "and \"note\", with plausible invented values. No text before or after the JSON.",
}


def stream(url, body, timeout):
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            yield time.perf_counter(), json.loads(data)


def one(url, kind, max_tokens, timeout):
    body = {
        "model": "bench",
        "messages": [{"role": "user", "content": DOCUMENT + "\n\n" + INSTRUCTIONS[kind]}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        # llama-server reuses a matching prompt prefix from the slot's KV cache unless told not to (--cache-ram 0 does
        # not stop it); Strata's equivalent is --prompt-cache 0 and --conversation-cache-mib 0 in its config
        "cache_prompt": False,
    }
    t0 = time.perf_counter()
    first = last = None
    text, reasoning = [], []
    finish, usage, timings = None, None, None
    chunks = 0
    for t, ev in stream(url, body, timeout):
        if ev.get("usage"):
            usage = ev["usage"]
        if ev.get("timings"):
            timings = ev["timings"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("reasoning_content"):
                reasoning.append(d["reasoning_content"])
            if d.get("content"):
                text.append(d["content"])
                chunks += 1
                first = first if first is not None else t
                last = t
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    t_end = time.perf_counter()
    out = "".join(text)
    n = (usage or {}).get("completion_tokens")
    rec = {
        "ttft_s": round(first - t0, 4) if first is not None else None,
        "total_s": round(t_end - t0, 4),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": n,
        "decode_tps": round((n - 1) / (last - first), 2) if n and n > 1 and last and last > first else None,
        "content_chunks": chunks,
        "finish_reason": finish,
        "short": finish != "length",
        "reasoning_chars": len("".join(reasoning)),
        "text_sha256": hashlib.sha256(out.encode()).hexdigest()[:16],
        "text_head": out[:160],
        "usage": usage,
        "timings": timings,
    }
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--kinds", default="prose,structured")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--load-s", type=float, default=None)
    ap.add_argument("--timeout", type=float, default=1800)
    a = ap.parse_args()
    with open(a.out, "a") as f:
        for kind in a.kinds.split(","):
            for rep in range(a.reps + 1):
                rec = {"label": a.label, "kind": kind, "rep": rep, "warmup": rep == 0, "load_s": a.load_s,
                       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       **one(a.url, kind, a.max_tokens, a.timeout)}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                s = rec.get("timings") or {}
                print(f"{a.label} {kind:10s} rep {rep}{' (warm-up)' if rep == 0 else '          '} "
                      f"prompt {rec['prompt_tokens']} tok, ttft {rec['ttft_s']} s, "
                      f"{rec['completion_tokens']} tok at {rec['decode_tps']} t/s"
                      f" | server: prompt {s.get('prompt_per_second', '-')} t/s, decode {s.get('predicted_per_second', '-')} t/s,"
                      f" drafts {s.get('draft_n_accepted', '-')}/{s.get('draft_n', '-')}"
                      f"{'' if not rec['short'] else '  [finish: ' + str(rec['finish_reason']) + ']'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
