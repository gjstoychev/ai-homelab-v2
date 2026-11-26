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

Options for servers that reuse prompts on their own, and for reproducing other measurements:
  --unique-prefix  every request starts with its own random line, so no request reuses another's prompt from a
                   prefix cache (vLLM's --enable-prefix-caching ignores cache_prompt)
  --thinking       thinking on (enable_thinking = true); TTFT and decode then count the first and last output token
                   of either kind, reasoning included (reasoning_tokens from usage where the server gives it)
  kind "mixed"     prose, then a JSON summary of it, in one answer
If the server has a Prometheus /metrics endpoint with speculative-decoding counters (vLLM: vllm:spec_decode_*), each
request records their change in "spec" (drafts, draft tokens, accepted tokens, accepted per position).
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
import uuid

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
    # short: no document, a few dozen prompt tokens, a long answer: decode alone (for --concurrency)
    "short": "Write a long story, at least 1,500 words, about a lighthouse keeper who writes down the weather every "
             "morning for forty years. Plain prose, no headings, no lists.",
    # structured: repetitive keys and syntax, easy to guess ahead
    "structured": "Turn the keeper's notebooks into data. Output only a JSON array of 60 objects, one per notebook, "
                  "each with the keys \"year\", \"wind_direction\", \"swell_m\", \"sky_colour\", \"ships_before_noon\" "
                  "and \"note\", with plausible invented values. No text before or after the JSON.",
    # mixed: prose first, then structured output
    "mixed": "Write about 400 words of plain prose on what the notebooks show about the weather over the forty "
             "years. Then output a JSON object summarizing it, with the keys \"decades\" (an array of objects with "
             "\"decade\", \"typical_wind\", \"stormy_days\" and \"remark\"), \"trend\" and \"most_ships_year\".",
}


def spec_counters(url):
    """Speculative-decoding counters from a Prometheus /metrics endpoint, as {name[{labels}]: value}; {} if none."""
    try:
        with urllib.request.urlopen(url + "/metrics", timeout=5) as r:
            body = r.read().decode("utf-8", "replace")
    except Exception:
        return {}
    out = {}
    for line in body.splitlines():
        if line.startswith("#") or "spec_decode" not in line or "_total" not in line:
            continue
        key, _, val = line.rpartition(" ")
        key = key.replace('model_name="bench",', "").replace(',model_name="bench"', "").replace('engine="0",', "")
        try:
            out[key] = out.get(key, 0) + float(val)
        except ValueError:
            pass
    return out


def spec_delta(before, after):
    d = {k: round(after[k] - before.get(k, 0)) for k in after if after[k] - before.get(k, 0)}
    if not d:
        return None
    get = lambda part: sum(v for k, v in d.items() if part in k and "per_pos" not in k)
    drafts, drafted, accepted = get("num_drafts"), get("num_draft_tokens"), get("num_accepted_tokens_total")
    per_pos = {k.split('position="')[1].split('"')[0]: v for k, v in d.items() if "per_pos" in k and 'position="' in k}
    return {"drafts": drafts, "draft_tokens": drafted, "accepted_tokens": accepted,
            "acceptance_length": round(1 + accepted / drafts, 3) if drafts else None,
            "accepted_per_position": {p: per_pos[p] for p in sorted(per_pos, key=int)} or None}


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


def one(url, kind, max_tokens, timeout, tag="", t_ref=None, thinking=False, unique=False, spec=False):
    if unique:
        tag = f"Request {uuid.uuid4().hex[:12]}.\n\n" + tag
    spec0 = spec_counters(url) if spec else None
    body = {
        "model": "bench",
        "messages": [{"role": "user", "content": tag + ("" if kind == "short" else DOCUMENT + "\n\n") + INSTRUCTIONS[kind]}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking},
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
            r = d.get("reasoning_content") or d.get("reasoning")
            if r:
                reasoning.append(r)
                if thinking:
                    first = first if first is not None else t
                    last = t
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
        "reasoning_tokens": ((usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "thinking": thinking,
        "unique_prefix": unique,
        "spec": spec_delta(spec0, spec_counters(url)) if spec0 else None,
        "text_sha256": hashlib.sha256(out.encode()).hexdigest()[:16],
        "text_head": out[:160],
        "reasoning_head": "".join(reasoning)[:160] if reasoning else None,
        "usage": usage,
        "timings": timings,
    }
    if t_ref is not None:                       # concurrent requests: start and end relative to the group's start
        rec["t_start"], rec["t_end"] = round(t0 - t_ref, 4), round(t_end - t_ref, 4)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--kinds", default=os.environ.get("BENCH_KINDS", "prose,structured"))
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("BENCH_MAX_TOKENS", "256")))
    ap.add_argument("--load-s", type=float, default=None)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--concurrency", default=os.environ.get("BENCH_CONCURRENCY", "1"),
                    help="N requests at once per repetition, or a list (1,2,4) run one after the other on the same "
                         "server, each as label-cN; each prompt starts with its own tag, so no two share a prefix")
    ap.add_argument("--thinking", action="store_true", default=os.environ.get("BENCH_THINKING") == "1")
    ap.add_argument("--unique-prefix", action="store_true", default=os.environ.get("BENCH_UNIQUE_PREFIX") == "1")
    a = ap.parse_args()
    a.spec = bool(spec_counters(a.url))
    levels = [int(x) for x in str(a.concurrency).split(",")]
    if levels != [1]:
        base = a.label
        for n in levels:
            a.concurrency, a.label = n, f"{base}-c{n}"
            concurrent_runs(a)
        return 0
    with open(a.out, "a") as f:
        for kind in a.kinds.split(","):
            for rep in range(a.reps + 1):
                rec = {"label": a.label, "kind": kind, "rep": rep, "warmup": rep == 0, "load_s": a.load_s,
                       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       **one(a.url, kind, a.max_tokens, a.timeout, thinking=a.thinking, unique=a.unique_prefix,
                             spec=a.spec)}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                s = rec.get("timings") or {}
                print(f"{a.label} {kind:10s} rep {rep}{' (warm-up)' if rep == 0 else '          '} "
                      f"prompt {rec['prompt_tokens']} tok, ttft {rec['ttft_s']} s, "
                      f"{rec['completion_tokens']} tok at {rec['decode_tps']} t/s"
                      f" | server: prompt {s.get('prompt_per_second', '-')} t/s, decode {s.get('predicted_per_second', '-')} t/s,"
                      f" drafts {s.get('draft_n_accepted', '-')}/{s.get('draft_n', '-')}"
                      f"{' | reasoning ' + str(rec['reasoning_tokens']) + ' tok' if a.thinking else ''}"
                      f"{' | MTP acceptance length ' + str(rec['spec']['acceptance_length']) if rec['spec'] else ''}"
                      f"{'' if not rec['short'] else '  [finish: ' + str(rec['finish_reason']) + ']'}", flush=True)


def concurrent_runs(a):
    from concurrent.futures import ThreadPoolExecutor
    with open(a.out, "a") as f, ThreadPoolExecutor(a.concurrency) as pool:
        for kind in a.kinds.split(","):
            for rep in range(a.reps + 1):
                t_ref = time.perf_counter()
                futs = [pool.submit(one, a.url, kind, a.max_tokens, a.timeout,
                                    f"Request {rep}.{i} of a batch test.\n\n", t_ref, a.thinking, a.unique_prefix)
                        for i in range(a.concurrency)]
                recs = [fu.result() for fu in futs]
                wall = max(r["t_end"] for r in recs)
                toks = sum(r["completion_tokens"] or 0 for r in recs)
                for i, r in enumerate(recs):
                    f.write(json.dumps({"label": a.label, "kind": kind, "rep": rep, "warmup": rep == 0,
                                        "load_s": a.load_s, "concurrency": a.concurrency, "stream": i,
                                        "group_wall_s": round(wall, 4), "group_tokens": toks,
                                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **r}) + "\n")
                f.flush()
                print(f"{a.label} {kind:10s} rep {rep}{' (warm-up)' if rep == 0 else '          '} x{a.concurrency}: "
                      f"{toks} tokens in {wall:.2f} s = {toks / wall:.1f} t/s total; per stream ttft "
                      f"{[r['ttft_s'] for r in recs]}, decode {[r['decode_tps'] for r in recs]}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
