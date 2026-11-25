#!/usr/bin/env python3
"""Ten fixed prompts with checkable answers, greedy, against any OpenAI-compatible server.

    scripts/quality-check.py --url http://127.0.0.1:8080 --label strata-det --out runs/x/quality.jsonl [--reps 2]

Not a benchmark of the model: the same prompts go to every engine and setting, so the question is whether an engine
or a setting changes the answers (text identical? still correct?). Every prompt REPS times, non-streamed
/v1/chat/completions, temperature 0, seed 42, thinking off, no prompt reuse (cache_prompt false for llama-server).
Each record keeps the full text, its SHA-256 and the result of a simple check (a number, a word, valid JSON ...).
The same arguments as scripts/oai-bench.py, so scripts/bench-run.sh can start the server for it (BENCH_CLIENT).
"""
import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request


def _json_person(t):
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return False
    try:
        o = json.loads(m.group(0))
    except ValueError:
        return False
    return (isinstance(o.get("name"), str) and isinstance(o.get("age"), int) and isinstance(o.get("tags"), list)
            and len(o["tags"]) == 3 and all(isinstance(x, str) for x in o["tags"]))


def _primes(t):
    return [int(x) for x in re.findall(r"\d+", t)][:10] == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


def _sorted_words(t):
    words = re.findall(r"[a-z]+", t.lower())
    want = ["apple", "banana", "cherry", "fig", "pear"]
    return [w for w in words if w in want] == want


def _palindrome_code(t):
    return "def is_palindrome" in t and ("lower()" in t or "casefold()" in t) and ("isalnum" in t or "re." in t)


PROMPTS = [
    ("arith", "What is 17 multiplied by 23? Answer with the number only.", 64,
     lambda t: re.search(r"\b391\b", t) is not None),
    ("capital", "What is the capital of Australia? Answer with one word.", 64,
     lambda t: "canberra" in t.lower()),
    ("primes", "List the first ten prime numbers, separated by commas. Nothing else.", 64, _primes),
    ("sort", "Sort these words alphabetically and output them separated by commas: pear, apple, fig, banana, cherry.", 64,
     _sorted_words),
    ("time", "A train leaves at 14:35 and arrives at 17:10 the same day. How long is the journey? Answer in hours and "
             "minutes, one short sentence.", 96,
     lambda t: re.search(r"\b2\s*(h|hours?)\b.*\b35\s*(m|min|minutes?)\b", t, re.I | re.S) is not None),
    ("author", "Who wrote the novel 'Pride and Prejudice'? Answer with the name only.", 64,
     lambda t: "austen" in t.lower()),
    ("german", "Translate into German, output only the translation: The weather is nice today.", 64,
     lambda t: "wetter" in t.lower() and "heute" in t.lower()),
    ("json", "Return a JSON object for a fictional person with the keys name (string), age (integer) and tags (an "
             "array of exactly three strings). Output only the JSON.", 128, _json_person),
    ("code", "Write a Python function is_palindrome(s) that returns True if s reads the same backwards, ignoring case "
             "and every character that is not a letter or digit. Output only the code.", 256, _palindrome_code),
    ("explain", "Explain in exactly three sentences why the sky is blue.", 160,
     lambda t: "scatter" in t.lower()),
]


def ask(url, prompt, max_tokens, timeout):
    body = {"model": "quality", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "seed": 42, "stream": False, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--load-s", type=float, default=None)
    ap.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args()
    with open(a.out, "a") as f:
        for rep in range(1, a.reps + 1):
            for pid, prompt, max_tokens, check in PROMPTS:
                r = ask(a.url, prompt, max_tokens, a.timeout)
                ch = r["choices"][0]
                text = ch["message"].get("content") or ""
                rec = {"label": a.label, "prompt_id": pid, "rep": rep, "load_s": a.load_s,
                       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "ok": bool(check(text)), "finish_reason": ch.get("finish_reason"),
                       "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
                       "sha256": hashlib.sha256(text.encode()).hexdigest()[:16], "text": text,
                       "timings": r.get("timings")}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"{a.label} rep {rep} {pid:8s} {'ok  ' if rec['ok'] else 'FAIL'} {rec['sha256']} "
                      f"{text[:70]!r}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
