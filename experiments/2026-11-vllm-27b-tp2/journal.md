# Journal: 2026-11-vllm-27b-tp2

Every step in order: why, the exact command, expected vs actual. Mistakes and dead ends stay in. Plan, pinned versions and method: [`README.md`](README.md).

## 0. Plan (2026-11-26)

Written from the source repository (README with its compose file, `kv-quality-test.md`, `kvq/run-server.sh`), the checkpoint's Hugging Face file list and history, and the two vLLM images' configs (read with `docker buildx imagetools inspect`, nothing pulled):

- `vllm/vllm-openai:v0.30.0`: `sha256:8a69ffad…200a4b90`, created 2026-09-21, `CUDA_VERSION=13.0.2`, `NV_CUDA_CUDART_VERSION=13.0.96-1`, `TORCH_CUDA_ARCH_LIST=7.5 8.0 8.6 8.9 9.0 10.0 12.0`.
- `vllm/vllm-openai:v0.32.0`: `sha256:befccf0e…7957c5b34`, created 2026-10-18, the same CUDA.
- `apt-cache policy`: `nvidia-driver-595-server-open` 595.91.07-0ubuntu0.24.04.1 (the source's driver), `nvidia-driver-595-open` 595.99.02, `cuda-toolkit-13-2` 13.2.2-1. Nothing installed.

The images need CUDA 13.0, which the installed driver (580.178.04) runs, so the reproduction starts on it (phase A) and the source's driver is a later step of its own (phase B).

## 0b. Disk (2026-11-26)

At the user's request: deleted from `/mnt/models` the two models of the finished `2026-11-llamacpp-flags` experiment (batiai IQ3_XXS `1d0ec19c`, DeepSeek-V4-Flash-0731 UD-IQ1_M `fbbb5b93`; both downloadable again at those revisions, `SHA256SUMS` were in their folders: batiai `4e461bef…`, `86755e0c…`; DeepSeek `b99fa246…`, `c8536187…`, `80a4eb4e…`). Then `scripts/move-model.sh` moved the GSQ-RCO GGUF (71 GB, checksums OK) and started on the Basalt GSQ-RCO pack (75 GB) to `/mnt/models`. The A400 SATA disk wrote at about 60 MB/s on these long writes (its cache full), not 250 MB/s as on the first move. Planned next: Docker cleanup (board-swap images and build cache; keep searxng, n8n and the vLLM images).

## 0c. Disk, finished (2026-11-26)

The Basalt GSQ-RCO pack move finished: the source is gone from `~/models` (the script deletes it only after the checksums pass), and `sha256sum -c SHA256SUMS` in `/mnt/models/qwen3.8-flash-next-basalt-gsq-rco-iq3xxs` gives OK again (75 GB). `/mnt/models` 146 of 220 GB used.

Docker cleanup: `docker rm` of the two hello-world containers; `docker rmi local/llama.cpp:b12464-nccl local/nvbandwidth:cuda13.0.0 ghcr.io/ggml-org/llama.cpp:full-cuda ollama/ollama:0.32.14 nvidia/cuda:13.0.0-base-ubuntu24.04 hello-world:latest`; `docker builder prune -a -f` (15.8 GB). Images went from 97.4 to 65.9 GB on disk, and the build cache to zero. Kept: searxng (running), n8n with its volume, alpine, the two vLLM images.

## 1. Images (2026-11-26)

`docker pull vllm/vllm-openai@sha256:…` for both digests, tagged `local/vllm-openai:v0.30.0` (30.7 GB on disk, 809 s) and `local/vllm-openai:v0.32.0` (33.1 GB, 312 s).

## 2. Checkpoint (2026-11-26)

`scripts/hf-fetch.sh gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 0cc27958cefbbe231782ec8511de8c4eb5233348 ~/models/qwen3.8-27b-nvfp4-0cc2795 <the 15 files of the source's wget loop>`: 652 s; 4 LFS files OK by SHA-256 (the three safetensors, `tokenizer.json`), 11 small git files OK by their git blob SHA-1. `hf-fetch.sh` learned the git-blob check for this (it handled LFS files only); `SHA256SUMS` has the SHA-256 of all 15.

## 3. Smoke test, first try (2026-11-26): not finished

New: `scripts/vllm-serve.sh configs/<name>.args` runs the container in the foreground and stops it (`docker stop`) on SIGTERM, so `scripts/bench-run.sh` can start and stop it like the other servers. `configs/fp8-mtp4.args`: the source's compose flags with fp8 KV and util 0.95 (the FP8 baseline), `--served-model-name bench`, no API key.

The container started and was loading (log in `runs/3-smoke/server.out`: fp8 KV, "Using KV cache scaling factor 1.0 for fp8_e4m3", LBNHC KV layout, no errors so far), but the waiting loop watched the PID of `setsid` started from an interactive shell, which forks and exits at once, so it reported "SERVER EXITED after 0 s" while the container kept running. The container was stopped by hand (`docker stop vllm-bench`) to end the session cleanly. **Next:** start it through `scripts/bench-run.sh` (a script, where `setsid` does not fork), e.g. `scripts/bench-run.sh runs/3-smoke fp8-mtp4 0 -- experiments/2026-11-vllm-27b-tp2/scripts/vllm-serve.sh experiments/2026-11-vllm-27b-tp2/configs/fp8-mtp4.args` (paths relative to where it runs), check `--default-chat-template-kwargs` is accepted by 0.30.0 (it is from the 0.32.0 compose), read the KV pool (source: 278,927 tokens), then the first requests. The client sends `cache_prompt` and `chat_template_kwargs`; vLLM sends no llama.cpp-style `timings`, so the prompt t/s and draft columns of `bench-summary.py` stay empty (TTFT and decode come from the client).

## 3. Smoke test (2026-11-26)

Kernel 7.0.0-44 since the reboot at 12:50 (installed by unattended-upgrades at 08:28; `docs/stack.md`); driver 580.178.04, both cards at 180 W.

```
scripts/bench-run.sh experiments/2026-11-vllm-27b-tp2/runs/3-smoke fp8-mtp4 1 -- \
  experiments/2026-11-vllm-27b-tp2/scripts/vllm-serve.sh experiments/2026-11-vllm-27b-tp2/configs/fp8-mtp4.args
```

Run from a script this time, so `setsid` does not fork and the waiter watches the right PID. Files: `runs/3-smoke/fp8-mtp4.*` (`.console.txt` is the script's output), requests in `results.jsonl`.

| | Expected (source) | Actual |
|---|---|---|
| Starts with the source's flags on 0.30.0 | yes | yes; `--default-chat-template-kwargs` (from the 0.32.0 compose) accepted |
| KV pool, fp8, MTP k=4, util 0.95 | 278,927 tokens | **273,333** (−2.0 %); "Available KV cache memory: 4.7 GiB" per card; vLLM notes util 0.95 is effectively 0.9245 once CUDA graph memory is profiled |
| Weights | — | 8.91 GiB per card, 7.1 s (the IPC weight cache is absent, so it falls back to disk) |
| Attention backend | — | FlashInfer |
| CUDA graphs | PIECEWISE | 7 sizes in 21 s, then 7 for the drafter in 1 s; engine init 72.9 s, of which compile 22.6 s |
| Ready | — | 119.1 s after start (page cache cold after the reboot) |
| Peak VRAM | — | 15,790 MiB on each card |
| MTP | acceptance > 0 | yes: prose window mean acceptance length 2.27 (per position 0.68 / 0.36 / 0.16 / 0.07), JSON window 4.02 (0.91 / 0.80 / 0.68 / 0.64) |

First numbers (shared client: thinking off, 256 tokens, about 4.3K-token prompts, one rep after a warm-up; not the reproduction mode yet):

| Prompt | TTFT | Decode |
|---|---|---|
| prose | 0.302 s | 62.95 t/s |
| structured (JSON) | 0.340 s | 114.41 t/s |

In range of the source's k=4 numbers (78.3 prose, 124.9 structured), which were measured with thinking on and 800 tokens; step 4 does both modes.

**A stop problem:** after the run the container was still up. `bench-run.sh` sent TERM to the process group at 12:57:04; vLLM began its shutdown ("send sigterm to process EngineCore") and hung there, still logging stats. `vllm-serve.sh` ran `docker stop -t 60`, but `bench-run.sh` sends KILL to the group after 30 s, which killed `docker stop` too. Stopped by hand (`docker stop -t 30 vllm-bench`, 19 s). Fix: `vllm-serve.sh` now uses `docker stop -t 15`, so Docker SIGKILLs a hung vLLM well inside the 30 s.

Check of the fix: `scripts/bench-run.sh experiments/2026-11-vllm-27b-tp2/runs/3-smoke stop-check 0 -- …` (start, stop, no requests). The shutdown hung at the same line again ("send sigterm to process EngineCore"); `docker stop -t 15` ended it, `bench-run.sh` returned after 97 s in total, no container left, both GPUs back at 2 MiB. Peak VRAM 15,480 MiB per card without requests. The hang is vLLM's own (both times); with the wrapper it now costs 15 s per stop.

**Next:** step 4, the phase A baseline (MTP off and k=4; shared clients and the reproduction mode; TTFT at about 0.5K, 4K and 32K).

## 4. Phase A baseline (2026-11-26)

**Client changes first.** The smoke test's TTFT (0.30 s for 4.3K tokens; the source: 0.99 s at 4.1K) was a cache hit: vLLM runs with `--enable-prefix-caching` and ignores `cache_prompt: false`, so every repeated prompt after the warm-up came from the prefix cache. Changes, all opt-in so earlier experiments' runs keep their meaning:

- `scripts/oai-bench.py`: `--unique-prefix` (a random first line per request), `--thinking` (thinking on; TTFT and decode count the first and last token of either kind, reasoning included, the source's way), a `mixed` kind (about 400 words of prose, then a JSON summary), and per request the change of vLLM's `vllm:spec_decode_*` counters from `/metrics` (drafts, drafted, accepted, accepted per position) when the server has them.
- `scripts/long-bench.py`: `--unique-prefix` for the ttft mode.
- `scripts/baseline-client.py` (this experiment): three passes on one server start, as `BENCH_CLIENT` of `bench-run.sh`: shared (prose and structured, thinking off, 256 tokens, 5 reps), reproduction (prose, structured and mixed, thinking on, 800 tokens, 3 reps), TTFT (`long-bench.py` targets 700 / 4,500 / 34,700, which give about the source's 523 / 4,123 / 32,812 tokens, 3 reps, into `long.jsonl`). A warm-up request (rep 0) per kind in the first two passes.
- `scripts/baseline-summary.py` writes `runs/4-baseline/summary.md`.
- `configs/fp8-nomtp.args`: `fp8-mtp4.args` without `--speculative-config`.

```
BENCH_CLIENT=experiments/2026-11-vllm-27b-tp2/scripts/baseline-client.py scripts/bench-run.sh \
  experiments/2026-11-vllm-27b-tp2/runs/4-baseline fp8-mtp4 5 -- \
  experiments/2026-11-vllm-27b-tp2/scripts/vllm-serve.sh experiments/2026-11-vllm-27b-tp2/configs/fp8-mtp4.args
# the same with fp8-nomtp
```

Check of the counters on the first request: 104 drafts + 153 accepted = 257 tokens for 256 generated, so the parsing is right.

**Results** (`runs/4-baseline/summary.md`; means of the measured reps):

| | Source | fp8, MTP off | fp8, MTP k=4 |
|---|---|---|---|
| KV pool, tokens | 351,618 / 278,927 | **348,478** (−0.9 %) | **273,333** (−2.0 %) |
| Ready after start | | 97.3 s | 78.6 s |
| Decode, shared, prose (thinking off, 256) | | 42.15 | 62.75 ± 3.68 (acc. len 2.26) |
| Decode, shared, structured (JSON) | | 42.12 | 117.02 ± 2.22 (4.22) |
| Decode, reproduction, prose (thinking on, 800) | 41.99 / **78.31** | **42.05** | **69.81 ± 6.01** (2.50), range 64.0–76.0 |
| Decode, reproduction, structured | 41.99 / **124.89** | 42.05 | **81.10 ± 1.35** (2.92) |
| Decode, reproduction, mixed | 41.99 / **81.61** | 42.05 | **73.97 ± 3.07** (2.66) |
| TTFT, ~0.5K (580 tokens; source 523) | 0.141 s | 0.187 s | 0.200 s |
| TTFT, ~4K (4,212; source 4,123) | 0.987 s | 1.314 s | 1.355 s |
| TTFT, ~32K (32,266; source 32,812) | 10.521 s | 12.417 s | 13.146 s |
| Needle at all three lengths | | 9/9 | 9/9 |
| Peak VRAM per card | | 15,668 MiB | 15,790 MiB |
| Peak power per card (limit 180 W) | | 144 / 136 W | 137 / 130 W |

What it shows:

- **Decode without MTP reproduces exactly:** 42.05–42.15 t/s against 41.99, the same for every kind of prompt (no drafter, so content does not matter). The KV pools are within 1–2 %.
- **With MTP, decode depends on the prompt, and the source's prompts are not published.** Thinking on, all 800 tokens of every reproduction request were reasoning (`reasoning_tokens` 800): the model was still thinking at the limit, for prose, structured and mixed alike. So the reproduction's "structured" measures reasoning about a JSON task, not JSON (81 t/s, acceptance length 2.9), while the shared client's thinking-off JSON gives 117 t/s at 4.2. The source's 124.9 t/s "structured" is in line with JSON output (ours: 117, −6 %), not with reasoning; its prose 78.3 sits at the top of our prose range (64–76, acceptance length 2.3–2.7). Decode tracks acceptance length closely; the engine side looks the same as the source's.
- **Prompt reading is 18–33 % slower than the source's:** 3,100–3,200 prompt t/s at 0.5–4K and 2,600 at 32K without MTP; MTP adds 3–6 % (the drafter also runs over the prompt). Not power (the cards peak at 130–144 W against 180 W). Both cards are on Gen5 x8 links, as on the source's board; with `NCCL_P2P_DISABLE=1` every all-reduce goes through host memory, and prompt reading is where TP=2 moves the most data. Candidates to separate later: the driver (phase B, the source's 595.91.07), the host path (CPU and memory: i7-14700 against the source's 9800X3D with DDR5), and the source's measurement (not published). Decode matches, so it is not the GPUs themselves.
- The first-request TTFT after a start (1.39 s warm-up) equals the steady one: no extra warm-up cost on the first prompt.

**Next:** step 5, the k sweep 0–7 (one restart per k) and decode at depth 0 / 32K / 131K for k=0 and k=4.

## 5. k sweep and decode at depth (2026-11-26)

New: `scripts/long-bench.py --mode depth` (a prose and a JSON task on a document of the target size, 512 tokens, thinking off, a random first line per request, the MTP counters per request); `scripts/sweep-client.py` (decode: prose, structured and mixed, thinking off, 640 tokens so the mixed answer reaches its JSON, 3 reps after a warm-up; with `SWEEP_DEPTH=1` also the depth pass at about 1K / 32K / 131K tokens, 2 reps per task); `scripts/run-sweep.sh` (one server start per k, depth for k=0 and 4); configs `fp8-mtp{1,2,3,5,6,7}.args` (only `num_speculative_tokens` differs from `fp8-mtp4.args`); `scripts/sweep-summary.py`. Requests that end before the token limit are left out of the means and counted.

```
experiments/2026-11-vllm-27b-tp2/scripts/run-sweep.sh        # from the repository root
```

(The sweep ran 13:26–14:17 UTC, eight server starts of 63–122 s each. Files: `runs/5-sweep/k<k>.*`, `results.jsonl`, `depth.jsonl`, `console.txt`, `summary.md`.)

**Results** (thinking off, 640 tokens, about 4.3K-token prompt, means of 3 reps; the source's in brackets, its sweep with thinking on and its own prompts):

| k | steps/s | prose t/s | structured t/s | mixed t/s | KV pool |
|--:|--:|--:|--:|--:|--:|
| 0 | 42.07 [41.92] | 42.07 [41.99] | 42.07 [41.99] | 42.07 [41.92] | 348,478 [351,618] |
| 1 | 35.64 [35.94] | 60.13 [63.91] | 69.55 [70.27] | 62.55 [63.41] | 286,695 [290,655] |
| 2 | 32.74 [33.26] | 67.92 [75.21] | 91.55 [95.40] | 72.05 [77.54] | 280,526 [285,321] |
| 3 | 30.12 [30.74] | 67.62 [78.10] | 107.20 [108.73] | 73.62 [83.01] | 279,567 [284,319] |
| 4 | 27.84 [28.57] | 63.89 [78.31] | 117.55 [124.89] | 73.04 [81.61] | 273,333 [278,927] |
| 5 | 25.89 [26.77] | 59.44 [72.66] | 123.02 [135.58] | 68.94 [79.03] | 268,479 [274,815] |
| 6 | 24.30 [25.19] | 56.21 [73.09] | 127.69 [135.36] | 62.77 [74.67] | 266,140 [272,533] |
| 7 | 22.81 [23.80] | 52.98 [70.84] | 133.89 [138.08] | 58.51 [76.50] | 262,950 [268,596] |

Acceptance length (prose / structured / mixed): k=4 2.29 / 4.22 / 2.62 [2.73 / 4.35 / –]; k=7 2.31 / 5.89 / 2.56 [2.96 / 5.76 / –]. One request (k=5, mixed, rep 3) ended after 284 tokens instead of 640 (finish "stop", a short answer, 100.6 t/s); it is left out of that cell's mean (`short` column of `summary.md`).

Decode at depth (the depth pass's own prompts, 512 tokens; the source's table in brackets, its depths 0 / 32,768 / 131,072; ours 950 / 32,834 / 126,603 tokens, since the sizing aims 5 % under the target):

| k | context | prose t/s | structured t/s | steps/s | acceptance prose / structured | TTFT |
|--:|--:|--:|--:|--:|---|--:|
| 0 | 1K | 42.25 [41.99] | 42.25 [42.00] | 42.25 | – | 0.3 s |
| 0 | 32K | 40.25 [39.98] | 40.26 [39.98] | 40.26 | – | 12.8 s |
| 0 | 127K | 35.08 [34.66] | 35.07 [34.66] | 35.08 [34.56] | – | 83.1 s |
| 4 | 1K | 77.98 [80.10] | 117.63 [123.33] | 27.91 | 2.78 / 4.23 | 0.3 s |
| 4 | 32K | 70.12 [69.60] | 112.94 [115.61] | 26.68 | 2.62 / 4.24 | 13.4 s |
| 4 | 127K | 68.30 [68.83] | 92.16 [92.41] | 23.73 [24.5] | 2.87 / 3.88 | 88.6 s |

What it shows:

- **The engine reproduces.** Steps/s at every k is within 1–4 % of the source's (the lower the k, the closer). Without MTP, 42.07 against 41.92 steps/s, and at depth 42.25 / 40.25 / 35.08 against 41.99 / 39.98 / 34.66 t/s. At k=4 the depth numbers match to 1–4 % on prose, structured and steps/s (68.30 against 68.83 prose and 92.16 against 92.41 JSON at 127K / 131K). The KV pool is 0.9–2.4 % smaller at every k.
- **The step-time law holds, with a slightly dearer draft position.** Linear fit over k=1..7 (all within 0.1 steps/s): step = 25.3 ms + 2.64 ms × k here against the source's 25.4 + 2.37 × k; measured k=0 step 23.8 ms against 23.9. The base is the same; each draft position costs 11 % more here. Not explained: drafting is a series of small kernels, so a slower host (i7-14700 against the 9800X3D) or the missing P2P could both do it; not tested.
- **The speed-up shape follows acceptance, and acceptance follows the prompt.** Structured keeps climbing to k=7 (133.9 t/s, acceptance length 5.89; the source 138.1, 5.76). Prose on this prompt peaks at k=2–3 (67.9 / 67.6) and falls from k=4; the source's prose, with its prompt, peaks at k=4 (78.3). Our depth-pass prose task, a different prose prompt, gives acceptance 2.78 at 1K and 2.87 at 127K for k=4, the source's 2.80 and 2.88, and 78.0 t/s: so the prose peak depends on the prose, not on the engine. The mixed answer (about 400 words, then JSON) behaves like prose here.
- **Decode at depth** loses 17 % without MTP and 12 % (prose) / 22 % (JSON) with k=4 from 1K to 127K. Step time at k=4: 35.8 / 37.5 / 42.1 ms at 1K / 32K / 127K; at k=0: 23.7 / 24.8 / 28.5 ms. The cost of the four draft positions grows from 12.2 ms to 13.6 ms (+12 %) over the same range, the source's depth effect (+35 % per position from 0 to 131K) is not as large here, though the fit differs (it used k=1..7 at each depth, we have k=0 and 4 only).
- **Prefill is the open difference.** The 127K prompt takes 83.1 s without MTP (1,523 t/s) and 88.6 s with k=4 (1,429 t/s); the source's 249K needle took 246.7 s (1,011 t/s at a deeper context). Step 6 measures it at 249K.

Not done in this step: the depth pass for k=1–3 and 5–7 (the source has all k); thinking on (step 4 has it for k=0 and 4).

**Next:** step 6, the 249K needle (k=6, util 0.95, as the source).
