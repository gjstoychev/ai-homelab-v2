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

## 6. The long needle (2026-11-26)

The source's needle: 249,450 prompt tokens, k=6, util 0.95, 246.7 s (1,011 t/s), recovered. `configs/fp8-mtp6.args` already has k=6 and util 0.95 (the KV pool: 266,140 tokens, 1.02× of the 262,144 context; the source 272,533, −2.4 %). New: `scripts/needle-client.py` (`long-bench.py --mode ttft`, one target, a random first line per request, 3 reps, into `needle.jsonl`), and `long-bench.py --chars` (a fixed document size in characters instead of the calibrated one).

Three attempts, the first two with `NEEDLE_TARGET`, which sizes the document by calibration (`runs/6-needle/fp8-mtp6*.{cmd,out,console.txt}`):

| Run | Document | Prompt tokens | Result |
|---|---|---|---|
| `fp8-mtp6`, target 262,000 | 885,918 chars (calibrated 3.56 chars/token) | 262,081 | **HTTP 400**: with 64 output tokens the request was 1 token over the 262,144 context; `long-bench.py` ends the target on an error, so no rep ran |
| `fp8-mtp6-250k`, target 250,000 | 737,106 chars (calibrated 3.11) | 225,215 | 3 reps, TTFT 223.5 / 223.5 / 223.4 s, **1,008 t/s**, needle found 3/3 |
| `fp8-mtp6-249k`, `--chars 835000` | 835,000 chars (estimated from the two runs above) | **252,872** | 3 reps, TTFT 272.1 / 272.2 / 272.2 s, **929 t/s**, needle found 3/3 |

The calibration is noisy at this size (the corpus changes character along its length: the same text gave 3.56 and 3.11 chars/token on two prefixes), which is why the first two missed the target by +5 % and −10 %; the fixed size landed 1.4 % above the source's 249,450 tokens. Peak 126 W per card, 15.8 GB VRAM.

**Result:** 272.2 s for 252,872 tokens against the source's 246.7 s for 249,450: **929 against 1,011 prompt tokens per second, 8 % slower** (the prompt is 1.4 % longer, which accounts for about 1 % of it: the rate falls with depth, 1,008 t/s average at 225K). That is the smallest of the prompt-reading gaps measured: −18 % at 0.5–4K, −27 % at 32K (TTFT 13.1 against 10.5 s with MTP, 12.4 s without), −8 % at 249K. The needle was recovered in every rep of both long runs. Decode after the prompt, 41–47 t/s for the 8-token answer, is not a decode measurement.

**Next:** step 7, the recipe: v0.32.0, NVFP4 KV, util 0.94 (the source's KV pool: 474,032), decode, the needle.

## 7. The recipe: v0.32.0, NVFP4 KV, util 0.94 (2026-11-26)

`configs/nvfp4-mtp4.args`: `fp8-mtp4.args` with image `local/vllm-openai:v0.32.0` (`sha256:befccf0e…`), `--kv-cache-dtype nvfp4`, `--gpu_memory_utilization 0.94`; everything else the source's compose (k=4, `max-num-seqs 2`, 262,144, PIECEWISE graphs). `scripts/recipe-client.py` runs, on one server start, step 4's three passes, step 5's depth pass (1K / 32K / 131K, k=4) and step 6's needle (835,000 characters, 252,877 tokens, 3 reps); `RECIPE_PASSES` picks passes.

**A mistake:** the first start (`runs/7-recipe/nvfp4-mtp4.*`, 150.6 s to ready: v0.32.0's first compile) completed the baseline passes, then `recipe-client.py` failed to find `long-bench.py` (a path one level too high: `experiments/scripts/…`), and `bench-run.sh` stopped the server (clean, no container left). Fixed, and the depth and needle passes ran on a second start (`nvfp4-mtp4-b.*`, ready in 94.8 s, same config, same image). The first run's console log is `nvfp4-mtp4.console.txt`, the second's `nvfp4-mtp4-b.console.txt`.

**Results** (`runs/7-recipe/summary.md`; the source's NVFP4 numbers in brackets, measured on nightly v0.32.0rc1.dev100, and fp8 from steps 4–5 for the same client):

| | NVFP4 here | fp8 here | Source NVFP4 |
|---|---|---|---|
| KV pool, tokens | **463,166** (1.77× at 262,144) | 273,333 | [474,032, 1.81×] (−2.3 %) |
| Decode, shared, prose / structured (thinking off, 256) | 58.50 / 108.25 | 62.75 / 117.02 | |
| Decode, reproduction, prose / structured / mixed (thinking on, 800) | 62.55 / 92.17 (80.2–104.4) / 71.95 | 69.81 / 81.10 / 73.97 | |
| TTFT at ~0.5K / 4K / 32K | 0.214 / 1.379 / 13.169 s | 0.200 / 1.355 / 13.146 s | [0.141 / 0.987 / 10.5 s, fp8 source] |
| Steps/s at ~1K / 32K / 127K, k=4 | 26.26 / 25.46 / 23.33 | 27.91 / 26.68 / 23.73 | [27.1 / 26.2 / 23.6 at 0 / 32K / 131K] |
| Acceptance length, depth-pass prose / structured, 1K | 2.58 / 4.14 | 2.78 / 4.23 | [prose 2.72 against 2.94 for fp8, −7.5 %] |
| Decode at 1K / 32K / 127K, prose | 67.88 / 68.17 / 61.58 | 77.98 / 70.12 / 68.30 | |
| Decode at 1K / 32K / 127K, structured | 108.44 / 110.42 / 95.09 | 117.63 / 112.94 / 92.16 | |
| TTFT at 127K | 89.3 s | 88.6 s | |
| Needle, 252,877 tokens (3 reps) | **279.3 s**, 905 t/s, found 3/3 | 272.2 s, 929 t/s (k=6) | [250.8K tokens in 274.8 s, 912.7 t/s; fp8 249,450 in 246.7 s] |
| Peak VRAM per card | 15,836 MiB | 15,790 MiB | |

What it shows:

- **The recipe works as published** on this machine: the KV pool is 2.3 % below the source's (the same gap as every fp8 pool), 1.77× of the full context, the needle is recovered at 253K tokens, no restart, no Xid.
- **NVFP4 costs a few per cent of steps/s and some acceptance, as the source found:** steps/s −6 % / −5 % / −2 % at 1K / 32K / 127K against fp8 (source: −5 % at depth 0), prose acceptance −7 % at 1K (source: −7.5 %), so prose decode is −13 % at 1K and −10 % at 127K, structured −8 % at 1K and +3 % at 127K (acceptance there 4.08 against 3.88 for fp8; one sample each). The step-rate decay at depth is −11 % from 1K to 127K against fp8's −15 %: the source's lead (−13 % against −18 %) points the same way.
- **Prompt reading: the NVFP4 needle is within 1 % of the source's** (905 against 912.7 t/s), but the source's own claim of a 10 % prefill cost against fp8 at 250K does not show here (279.3 against 272.2 s, 2.6 %): its fp8 number is 8 % ahead of ours, its NVFP4 number is not. At 4K and 32K the gap to the source remains (TTFT 1.38 s against about 1.0 s, 13.2 against 10.5 s: the source's own tokens/s for NVFP4 at those lengths, 4,093–4,178 and 3,051, equal its fp8). So the gap is not specific to the KV type; it shrinks with the prompt (in seconds it grows: +0.4 s, +2.7 s, +4.5 s). Not explained.
- **Variance:** the thinking-on structured request varies from 80 to 104 t/s between random first lines (3 reps), so a mean of 3 is soft there; the shared client's JSON (5 reps) is steady at 103–114.
- Peak power 183 W in one sample of the needle start (limit 180 W; sampling every 0.5 s).

**Next:** step 8, two GPUs other ways: pipeline parallelism, `--max-num-seqs` 1 / 2 / 4 with 1, 2 and 4 requests at once, and the 150 W cap.

## 8. Two GPUs, other ways (2026-11-26; the 150 W cap is still to do)

Base: the recipe of step 7 (v0.32.0, NVFP4 KV, util 0.94, k=4, thinking off). `scripts/run-step8.sh` starts one configuration at a time, `scripts/step8-client.py` measures it on that start: `oai-bench.py --concurrency 1,2,4` (prose and structured, 256 tokens, 3 reps after a warm-up, a random first line per request; each level is a label `NAME-cN`) and one 32K-token prompt (2 reps). Configs: `nvfp4-mtp4.args` (the recipe, `--max-num-seqs 2`), `-seqs1`, `-seqs4`, and the pipeline-parallel ones below. Files: `runs/8-parallel/` (`results.jsonl`, `long.jsonl`, `summary.md` from `scripts/step8-summary.py`, `console-*.txt`, one set of `.cmd/.out/.gpu.csv/.ram.csv` per start).

**Tensor parallelism, `--max-num-seqs` 1 / 2 / 4** (all groups in which every request ran to its token limit; total t/s = generated tokens over the group's wall time, first request sent to last stream closed; one request at once: 1.4 s of that is the prompt):

| `max-num-seqs` | KV pool | prose, 1 / 2 / 4 requests | structured, 1 / 2 / 4 requests | TTFT at 32K |
|--:|--:|--:|--:|--:|
| 1 | 469,957 | 44.1 / 43.9 / 44.5 | 67.5 / 68.7 / 68.9 | 13.16 s |
| 2 (the recipe) | 463,166 | 44.2 / 68.6 / 70.0 | 68.5 / 96.9 / 97.1 | 13.14 s |
| 4 | 446,867 | 45.6 / 69.4 / **94.0** | 67.6 / 97.2 / 120.2 (one group; the others had a request that stopped early) | 13.16 s |

- Total throughput rises only up to as many requests as there are slots: with `max-num-seqs` 1 two or four requests simply queue (4th request's first token at 19 s), with 2 the last of four waits 10 s, with 4 it waits 5.7 s. Per-stream decode falls as streams share the steps (prose 61 → 51 → 38 t/s at 1 / 2 / 4 requests with 4 slots), the total still rises.
- Each extra slot costs KV pool (463K → 447K from 2 to 4 slots). Time to first token at 32K is the same for all three: one prompt is read at a time, whatever the slots.
- Early stops: under concurrency 7 of the 63 measured structured streams of the TP runs (none of the 21 of the PP run, none of the prose streams) ended early with a short answer (an error object, a single record, `[]`): the "Request n.i of a batch test" line at the top changes what the model answers. They are flagged in `results.jsonl` (`short`) and the groups containing them are left out of the totals above.

**Pipeline parallelism** (`--pipeline-parallel-size 2` in place of `--tensor-parallel-size 2`, `--max-num-seqs 4`, MTP k=4, NVFP4 KV). It took five starts (`runs/8-parallel/pp2*.out`, `console-1.txt` … `console-5.txt`):

| Start | Setting | Result |
|---|---|---|
| 1 | default 32 / 32 layers, util 0.94, 262,144 | **no KV memory on the last stage**: weights 13.52 GiB there against 9.65 GiB on the first (the output layer and the MTP head; tensor parallelism has 8.8 GiB on each card), "No available memory for the cache blocks" |
| 2 | `VLLM_PP_LAYER_PARTITION=39,25` (layers on the stages; weights 11.06 / 12.12 GiB), 262,144 | KV memory 2.51 GiB, 3.11 needed: vLLM: at most 206,048 tokens |
| 3 | 39 / 25, 196,608 | at most 115,808 tokens |
| 4 | 39 / 25, 98,304 | started (pool 112,347), then **CUDA out of memory on the second card during the first request** (144 MiB free, 190 MiB asked): the profile run underestimates a live request |
| 5 | `41,23`, util 0.92, 65,536 | **works**: pool 107,817 tokens, 1.65× of the context |

So with MTP and NVFP4 KV, pipeline parallelism holds a pool of 108K tokens (tensor parallelism: 463K) and a 64K context on two 16 GB cards, against 262K. Measured on start 5 (not like for like: context 65,536, util 0.92, 41 / 23 layers):

| | TP, 4 slots | PP, 4 slots |
|---|--:|--:|
| Decode per stream, 1 request, prose / structured | 60.7 / 108.1 | **36.0 / 66.8** (−41 / −38 %) |
| Total, 2 requests, prose / structured | 69.4 / 97.2 | 53.2 / 81.2 |
| Total, 4 requests, prose / structured | **94.0** / 120.2 (1 group) | 86.0 / **128.9** |
| TTFT, ~4.3K prompt | 1.41 s | **1.08 s** |
| TTFT, 32K prompt | 13.16 s | **9.29 s** (−29 %) |
| Mean TTFT at 4 requests (prose) | 3.73 s | 2.59 s |

- Pipeline parallelism loses a lot on one stream (every token goes through both cards one after the other), gains on the total as more requests keep both stages busy (prose 31 → 53 → 86 t/s at 1 / 2 / 4 requests, against 46 → 69 → 94), and **reads prompts faster: 9.29 s for 32K tokens, under the source's 10.5 s on tensor parallelism**. That supports the suspicion that tensor parallelism's all-reduces (no peer-to-peer, `NCCL_P2P_DISABLE=1`, so through host memory) are what slows prompt reading here; not tested (a run with another all-reduce path would settle it).
- Peak VRAM 15,842 / 15,606 MiB; peak power sample 183 W (limit 180 W).

**Not done:** the 150 W cap (needs `sudo nvidia-smi -pl 150` and a reset to 180 W by the user, in their own shell); pipeline parallelism without MTP, which might hold a longer context; pipeline parallelism at long contexts.

## 8b. The 150 W cap (2026-11-26)

The user ran `sudo nvidia-smi -pl 150` (both cards: 180 → 150 W; the source runs its cards at 150 W). The recipe configuration (`tp2-seqs2`, `max-num-seqs` 2, thinking off) was measured again as in step 8: `STEP8_OUT=experiments/2026-11-vllm-27b-tp2/runs/8-cap150 experiments/2026-11-vllm-27b-tp2/scripts/run-step8.sh tp2-seqs2` (`run-step8.sh` got the `STEP8_OUT` override for this). The `.cmd` of the run records `power.limit 150.00 W`; same kernel 7.0.0-44, driver 580.178.04. Files: `runs/8-cap150/`.

| | 180 W | 150 W |
|---|--:|--:|
| Decode per stream, 1 request, prose / structured | 58.1 / 110.7 | 60.8 / 106.6 |
| Total t/s, 2 requests, prose / structured | 68.6 / 96.9 (one group short) | 70.0 / 95.6 |
| Total t/s, 4 requests, prose / structured | 70.0 / 97.1 (one group short) | 69.3 / 96.8 |
| TTFT at 32K | 13.14 s | 13.32 s (+1.4 %) |
| Peak power per card in the run (sampled every 0.5 s) | 139 W | 130 W |

**No measurable cost at 150 W:** the differences are inside the run-to-run spread (prose and structured swap signs), except the 32K prompt, which is 1.4 % slower. These runs use little power in the first place: mean 84–86 W per card, peaks of 130–139 W, so a 150 W limit is hardly reached; the cap did bind the clock at no sample (max 2,902 MHz in both). **Not tested:** the 250K needle, which peaked at 183 W in step 7 (a sample above the 180 W limit) and so is where a cap would bite; and the pipeline-parallel run.

The cap stays at 150 W until the user resets it (`sudo nvidia-smi -pl 180`; it does not survive a reboot).

## 9. Phase B: driver 595.91.07 (2026-11-26)

The user installed `nvidia-driver-595-server-open` 595.91.07 by hand (580 packages removed, DKMS built for 7.0.0-44), put it on `apt-mark hold`, and rebooted. Checked after the reboot: both cards at 595.91.07 and 180 W, both visible inside `local/vllm-openai:v0.32.0`, `nvidia-persistenced` active, `nvidia-smi` reports CUDA 13.2. A mistake in the check command: `docker run … local/vllm-openai:v0.32.0 nvidia-smi -L` hands `-L` to the image's entrypoint (`vllm`) and fails with "unrecognized arguments: -L"; it needs `--entrypoint nvidia-smi`. Not a driver problem. `docs/stack.md` updated (commit cd0459c).

### 9a. Smoke test, driver 595 (same config as step 3)

```
scripts/bench-run.sh experiments/2026-11-vllm-27b-tp2/runs/9-driver595/smoke fp8-mtp4 1 -- \
  experiments/2026-11-vllm-27b-tp2/scripts/vllm-serve.sh experiments/2026-11-vllm-27b-tp2/configs/fp8-mtp4.args
```

Files: `runs/9-driver595/smoke/fp8-mtp4.*`, requests in `results.jsonl`. Image v0.30.0 (config unchanged), kernel 7.0.0-44.

| | Driver 580 (step 3) | Driver 595 |
|---|---|---|
| KV pool, fp8, MTP k=4, util 0.95 | 273,333 | **278,927** (the source's figure exactly) |
| Available KV memory per card | 4.7 GiB | 4.81 GiB |
| Ready | 119.1 s | 98.3 s (engine init 40.5 s) |
| Peak VRAM per card | 15,790 MiB | 15,792 MiB |
| Prose TTFT / decode | 0.302 s / 62.95 t/s | 0.327 s / 67.42 t/s |
| Structured TTFT / decode | 0.340 s / 114.41 t/s | 0.333 s / 117.96 t/s |
| MTP acceptance length (prose / JSON) | 2.27 / 4.02 | 2.39 / 4.18 |

One rep each, so the decode differences (+7 % prose, +3 % JSON) are indications, not results; the acceptance lengths differ too, so part of it may be the prompt/sampling noise rather than the driver. The KV pool difference is a memory-accounting effect (the driver's reserved memory, or the way vLLM's profiler sees it) and is not noise. The ready time is not comparable: page cache state differed. Container stopped cleanly (no `vllm-bench` left).

**Next:** step 4 again (baseline-client on fp8-mtp4 and fp8-nomtp), the key question being TTFT at 4K and 32K.

### 9b. Step 4 again: baseline on driver 595 (2026-11-26)

```
BENCH_CLIENT=experiments/2026-11-vllm-27b-tp2/scripts/baseline-client.py scripts/bench-run.sh \
  experiments/2026-11-vllm-27b-tp2/runs/9-driver595/baseline <config> 5 -- \
  experiments/2026-11-vllm-27b-tp2/scripts/vllm-serve.sh experiments/2026-11-vllm-27b-tp2/configs/<config>.args
# fp8-mtp4, then fp8-nomtp; summary: runs/9-driver595/baseline/summary.md
```

Everything else as in step 4 (v0.30.0, same client, same prompts); only the driver changed. Both runs finished, no container left.

| | Source | Driver 580 (step 4) | Driver 595 |
|---|---|---|---|
| KV pool, MTP off | 351,618 | 348,478 | **351,618** |
| KV pool, MTP k=4 | 278,927 | 273,333 | **278,927** |
| Ready (cold→warm) | | 97.3 / 78.6 s | 78.0 / 62.4 s |
| Decode, MTP off (all kinds) | 41.99 | 42.05–42.15 | 42.25–42.35 (+0.5 %) |
| Decode k=4, shared prose / JSON | | 62.75 / 117.02 | 64.76 / 116.44 |
| Decode k=4, repro prose / structured / mixed | 78.31 / 124.89 / 81.61 | 69.81 / 81.10 / 73.97 | 67.82 / 85.76 / 78.25 |
| TTFT ~0.5K (MTP off / k=4) | 0.141 s | 0.187 / 0.200 s | 0.184 / 0.197 s |
| TTFT ~4K (MTP off / k=4) | 0.987 s | 1.314 / 1.355 s | 1.310 / 1.347 s |
| TTFT ~32K (MTP off / k=4) | 10.521 s | 12.417 / 13.146 s | **12.404 / 13.112 s** |
| Needle at the three lengths | | 9/9 | 9/9 |
| Peak VRAM per card (off / k=4) | | 15,668 / 15,790 MiB | 15,692 / 15,792 MiB |

What it shows:

- **The driver does not explain the slow prompt reading.** TTFT at 0.5K / 4K / 32K is identical to phase A within 0.3 % (32K: 12.40 s against 12.42 s without MTP; 13.11 s against 13.15 s with). The gap to the source stays at +33 % / +33 % / +18 % (MTP off). The candidates left from step 4 are the host path (CPU and memory: i7-14700 against the 9800X3D with DDR5), the NCCL path (`NCCL_P2P_DISABLE=1`, all-reduce through host memory), and the source's unpublished measurement.
- **Decode is unchanged.** MTP off is +0.5 % (42.3 against 42.1, tiny but consistent across all five kinds; the spread is 0.01). The k=4 differences (prose 62.75 → 64.76, repro structured 81.1 → 85.8) are inside the run-to-run spread, and the acceptance lengths move with them (structured repro 2.92 → 3.04). The smoke test's apparent +7 % was noise.
- **The KV pools now equal the source's to the token** (351,618 and 278,927), 0.9 % and 2.0 % higher than on 580: the new driver leaves vLLM's profiler the same free memory the source saw.

**Next:** step 6 (the 252K needle) and step 7 (the v0.32.0 NVFP4 recipe) on 595, then the NCCL P2P question as a separate test.

### 9c. Steps 6 and 7 again on driver 595 (2026-11-26)

```
NEEDLE_CHARS=835000 BENCH_CLIENT=experiments/2026-11-vllm-27b-tp2/scripts/needle-client.py scripts/bench-run.sh \
  experiments/2026-11-vllm-27b-tp2/runs/9-driver595/needle fp8-mtp6-252k 3 -- \
  experiments/2026-11-vllm-27b-tp2/scripts/vllm-serve.sh experiments/2026-11-vllm-27b-tp2/configs/fp8-mtp6.args
BENCH_CLIENT=experiments/2026-11-vllm-27b-tp2/scripts/recipe-client.py scripts/bench-run.sh \
  experiments/2026-11-vllm-27b-tp2/runs/9-driver595/recipe nvfp4-mtp4 5 -- \
  experiments/2026-11-vllm-27b-tp2/scripts/vllm-serve.sh experiments/2026-11-vllm-27b-tp2/configs/nvfp4-mtp4.args
```

Both ran back to back without a hitch (the step 7 path mistake is fixed, so one start did all passes). Files: `runs/9-driver595/needle/`, `runs/9-driver595/recipe/` (`summary.md`, `depth.jsonl`, `long.jsonl`, `needle.jsonl`). A note on my own bookkeeping: the first "finished" notification came from the launcher shell, not from the runs; I checked that `vllm-bench` was still up and waited for the real marker.

| | Source | Driver 580 | Driver 595 |
|---|---|---|---|
| fp8 k=6 KV pool (util 0.95) | 272,533 | 266,140 | **272,533** |
| fp8 needle, prompt tokens | 249,450 | 252,872 | 252,876 (3 reps) |
| fp8 needle TTFT / rate | 246.7 s / 1,011 t/s | 272.2 s / 929 t/s | **271.95 s / 930 t/s** |
| fp8 needle found | | 3/3 | 3/3 |
| NVFP4 KV pool (v0.32.0, util 0.94) | 474,032 | 463,166 | **474,032** |
| NVFP4 needle TTFT / rate | 274.8 s (250.8K) / 912.7 t/s | 279.3 s / 905 t/s | **279.09 s / 906 t/s** |
| NVFP4 needle found (all 12 requests of the run) | | 3/3 | 12/12 |
| NVFP4 TTFT ~0.5K / 4K / 32K | | 0.214 / 1.379 / 13.169 s | 0.210 / 1.369 / 13.133 s |
| NVFP4 TTFT 127K | | 89.3 s | 89.15 s |
| NVFP4 decode shared prose / JSON | | 58.50 / 108.25 | 61.75 / 113.79 |
| NVFP4 decode repro prose / structured / mixed | | 62.55 / 92.17 / 71.95 | 66.04 / 82.91 / 68.76 |
| NVFP4 decode at 1K / 32K / 127K, prose (2 reps) | | 67.88 / 68.17 / 61.58 | 70.3 / 68.3 / 66.4 |
| NVFP4 decode at 1K / 32K / 127K, structured | | 108.44 / 110.42 / 95.09 | 113.0 / 112.3 / 98.8 |
| Peak VRAM per card (fp8 k=6 / NVFP4) | | 15.8 GB / 15,836 MiB | 15,842 / 15,842 MiB |

What it shows:

- **Both KV pools equal the source's to the token** (fp8 k=6 272,533; NVFP4 474,032), as the fp8 pools in step 9b did: the 2 % memory gap of phase A came from the driver, not from the engine or the cards.
- **Prompt reading at 253K is unchanged by the driver** (271.95 against 272.2 s; 279.1 against 279.3 s), and so is TTFT at every length (127K: 89.15 against 89.3 s). The remaining gaps to the source are 10 % at 253K for fp8 (930 against 1,011 t/s), and the NVFP4 needle is within 1 % of the source's rate (906 against 912.7; prompts differ by 0.8 %).
- **Decode moves up by 3 to 8 % in most rows** (prose 1K 67.9 → 70.3, 127K 61.6 → 66.4, JSON 127K 95.1 → 98.8; shared prose 58.5 → 61.8), but the reproduction rows go both ways (structured 92.2 → 82.9, whose spread is 80–104 t/s) and the depth pass has two reps per cell. The tendency is up; it is not established by this data. It would need the k sweep or more reps to separate from prompt noise.
- No Xid, no restart, no container left.

**Phase B so far:** the driver (595.91.07, the source's) fixes the KV memory gap and changes nothing in prompt reading or decode beyond noise. The prompt-reading gap (18–33 % at 0.5–32K, 8–10 % at 253K) is therefore something else: the NCCL path, the host, or how the source measured.

**Next:** the NCCL P2P test (is `NCCL_P2P_DISABLE=1` needed here, and does enabling P2P change prompt reading?), then Strata/Basalt checks on 595.

### 9d. The NCCL P2P test (2026-11-26)

Question: is the slow prompt reading (TP=2 all-reduce through host memory) caused by `NCCL_P2P_DISABLE=1`, and does enabling P2P help? First the driver's view: `nvidia-smi topo -p2p r` under 595.91.07 still reads `CNS` (chipset not supported) between the cards, as on 580 (`docs/hardware.md`); `topo -m` shows `PHB` (through the CPU's host bridge), no NVLink. So no change was expected; the test is there to see what NCCL does and to put a measurement behind it.

Configs (`configs/fp8-nomtp-p2pon.args`, `configs/fp8-nomtp-p2psys.args`, both `fp8-nomtp.args` with NCCL logging on, `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM`):

- `p2pon`: `NCCL_P2P_DISABLE=0` (the baseline sets 1);
- `p2psys`: also `NCCL_P2P_LEVEL=SYS` (try P2P at any distance).

```
BENCH_CLIENT=experiments/2026-11-vllm-27b-tp2/scripts/baseline-client.py scripts/bench-run.sh \
  experiments/2026-11-vllm-27b-tp2/runs/9-driver595/p2p <config> 3 -- \
  experiments/2026-11-vllm-27b-tp2/scripts/vllm-serve.sh experiments/2026-11-vllm-27b-tp2/configs/<config>.args
```

Files: `runs/9-driver595/p2p/` (`summary.md`, the NCCL log in each `.out`). Three reps per kind instead of five, to keep it short.

| | `P2P_DISABLE=1` (9b, 5 reps) | `P2P_DISABLE=0` | `P2P_DISABLE=0` + `LEVEL=SYS` |
|---|---|---|---|
| NCCL transport (from the log) | not logged | `Channel 00/01 : 0[0] -> 1[1] via SHM/direct` | same, and "P2P is disabled between connected GPUs 0 and 1" 18 times |
| TTFT ~0.5K / 4K / 32K | 0.184 / 1.310 / 12.404 s | 0.185 / 1.308 / 12.413 s | 0.185 / 1.306 / 12.418 s |
| Decode, shared prose / JSON | 42.35 / 42.33 | 42.35 / 42.32 | 42.37 / 42.34 |
| Needle | 9/9 | 9/9 | 9/9 |
| Ready | 62.4 s (MTP k=4) / 78.0 s | 62.9 s | 68.4 s |

**Result: no difference at all** (TTFT within 0.1 %, decode within 0.1 %). NCCL moves the data through shared host memory (`SHM/direct`) whether or not P2P is "disabled" by the flag, because the driver offers no peer access between these cards; forcing `LEVEL=SYS` gets the message that P2P is disabled between the GPUs, and the same SHM channels. The `P2P_DISABLE=1` baseline's transport was not logged, but its timings equal the other two to the third digit, so it is the same path. vLLM's own all-reduce shortcuts are off here for other reasons (custom all-reduce disabled by flag; FlashInfer all-reduce "not supported for world_size=2"; the symmetric-memory communicator "Device capability 12.0 not supported").

What this leaves for the prompt-reading gap (+33 % at 4K, +18 % at 32K, +10 % at 253K against the source): the flag is not it, and neither is the driver. Remaining: the host (CPU and memory: i7-14700 against the source's 9800X3D with DDR5, since all-reduce goes through host memory), and the source's measurement (not published). An all-reduce cost estimate or an NCCL all-reduce benchmark (`nccl-tests` over SHM at the prefill's message sizes) would say how much of the 12.4 s the host path takes; not run.

**Next:** Strata and Basalt checks on 595 (one config each from the Flash-Next experiments), then the optional steps 10 (Basalt on the 27B), 11 (quality) and 12 (write-up).

### 9e. Strata and Basalt on driver 595 (2026-11-26)

One two-GPU MTP config of each Flash-Next experiment, same configs, same client (`scripts/oai-bench.py` through `scripts/bench-run.sh`, 5 reps, thinking off), to see whether the driver changes the other two engines:

```
scripts/bench-run.sh experiments/2026-11-vllm-27b-tp2/runs/9-driver595/strata mtp 5 -- \
  env -C ~/strata/v0.2.21/src/terminal .venv/bin/python serve/server.py --engine strata \
  --config experiments/2026-11-strata/configs/strata-2gpu-mtp.json --port 8080
BENCH_ENGINE_LOG=~/.cache/basalt/2d72cadc08663624/engine.log scripts/bench-run.sh \
  experiments/2026-11-vllm-27b-tp2/runs/9-driver595/basalt peer-mtp 5 -- \
  env -C ~/basalt/v0.5.7/src bin/basalt serve --config experiments/2026-11-basalt/configs/basalt-peer-mtp.json --python .venv/bin/python
```

**A dead end first:** the first start of both failed at once ("missing model shard", "file not found"): the Strata GGUF and the Basalt pack were in the archive `/mnt/models` (archived earlier the same day), not in `~/models`. The failed run directories were deleted (nothing was measured). Both models were moved back with `scripts/move-model.sh` (checksums OK after the copy: Strata 71 GB, copy 139 s, verify 48 s; Basalt 75 GB, copy 146 s, verify 50 s; the SATA disk is faster than the 250 MB/s noted on 2026-11-25 when it reads), then the runs repeated.

| | Driver 580 | Driver 595 |
|---|---|---|
| Strata 2-GPU MTP, prose decode | 86.2 ± 1.0 t/s (53 % drafts accepted) | 90.6 ± 3.9 t/s (56 %) |
| Strata, structured decode | 132.4 ± 3.6 (86 %) | 134.4 ± 3.3 (85 %) |
| Strata, TTFT / prompt t/s (prose) | 2.46 s / 1,765 | 2.47 s / 1,759 |
| Strata load | 20.4 s | 22.9 s |
| Basalt peer MTP, prose decode | 113.1 ± 2.3 (65 %) | 114.5 ± 3.3 (66 %) |
| Basalt, structured decode | 193.3 ± 10.1 (80 %) | 199.2 ± 6.1 (83 %) |
| Basalt, TTFT / prompt t/s (prose) | 2.34 s / 1,855 | 2.33 s / 1,865 |
| Basalt load | 23.0 s | 19.9 s |
| VRAM per card, Strata / Basalt | 15,032 / 14,938; 15,372 / 15,532 MiB | 15,004 / 14,912; 15,256 / 15,534 MiB |

Files: `runs/9-driver595/strata/`, `runs/9-driver595/basalt/`. The 580 references are `2026-11-strata/runs/10-strata-2gpu/summary.md` and `2026-11-basalt/runs/6-7-modes` (peer-mtp).

**Result: the driver changes nothing measurable for either engine.** Prompt reading within 0.4 % in both (so, again, the driver is not what limits it); decode differences (+5 % Strata prose, +3 % Basalt structured) follow the draft acceptance rate (53 → 56 %, 80 → 83 %) and are inside two spreads. Both engines started and ran on 595.91.07 without a change to the build (Strata's and Basalt's CUDA 13.0 builds run on the CUDA 13.2 driver, as they should).

**Phase B in one line:** driver 595.91.07 corrects vLLM's KV memory accounting (pools now equal the source's) and changes nothing else measured: not TTFT, not prompt reading, not the needle, not Strata or Basalt.

**Next:** the two models go back to the archive; then the optional steps 10 (Basalt on the 27B), 11 (quality), 12 (write-up).

### 9f. NCCL all-reduce between the cards (2026-11-26)

Question left by 9d: how much of the prompt-reading time is the tensor-parallel all-reduce through host memory? nccl-tests v2.22.0 built with `scripts/build-nccl-tests.sh` (commit `826b3e6`, against the host's NCCL 2.28.9, CUDA 13.0, `sm_120`, no MPI; `docs/stack.md` updated), run with `experiments/2026-11-vllm-27b-tp2/scripts/run-nccl-tests.sh runs/9-driver595/nccl-tests` (GPUs idle, both settings of `NCCL_P2P_DISABLE`, bf16 sum, 20 warm-up and 100 timed iterations, validation on: 0 wrong). Files: `runs/9-driver595/nccl-tests/allreduce-p2pdisable{1,0}.{cmd,out}`; the single-size runs' result rows are interleaved with NCCL's log lines in the `.out`.

**Message sizes** (from the model's `config.json`: hidden 5120, 64 layers): one all-reduce moves tokens × 5120 × 2 bytes. A prefill chunk is `--max-num-batched-tokens 4096` tokens = **40 MiB**; one decode token is 10 KiB, five tokens (MTP k=4 verifies 5) 50 KiB. Assumed: two all-reduces per layer (attention or linear-attention output, MLP output), 128 per forward pass, bf16.

| Size | Time, `P2P_DISABLE=1` | Time, `P2P_DISABLE=0` | Bus bandwidth |
|---|---|---|---|
| 10 KiB (1 decode token) | 10.35 µs | 10.33 µs | 1.0 GB/s |
| 50 KiB (5 tokens, MTP k=4) | 22.04 µs | 22.15 µs | 2.3 GB/s |
| 1 MiB | 151 µs | 152 µs | 6.9 GB/s |
| 4 MiB / 8 MiB | 581 / 1,149 µs | | 7.2 / 7.3 GB/s |
| **40 MiB (a 4,096-token prefill chunk)** | **5.71 ms** | **5.66 ms** | **7.35 / 7.41 GB/s** |
| 128 MiB | 18.1 ms | | 7.4 GB/s |

Transport in both settings: `via SHM/direct/direct` (the driver's `CNS`, as in 9d), so the flag is again irrelevant; the link tops out near 7.4 GB/s, which is host memory (copy in, copy out), not the PCIe x8 links (about 28 GB/s each way, Gen5).

**What it means for the measured TTFT** (all-reduce time only, 128 per chunk at the sizes above, interpolated in the sweep for partial chunks; not overlapped with compute, as in a plain TP run):

| Prompt | all-reduce estimate | TTFT measured (9b, MTP off) | share |
|---|---|---|---|
| 578 tokens (one chunk, 5.7 MiB, about 0.8 ms each) | about 0.10 s | 0.184 s | about 56 % |
| 4,212 tokens (4,096 + 116) | 0.731 + 0.022 = 0.75 s | 1.310 s | about 57 % |
| 32,267 tokens (7 × 4,096 + 3,595) | 7 × 0.731 + 0.64 = 5.76 s | 12.40 s | about 46 % |
| Decode, one token (k=0) | 128 × 10.3 µs = 1.3 ms | step 23.8 ms | about 5.5 % |
| Decode, k=4 step | 128 × 22 µs = 2.8 ms | step 35.9 ms | about 7.8 % |

**Result: about half of every prompt read is the all-reduce, through host memory at 7.4 GB/s.** Decode is spared (small messages, 5–8 % of a step), which fits the finding that decode reproduces the source's numbers while prompt reading does not. If the source's machine reduced faster, the gap to its TTFT (0.99 s at 4K, 10.5 s at 32K, that is 0.32 s and 1.9 s less than here) needs the all-reduce to be about 40 % cheaper (0.75 → 0.43 s at 4K; 5.76 → 3.9 s at 32K): plausible for a different host path, but only a measurement on that machine would show it. This is an estimate, not a profile.

Caveats: (1) nccl-tests used the host's NCCL 2.28.9; vLLM's images carry NCCL 2.29.7 (torch 2.13.0+cu130 in both). The same transport is chosen, but not measured. (2) The two all-reduces per layer, and the absence of overlap with compute, are assumptions; a profile of one prefill chunk (nsys or the torch profiler) would settle both. (3) The host here: DDR5-6000 CL40 2×32 GB (`docs/hardware.md`), i7-14700; the source's memory and CPU are not published beyond "9800X3D with DDR5".

**Next:** an NCCL run inside the vLLM image (2.29.7) is a cheap check of caveat 1; a profiled prefill chunk would settle 2. Strata and Basalt do not use NCCL (their prompt reading is also unchanged by the driver); the optional steps 10, 11, 12 remain.

### 9g. In-container NCCL, a profiled prefill chunk, and the protocol (2026-11-26)

**Caveat 1 of 9f closed.** `scripts/run-nccl-tests.sh OUTDIR IMAGE` runs the same nccl-tests binary inside the vLLM images (the image's own NCCL and CUDA runtime via `LD_LIBRARY_PATH`; the build lives on the host and is mounted read-only). A mistake on the way: `~/nccl-tests/current` is a symlink with an absolute target, dangling inside the container (exit 127); the script now resolves it to `v2.22.0`. Both images load NCCL 2.30.7 (the library reports 23007, torch's `nccl.version()` says 2.29.7), the host's is 2.28.9:

| 40 MiB all-reduce | Host (2.28.9) | v0.30.0 image | v0.32.0 image |
|---|---|---|---|
| `P2P_DISABLE=1` | 5.71 ms (7.35 GB/s) | 5.58 ms (7.52) | 5.57 ms (7.53) |
| `P2P_DISABLE=0` | 5.66 ms (7.41) | 5.56 ms (7.54) | |
| 10 KiB / 50 KiB | 10.35 / 22.04 µs | 10.3 / 22.2 µs | 10.3 / 21.9 µs |

2.5 % faster than the host's library, same transport (`SHM/direct`), so 9f's estimates hold (about 2 % lower). Files `runs/9-driver595/nccl-tests/allreduce-{host,container-v0.30.0,container-v0.32.0}-p2pdisable{1,0}.*` (the host files were renamed from `allreduce-p2pdisable*`).

**The profile.** `configs/fp8-nomtp-profile.args` (`fp8-nomtp.args` plus `--profiler-config` torch, no stacks, gzip, traces into the mounted cache directory), `scripts/profile-client.py` (two warm-up requests, `/start_profile`, one request of 4,220 tokens with `max_tokens` 1, `/stop_profile`), `scripts/trace-summary.py`. Run through `bench-run.sh` as usual; the profiled TTFT was 1.317 s against 1.30 s for the unprofiled warm-ups, so the profiler hardly disturbs it. Files: `runs/9-driver595/profile/` (`summary.md`, `profile.jsonl`, the two gzip traces in `traces/`, 2.7 MB; the root-owned originals in the container's cache directory were deleted with an `alpine` container).

| Per rank (0 / 1) | Value |
|---|---|
| Span first to last kernel | 1,299.3 / 1,299.4 ms (GPU busy 98 / 99 %) |
| NCCL all-reduce kernels | 260 / 260, 725.7 / 742.8 ms = **55.8 % / 57.2 % of the span** |
| of which the full chunk (129 kernels of about 40 MiB) | mean 5.35 / 5.37 ms, 690 / 693 ms |
| of which the remainder chunk (131 kernels, 116 tokens) | mean 275 / 286 µs |
| cutlass GEMMs (514) | 309 / 307 ms = 23.8 / 23.6 % |
| flashinfer prefill attention (32 full-attention layers) | 58 ms = 4.5 % |
| delta-rule / conv / norm / quantization kernels | 1–2 % each |

**Result: the estimate of 9f was right.** All-reduce is 56–57 % of a 4.2K-token prefill (estimated 57 %), and the GEMMs, which are the actual model, 24 %. The two ranks are almost symmetric (mean kernel times within 0.5 %), so the 5.35 ms is not mostly waiting for the other card; it is the copy through host memory. In vLLM the full-chunk all-reduce runs at 5.35 ms against 5.57 ms standalone, so the benchmark's number carries over to the real run (the 4 % is overlap with neighbouring kernels, or the 129th kernel's mix). Without the all-reduce this prefill would take about 0.57 s instead of 1.3 s.

**A wrong guess, corrected.** The kernel is named `ncclDevKernel_AllReduce_Sum_bf16_RING_LL`, which I took for the low-latency protocol (half the bandwidth, built for small messages) and a tuning chance. `scripts/run-nccl-proto.sh` forces `NCCL_PROTO` in the v0.30.0 image (`proto-container-v0.30.0.out`): at 40 MiB, unset 5.56 ms (7.54 GB/s), `Simple` 5.57 ms (7.53), `LL128` 9.81 ms (4.28), `LL` 13.85 ms (3.03); at 10 KiB / 50 KiB, LL 10.3 / 22.0 µs, unset 15.5 (one noisy start; 10.3 in the other runs) / 22.3. NCCL already picks the fast protocol at 40 MiB; the `RING_LL` in the name is the entry point NCCL uses for the ring algorithm with every protocol, not the protocol that ran. No tuning gain here.

**What this leaves:** about 0.7 s of every 4K prompt, and 5.8 s of every 32K prompt, is moving 40 MiB twice per layer through host memory at 7.4 GB/s. The source's faster TTFT would need that to be about 40 % cheaper; that is a property of the host's memory path (copy in, copy out through DDR5), not of the driver, the NCCL flags or the protocol. Options that would change it are outside this machine's current setup: P2P-capable cards or board (no), a smaller message (sequence parallelism or fp8 all-reduce: not in vLLM's TP path here), or pipeline parallelism, which step 8 already measured (32K in 9.3 s against 13.2 s with TP, at a 40 % cost on one stream).

**Next:** step 10 (Basalt on the 27B).

## 10. Basalt on the 27B (2026-11-26)

Basalt v0.5.7 runs Qwen3.8-27B (dense, `qwen35`), but the build of the Flash-Next experiment left it out (`BASALT_MODELS=qwen4exp`). `scripts/build-basalt.sh` now takes `BASALT_MODELS` and `BASALT_ROOT` from the environment (the driver package list in `build-info.txt` no longer names 580 only); a second build went to `~/basalt-qwen35/v0.5.7` (`BASALT_MODELS='qwen4exp;qwen35'`, the pinned Flash-Next build is untouched). Pack: `jesdga/Qwen3.8-27B-Basalt` revision `10c9a90e0bf9e0809c37ae439c735647b835b74e`, `Qwen3.8-27B-QUASAR-NVFP4.basalt`, 21,149,505,930 bytes, fetched and checked with `scripts/hf-fetch.sh` into `~/models/qwen3.8-27b-basalt-quasar-nvfp4`. `docs/stack.md` updated.

**What is not the same as vLLM's run:** the weights are QUASAR's quantization-aware NVFP4 checkpoint (vLLM runs `qwen3.8-27b-nvfp4-0cc2795`), the drafter is DFlash 2 (a block drafter, `spec` up to 16) instead of MTP, the KV cache is int8 (vLLM fp8), and the two cards run in `gpu-mode split` (the main card holds the first layers, the head, the drafter and the encoder, the second card the later layers; a pipeline, not tensor parallelism). `split` because the pack with its drafter takes 16 GiB, more than one card; `peer` needs the model to fit the main card. Context 262,144 with int8 KV, `parallel` 1, `prompt-cache` 0. Configs `configs/basalt27-split.json` (drafting on, the default) and `configs/basalt27-split-nodraft.json` (`"spec": 0`).

```
BENCH_CLIENT=experiments/2026-11-vllm-27b-tp2/scripts/baseline-client.py scripts/bench-run.sh \
  experiments/2026-11-vllm-27b-tp2/runs/10-basalt27b <label> 5 -- env -C ~/basalt-qwen35/v0.5.7/src \
  bin/basalt serve --config experiments/2026-11-vllm-27b-tp2/configs/basalt27-<label>.json --python .venv/bin/python
NEEDLE_CHARS=835000 BENCH_CLIENT=experiments/2026-11-vllm-27b-tp2/scripts/needle-client.py scripts/bench-run.sh \
  experiments/2026-11-vllm-27b-tp2/runs/10-basalt27b/needle split-252k 3 -- <the same server command, split.json>
```

The same clients as step 4 and 6 (step 4's three passes, then the 252K needle). Files: `runs/10-basalt27b/` (`split*`, `needle/`, `long.jsonl`, `results.jsonl`). The first start went through without a problem; the needle was found in every request (12/12 with the three TTFT lengths and the 252K reps).

| | vLLM TP=2, fp8 KV (driver 595, steps 9b/9c) | Basalt split, int8 KV | Source (vLLM, 595) |
|---|---|---|---|
| Decode, no drafting, all kinds | 42.3 t/s | **24.95 t/s** (24.3 with thinking on) | 41.99 |
| Decode drafted, shared prose / JSON (thinking off, 256) | 64.76 / 116.44 (MTP k=4) | 46.57 ± 1.91 / **127.99 ± 5.04** | |
| Decode drafted, repro prose / structured / mixed (thinking on, 800) | 67.82 / 85.76 / 78.25 | 51.40 / 80.03 / 57.58 | 78.31 / 124.89 / 81.61 |
| TTFT ~0.5K (MTP / drafter on; off) | 0.197; 0.184 s | 0.186; 0.179 s | 0.141 s |
| TTFT ~4K | 1.347; 1.310 s | **0.986; 0.940 s** | 0.987 s |
| TTFT ~32K | 13.112; 12.404 s | **5.613; 5.193 s** | 10.521 s |
| Prompt rate at 32K | 2,461 t/s | **5,749 t/s** (6,213 without drafter) | |
| Needle, 252,887 tokens (3 reps) | 272.0 s (fp8 k=6); 279.1 s (NVFP4 recipe) | **114.5 s** (2,210 t/s), found 3/3 | 246.7 s (249,450 tokens); 274.8 s (NVFP4) |
| Peak VRAM per card | 15.7–15.8 GB | 14.4 / 14.8 GB | |
| Ready | 62–78 s | 8.2 s (page cache warm) | |

What it shows:

- **Prompt reading is where Basalt wins, by a lot:** 1.4× at 4K, 2.2–2.4× at 32K, 2.4× at 252K (114.5 s against 272 s; and 2.2× faster than the source's own vLLM number). It is a pipeline over the cards, so no all-reduce goes through host memory (compare 9f/9g: that is about half of vLLM's prompt read), and the split hands a chunk's rows over once per chunk, not twice per layer. The prompt rate rises with length (3,113 → 4,272 → 5,749 t/s) as the pipeline fills; at 252K it is 2,210 t/s, in line with attention's cost growth.
- **Plain decode is where vLLM wins:** 24.95 against 42.3 t/s. A pipeline split runs the cards one after the other for every token, and each card's layers wait for the other's; the Basalt guide prices it at 0.37 ms per layer the second card runs. Tensor parallelism (vLLM) runs both at once at the cost of the all-reduces, and at decode those are small (9f).
- **Drafted decode: the drafter decides, and it depends on the prompt.** Basalt's DFlash 2 drafts structured output very well (128 t/s, 5.1× its plain decode; vLLM's MTP k=4: 116, 2.75×) and prose poorly (46.6 t/s, 1.9×; vLLM 64.8, 1.5× of a higher base). Net: JSON +10 % over vLLM, prose −28 %, repro mixed −26 %. That pattern (drafter strong where output repeats the prompt's shape) is the same one the vLLM runs showed with acceptance length.
- Drafting costs prompt reading 6–8 % (TTFT 4K 0.986 against 0.940 s; the drafter reads the prompt too), as MTP does in vLLM (3–6 %).
- Not compared here: output quality (the weights differ, QUASAR's QAT against vLLM's checkpoint; step 11), several requests at once, and `peer` mode on one card with a smaller context.

**Next:** step 11, the quality check with the ten prompts.

## 11. Quality check (2026-11-26)

The ten fixed prompts of `scripts/quality-check.py` (greedy, seed 42, thinking off, non-streamed, two repetitions each), compared with `scripts/quality-compare.py` against vLLM fp8 without MTP as the reference. Five runs, each its own server start through `scripts/bench-run.sh` with `BENCH_CLIENT=scripts/quality-check.py` (vLLM: the configs `fp8-nomtp`, `fp8-mtp4`, `nvfp4-mtp4` (v0.32.0, NVFP4 KV); Basalt: `basalt27-split-nodraft` and `basalt27-split`, labels `basalt-split-nodraft`, `basalt-split`), driver 595.91.07. Files: `runs/11-quality/` (`results.jsonl` with the full texts, `quality.md` the comparison, one `.cmd`/`.out` per run).

| prompt | fp8-nomtp (reference) | fp8-mtp4 | nvfp4-mtp4 | basalt-split-nodraft | basalt-split |
|---|---|---|---|---|---|
| arith, capital, primes, sort, author, german | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| time | ✓ | = ✓ | ≠ ✓ | = ✓ | = ✓ |
| json | ✓ | ≠ ✓ | ≠ ✓ | ≠ ✓ | ≠ ✓ |
| code | ✓ | ≠ ✓ | ≠ ✓ | = ✓ | = ✓ |
| explain | ✓ | ≠ ✓ | ≠ ✓ | ≠ ✓ | ≠ ✓ |

(= the same text as the reference in both repetitions, ≠ different, ✓ the prompt's check passes; no run disagrees with its own repetition, no `~`.)

What it shows:

- **Every prompt's check passes in every setting, 10 of 10 in all five runs, and the answers of the two repetitions are always identical.** The differences are wording, not correctness.
- **Drafting changes vLLM's text, not Basalt's.** vLLM with MTP k=4 differs from vLLM without it on 3 prompts (json: a different name, "Evelyn" for "Elias"; code: `''.join(...)` for a list; explain: a different first sentence), and the NVFP4 KV run differs on 4 (the same three and `time`, a rephrased sentence). That is what speculative decoding does when the verification pass runs on other batch shapes than plain decoding: the logits differ in the last bits and a greedy tie goes the other way; vLLM does not promise the same text. Basalt's `split` with its drafter gives, on every one of the ten prompts, the same text as `split-nodraft` (checked by hash), as its guide says (output "exactly plain decoding's"). Determinism is a property of the engine's kernels here, not of the model.
- **Basalt's QUASAR pack answers like vLLM's checkpoint on 8 of 10 prompts** (text identical to vLLM fp8 without MTP, checked by hash), and differs on json (another invented name) and explain (a different second clause). The weights are not the same file, so this is about the same level of agreement as between vLLM's own settings.
- **Not shown by this test:** whether one setting is *better*. Ten prompts, greedy, a simple check each; the KV cache type (fp8, NVFP4) and the quantizations would need a distribution-level measure (KL against BF16, or the source's `kv-quality-test.md` method, which was not run). The statement is only: no setting broke an answer, and the engines differ in whether drafting is invisible.

**Next:** step 12, the write-up (README: setup, results table, what was found; the root `README.md` index).

## 13. Basalt peer mode on the 27B (2026-11-26)

Asked for after the write-up: the one Basalt mode step 10 left out. Result: **it does not start on these cards with the published pack**, at any context where it would be of use. `runs/10-basalt27b/peer-probe.txt` has the three starts and their refusals (configs `basalt27-peer-try*.json`; every start ended at "planning weight allocation", nothing was loaded, the cards stayed at 2 MiB):

| Config | Needs on GPU 0 | Free |
|---|---|---|
| `peer`, context 32,768, drafting on | 19.7 GiB | 14.8 GiB |
| `peer`, context 32,768, `"spec": 0` | 17.4 GiB | 14.8 GiB |
| `peer`, context 8,448 (the smallest that gives a second card anything to read), `"spec": 0` | 16.6 GiB | 14.8 GiB |

Why: in `peer` the main card holds the whole model (the second card only reads the last layers of long prompts), and the pack takes 16.0 GiB with its DFlash 2 drafter; `"spec": 0` stops drafting but the drafter's weights stay in the file and on the card (17.4 against 19.7 GiB is the drafter's cells, not its weights). The Basalt guide's own number for a pack without a drafter is 14.0 GiB, which would be about 14.6 GiB at 8,448 tokens of context: it might fit, but at that context the peer never engages (it reads only prompt parts over two chunks, 8,192 tokens), so it would test nothing. The guide's peer figures (+17–27 % prompt reading) are for an RTX 5090 main card, which holds the model. For two 16 GB cards `split` is the 27B's only useful two-card mode, which is what step 10 measured.

A mistake on the way: my first cleanup used `pkill -f "basalt serve"`, which matched my own shell command and killed it (exit 144); nothing was left running (checked: no Basalt process, cards at 2 MiB), and the second and third probes ran separately.

Not done: building a no-drafter pack from QUASAR's checkpoint (about 20 GB download, converter with 16 processes) to see whether a 14 GiB main card holds it; with a context under 8,448 it would not change the answer for long prompts.

## 14. The 252K needle at 150 W (2026-11-26)

The one gap step 8b left: the needle, the heaviest load in the experiment, was not run capped. The user set the cap with `sudo nvidia-smi -pl 150` (both cards, from 180 W; confirmed with `nvidia-smi --query-gpu=index,power.limit`), driver 595.91.07, then two runs, 3 reps each, `NEEDLE_CHARS=835000`, the configs of step 6 and 7 unchanged (`runs/12-needle-cap150/`, with the `.gpu.csv` of each):

```
NEEDLE_CHARS=835000 BENCH_CLIENT=experiments/2026-11-vllm-27b-tp2/scripts/needle-client.py scripts/bench-run.sh \
  experiments/2026-11-vllm-27b-tp2/runs/12-needle-cap150 fp8-mtp6-cap150 3 -- \
  experiments/2026-11-vllm-27b-tp2/scripts/vllm-serve.sh experiments/2026-11-vllm-27b-tp2/configs/fp8-mtp6.args
RECIPE_PASSES=needle BENCH_CLIENT=experiments/2026-11-vllm-27b-tp2/scripts/recipe-client.py scripts/bench-run.sh \
  experiments/2026-11-vllm-27b-tp2/runs/12-needle-cap150 nvfp4-mtp4-cap150 3 -- \
  experiments/2026-11-vllm-27b-tp2/scripts/vllm-serve.sh experiments/2026-11-vllm-27b-tp2/configs/nvfp4-mtp4.args
```

**A mistake of mine first.** I started with the fp8 k=6 needle, saying it was the run that peaked at 183 W. It was not: the 183 W sample is the NVFP4 recipe's needle (step 7; 183.0 W on card 0 and 172 W on card 1 in `runs/7-recipe/nvfp4-mtp4-b.gpu.csv`). The fp8 needle never drew more than 126 W, so a 150 W cap could not touch it. Both were run.

| | Uncapped (180 W) | Capped (150 W) |
|---|---|---|
| fp8, k=6: TTFT, 252,876 tokens | 271.95 s (930 t/s) | 272.97 s (926 t/s), **+0.4 %** |
| fp8, k=6: peak power, card 0 / 1 | 125.7 / 117.1 W | 124.8 / 115.7 W (cap never reached) |
| fp8, k=6: mean SM clock, card 0 (active) | 2,828 MHz | 2,818 MHz |
| NVFP4 recipe: TTFT, 252,875 tokens | 279.09 s (906 t/s; step 7: 279.3 s) | 287.17 s (881 t/s), **+2.9 %** |
| NVFP4: peak power, card 0 / 1 | 183.8 / 172.1 W (9c; 183.0 / 172.2 in step 7) | **153.2 / 154.0 W** (the cap, with sampling overshoot) |
| NVFP4: samples above 150 W, card 0 | 2,037 of 3,405 (9c, whole run) | 677 of 1,965 (needle run) |
| NVFP4: SM clock, card 0, minimum while active | 2,745 MHz (9c) | 2,617 MHz |
| Needle found | 3/3 | 3/3 in both runs |

The uncapped NVFP4 files hold the whole recipe run (baseline, depth and needle passes), the capped one only the needle, so the power means are not comparable between them; the peaks, the TTFT and the clock minimum are. 

**Result: the cap is free for the fp8 needle (+0.4 %, it never draws 150 W) and costs 2.9 % on the NVFP4 needle,** the one step that draws more than 150 W (NVFP4's quantize and dequantize work adds to the load, 160 W average on card 0 against 119 W for fp8 at depth). Together with step 8b (decode within the spread, the 32K prompt 1.4 % slower), a 150 W limit costs 0–3 % in every run here, for 30 W less per card in the heaviest case. Not tested: the 150 W limit with several requests at once (the cards draw most with `max-num-seqs` 4).

The cards stay at 150 W until the user resets them (`sudo nvidia-smi -pl 180`; it does not survive a reboot).

## 15. Basalt concurrency on the 27B (2026-11-26)

The measurement step 10 left out: several requests at once on Basalt's `split` mode, with step 8's client unchanged (`scripts/step8-client.py`: `oai-bench.py --concurrency 1,2,4`, prose and structured, thinking off, 256 tokens, 3 reps after a warm-up, a random first line per request; then one 32K prompt, 2 reps), so the numbers sit next to vLLM's. Configs: `configs/basalt27-split-nodraft-par4.json` and `basalt27-split-par4-163k.json` (the step 10 configs with `"parallel": 4`; the second also `"max-context": 163840`). Files: `runs/13-basalt27b-concurrency/` (`results.jsonl`, `long.jsonl`, `summary.md`, one set of `.cmd/.out/.gpu.csv` per start; the refused start in `refused-262144-drafted/`).

```
BENCH_CLIENT=experiments/2026-11-vllm-27b-tp2/scripts/step8-client.py scripts/bench-run.sh \
  experiments/2026-11-vllm-27b-tp2/runs/13-basalt27b-concurrency <label> 3 -- env -C ~/basalt-qwen35/v0.5.7/src \
  bin/basalt serve --config experiments/2026-11-vllm-27b-tp2/configs/basalt27-<label>.json --python .venv/bin/python
```

**The drafted config with 4 slots does not start at the full context** (`refused-262144-drafted/`): "gpu-mode split: GPU 0 takes layers 0-26 at share 0.50 and needs 14.8 GiB, 14.7 GiB free; lower its gpu-share, max-context or parallel". The drafter's cells and the three extra slots' state (157 MiB each) tip the main card over by 0.1 GiB. I took the second lever, `max-context` 163,840 (Basalt shares the context between the slots: 40,960 positions each, enough for the 32K prompt), under a new config name; without the drafter 4 slots fit at 262,144 (65,536 each). So the two Basalt runs do not have the same context, and neither equals vLLM's pool (446,867 tokens shared dynamically by 4 slots).

A slip of mine on the way: a pending wait command of mine would have moved the finished run's files and edited the configs the moment the first run ended; I noticed in time and stopped that task (`TaskStop`) before it fired. Nothing was lost; the refused start's files were then moved by hand to `refused-262144-drafted/`.

Total t/s (generated tokens over the group's wall time, which for one request includes the prompt's 1 s), 1 / 2 / 4 requests at once, 3 groups each, 0 groups with an early-stopped stream:

| | prose | structured | Mean TTFT at 4 requests (prose) | TTFT at 32K | Peak VRAM per card |
|---|--:|--:|--:|--:|--:|
| Basalt split, no drafter, 4 slots, 262,144 | 22.9 / 39.5 / 64.2 | 22.9 / 38.4 / 64.1 | 2.72 s (max 4.96) | 5.20 s | 14.0 / 13.6 GB |
| Basalt split, DFlash 2, 4 slots, 163,840 | 39.6 / 57.5 / **84.1** | 89.6 / 90.3 / 103.3 | 2.87 s (max 5.28) | 5.62 s | 13.9 / 14.0 GB |
| vLLM TP=2, `max-num-seqs` 4, MTP k=4, NVFP4 KV (step 8) | 45.6 / 69.4 / **94.0** | 67.6 / 97.2 / 120.2 (1 group) | 3.73 s (max 5.66) | 13.16 s | 15.8 GB |
| vLLM PP=2, `max-num-seqs` 4, 65,536 (step 8) | 31.3 / 53.2 / 86.0 | 52.2 / 81.2 / 128.9 | 2.59 s (max 3.89) | 9.29 s | 15.8 / 15.6 GB |

Decode per stream at 1 / 2 / 4 requests, Basalt drafted: prose 46.6 / 38.3 / 28.7 t/s, structured 137.9 / 79.3 / 42.4 (vLLM TP=2, 4 slots: prose 60.7 / 51.2 / 38.0, structured 108.1 / 88.3 / 62.5).

What it shows:

- **Basalt's slots share the weight pass, and the totals rise without a drafter: 22.9 → 39.5 → 64.2 t/s (2.8× at 4 requests)**, against vLLM TP's 2.1× for prose with MTP (45.6 → 94.0, drafting included). Plain decode at 4 requests: Basalt 64 t/s; no vLLM number without MTP was measured at 4 slots, so there is no like-for-like.
- **With drafting the picture depends on the output.** Prose: 39.6 → 84.1 (2.1×), 11 % below vLLM TP at 4 requests (94.0), 13–17 % below at 1–2. Structured: **one request is the best case, 89.6 t/s total (138 per stream), 33 % above vLLM (67.6)**, but two requests add nothing (90.3: each stream falls to 79 t/s) and four reach 103.3, 14 % below vLLM's 120.2 (a single complete group there). The guide says why: a window of drafted rows takes 8 of the 16 rows a pass can carry, so only two drafted slots share a pass; a drafter that is excellent for one stream has little left to batch.
- **Prompt reading is unchanged by the slots:** 32K in 5.20 s without and 5.62 s with the drafter (step 10, one slot: 5.19 / 5.61 s). A prompt is read by one slot at a time on both engines; Basalt's is 2.3–2.5× faster than vLLM TP=2 (13.16 s) and 1.6× faster than vLLM PP=2 (9.29 s).
- **No stream answered early in Basalt's 112 requests** (0 short groups, against 7 of 63 structured TP streams in vLLM at step 8): the same prompts and the same "request n.i of a batch test" line. Two separate things, not explained here: vLLM's early stops came with MTP and batched drafts, Basalt's output does not depend on the batch (its guide says rows are bitwise independent).
- Peak VRAM falls to 13.9–14.0 GB per card at 4 slots (the lower context and the unchanged weights; the vLLM runs sit at 15.8 GB because their pool takes what is left).

**Net for two cards and several clients:** vLLM TP=2 gives the best totals for prose from 1 to 4 requests (and holds 262K context with 463K pool), Basalt split the best single-stream JSON and by far the fastest prompts; at 4 requests both are within 11–14 % of each other, in the opposite direction for prose and JSON. Not tested: more than 4 slots (Basalt allows 8), the 4-slot start at 262,144 with the drafter (needs a different `gpu-share`), and a plain-decode vLLM at 4 slots for a like-for-like no-drafter comparison.
