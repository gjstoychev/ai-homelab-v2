# vLLM: Qwen3.8-27B on 2× RTX 5060 Ti with tensor parallelism, a reproduction

Status: **done** (2026-11-26), steps 0–12; the findings are [below](#findings). Every step with the exact command, expected vs actual: [`journal.md`](journal.md).

## Question

Can `formula` reproduce the published dual RTX 5060 Ti numbers for Qwen3.8-27B on vLLM (TP=2, NVFP4 weights, MTP: 78.3 t/s prose, 124.9 t/s structured, a 249K-token prompt in 247 s), with the source's pinned versions? Then, for the main question of this repository: **how do the two cards do best with a dense 27B model** that does not fit one card, with tensor parallelism (every layer split across both cards) against pipeline parallelism (layers divided between them), and against Basalt's peer and split modes on the same model?

Sources: the video *Best Ways to Run Qwen3.8-27B on Dual RTX 5060 Ti Cards* (RepoChad, 2026-09-25) and the repository behind it, [skrodahl/qwen38-27B-dual-rtx5060](https://github.com/skrodahl/qwen38-27B-dual-rtx5060) (README, `kv-quality-test.md`, `kvq/`). The source's machine has the same cards (2× RTX 5060 Ti 16 GB, Gen5 x8/x8) and 64 GB of DDR5, with a Ryzen 7 9800X3D.

## Why the versions stay pinned

The video and the repository pin the checkpoint on purpose: later revisions of `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` removed the MTP head (commit `7b770e5`, 2026-09-01: "Final build: NVFP4 incl. lm_head, MTP head removed, 3 shards -> 2"). Without it, speculative decoding does not start and decode falls to about 42 t/s. The newest checkpoint or vLLM is therefore not "better" here; it is a different experiment. Newer versions are a later, separate step, if at all.

## Pinned versions

| Part | Version | Note |
|---|---|---|
| Checkpoint | `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` at **`0cc27958cefbbe231782ec8511de8c4eb5233348`** (2026-08-19), 18.8 GB in 3 safetensors shards, Apache-2.0 | the last revision with the MTP head; lm_head in NVFP4 (`078c9f6`) |
| vLLM for the FP8-KV tables | `vllm/vllm-openai:v0.30.0`, **`sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`** (built 2026-09-21) | the source gives the version, not the tag or digest; the digest is the tag's today |
| vLLM for the published recipe (NVFP4 KV) | `vllm/vllm-openai:v0.32.0`, **`sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34`** (built 2026-10-18) | the source's NVFP4 tables were measured on a nightly before the release (`nightly-77154ca6…`, v0.32.0rc1.dev100), not re-run on the release |
| CUDA in both images | **13.0.2** (`CUDA_VERSION=13.0.2`, `NV_CUDA_CUDART_VERSION=13.0.96-1`, `TORCH_CUDA_ARCH_LIST` includes 12.0) | read from the image configs, without pulling |
| NVIDIA driver, phase A | 580.178.04 (replaced by 595.91.07 on 2026-11-26) | runs CUDA 13.0.2 images (minimum 580.95.05) |
| NVIDIA driver, phase B | **595.91.07**, the source's: Ubuntu's `nvidia-driver-595-server-open` 595.91.07-0ubuntu0.24.04.1, on `apt-mark hold`, installed by hand and rebooted on kernel 7.0.0-44 | the same package family as phase A (server, open kernel module); `nvidia-persistenced` active |
| NCCL | host 2.28.9 (nccl-tests only); in the vLLM images 2.30.7 (the loaded library; torch reports 2.29.7) | NCCL tests: nccl-tests v2.22.0 (`826b3e6`) |
| Basalt for step 10 | v0.5.7 built with `BASALT_MODELS='qwen4exp;qwen35'` in `~/basalt-qwen35`; pack `jesdga/Qwen3.8-27B-Basalt` at `10c9a90e`, QUASAR NVFP4 weights, DFlash 2 drafter | not vLLM's checkpoint |

The source's host: Ubuntu 26.04, kernel 7.0.0-31, driver 595.91.07, CUDA 13.2 (on the host; the containers use their own 13.0.2). Here: Ubuntu 24.04.5, kernel 7.0.0-44 (since the reboot at 12:50 on 2026-11-26; the first, unfinished smoke try ran on 7.0.0-43), i7-14700. Inside Docker the host's CUDA toolkit does not matter; the driver does.

**Two phases** separate the driver from everything else: phase A measures on the current driver, phase B installs 595.91.07 by hand and repeats the key runs. The difference is the driver's effect, and phase B is the exact reproduction.

## The source's configuration (FP8-KV baseline and recipe)

From the README's compose file and text (the compose is the v0.32.0 NVFP4-KV recipe; the FP8 baseline differs in `--kv-cache-dtype fp8` and `--gpu_memory_utilization 0.95`):

- `--tensor-parallel-size 2`, `--max-model-len 262144`, `--max-num-seqs 2`, `--max-num-batched-tokens 4096`
- `--speculative-config '{"method": "mtp", "num_speculative_tokens": 4}'` (k=4, the operating point of a 0–7 sweep)
- `--compilation-config '{"cudagraph_mode": "PIECEWISE"}'`, `--no-enable-flashinfer-autotune` ("FlashInfer autotune OOMs on vLLM versions after 0.28.0"), `--disable_custom_all_reduce`
- `--mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype bfloat16`, `--enable-prefix-caching --enable-chunked-prefill`
- `--limit-mm-per-prompt '{"video": 0}' --mm-processor-kwargs '{"max_pixels": 1000000}'`, `--reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice`, `--trust-remote-code`, `--load-format ipc_cache`
- environment: `NCCL_P2P_DISABLE=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=67108864`; `ipc: host`
- power: "Capped at 150 W per card (stock max 198 W)"; these cards' limit is 180 W. The SM clock lock (`nvidia-smi -lgc 300,2500`, added 2026-10-14) came after the decode tables.

Changes here: `--served-model-name bench` (the shared clients send that name), no `--api-key`, the model directory mounted read-only from `~/models/`.

## The published numbers to reproduce

| | Source (fp8 KV, v0.30.0) |
|---|---|
| Decode, MTP off (all kinds) | 41.99 t/s |
| Decode, MTP k=4, prose / structured / mixed | 78.31 / 124.89 / 81.61 t/s |
| KV pool, MTP off / k=4 | 351,618 / 278,927 tokens |
| TTFT at 523 / 4,123 / 32,812 prompt tokens | 0.141 / 0.987 / 10.521 s |
| 249,450-token needle | 246.7 s (1,011 t/s), recovered |
| k sweep 0–7, decode at depth 0 / 32K / 131K | README tables (journal) |

The source's measurements: greedy, thinking on (reasoning tokens counted in decode), "decode tok/s excludes prefill", one vLLM restart per k, the first run after a restart discarded; the harness itself is not published.

## Method

- **The shared clients** (`scripts/bench-run.sh`, `oai-bench.py`, `long-bench.py`, `quality-check.py`), so the 27B numbers sit next to the Flash-Next ones in the same terms (thinking off, 256 tokens, prose and JSON).
- **A reproduction mode** closer to the source: thinking on, 800 output tokens, prose, structured and a mixed prompt (prose then a JSON summary), MTP acceptance from vLLM's metrics. The source's prompt texts are not published, so the reproduction is in kind, not prompt for prompt.
- Power: the runs at the cards' 180 W first; the 150 W cap is a step of its own (`sudo nvidia-smi -pl 150`, run by the user).

## Steps

0. **Disk:** the finished Flash-Next comparison models archived; unused Docker images removed (journal).
1. **Images:** `docker pull vllm/vllm-openai@sha256:8a69ff…` and `@sha256:befccf…`, tagged locally as `vllm-openai:v0.30.0` / `v0.32.0`; record size and the image's vLLM, PyTorch and FlashInfer versions.
2. **Checkpoint:** the 15 files the source's `wget` loop takes, at `0cc27958`, into `~/models/qwen3.8-27b-nvfp4-0cc2795/`, checked against the Hub's hashes (`scripts/hf-fetch.sh` handles LFS files only; the small config files need the git blob hash, a small change to the script).
3. **Phase A, smoke test:** v0.30.0, fp8 KV, MTP k=4, the source's flags. From the log: the KV pool (source: 278,927), that MTP loaded (acceptance not zero), CUDA graphs, memory per card.
4. **Phase A, baseline:** MTP off and k=4, our standard prompts and the reproduction mode; TTFT at about 0.5K, 4K and 32K.
5. **k sweep 0–7** (one restart per k) and **decode at depth** 0 / 32K / 131K for k=0 and k=4.
6. **The 249K needle** (k=6, util 0.95, as the source), with `scripts/long-bench.py`.
7. **The recipe:** v0.32.0, NVFP4 KV, util 0.94: KV pool (source: 474,032), decode, the needle.
8. **Two GPUs, other ways:** pipeline parallelism (`--pipeline-parallel-size 2` instead of TP=2), `--max-num-seqs` 1 / 2 / 4 with 1, 2 and 4 requests at once (total t/s), and the 150 W cap.
9. **Phase B:** driver 595.91.07 installed by hand (dry run first; a reboot), `docs/stack.md` updated; then steps 3–4 and 7 again, and one Strata and one Basalt configuration from the Flash-Next experiments as a check that the new driver changes nothing there.
10. **Optional, Basalt on the 27B:** Basalt runs Qwen3.8-27B in peer and split mode; its pack is a separate download. Same prompts, against vLLM TP=2.
11. **Quality check** with the ten prompts (vLLM, and Basalt if step 10 ran).
12. **Write-up.**

## Disk

| | Size | Where |
|---|---|---|
| Checkpoint at `0cc27958` | 18.8 GB | `~/models/` |
| Two vLLM images | to be measured (step 1) | Docker, on the NVMe |
| vLLM's compile cache | small | `~/.cache/vllm` (mounted into the container) |

After step 0 the NVMe has about 268 GB free.

## Results

Phase A (driver 580.178.04, kernel 7.0.0-44, 180 W), vLLM 0.30.0, fp8 KV. Decode in t/s; the source's prompts are not published, so the reproduction mode is in kind. Details and the shared client's numbers: [`runs/4-baseline/summary.md`](runs/4-baseline/summary.md), journal step 4.

| | Source | Here, MTP off | Here, MTP k=4 |
|---|---|---|---|
| KV pool, tokens | 351,618 / 278,927 | 348,478 | 273,333 |
| Decode, prose (thinking on, 800 tokens) | 41.99 / 78.31 | 42.05 | 69.81 (64.0–76.0) |
| Decode, structured | 41.99 / 124.89 | 42.05 | 81.10 with thinking (all reasoning); 117.02 as JSON, thinking off |
| Decode, mixed | 41.99 / 81.61 | 42.05 | 73.97 |
| TTFT at ~0.5K / 4K / 32K tokens | 0.141 / 0.987 / 10.521 s | 0.187 / 1.314 / 12.417 s | 0.200 / 1.355 / 13.146 s |

**k sweep, MTP k=0–7** (step 5; thinking off, 640 tokens, about 4.3K-token prompts; details and the source's numbers per cell in journal step 5, [`runs/5-sweep/summary.md`](runs/5-sweep/summary.md)):

| k | steps/s | prose t/s | structured t/s | mixed t/s |
|--:|--:|--:|--:|--:|
| 0 | 42.07 | 42.07 | 42.07 | 42.07 |
| 1 | 35.64 | 60.1 | 69.6 | 62.6 |
| 2 | 32.74 | 67.9 | 91.6 | 72.1 |
| 3 | 30.12 | 67.6 | 107.2 | 73.6 |
| 4 | 27.84 | 63.9 | 117.6 | 73.0 |
| 5 | 25.89 | 59.4 | 123.0 | 68.9 |
| 6 | 24.30 | 56.2 | 127.7 | 62.8 |
| 7 | 22.81 | 53.0 | 133.9 | 58.5 |

Steps/s is within 1–4 % of the source's at every k, and the step time follows a straight line, 25.3 ms + 2.64 ms × k (the source: 25.4 + 2.37 × k). Prose and mixed peak at k=2–4 and fall after, structured keeps climbing to k=7. Decode at depth, k=0 / k=4, at about 1K / 32K / 127K tokens: 42.25 / 40.25 / 35.08 and prose 78.0 / 70.1 / 68.3, JSON 117.6 / 112.9 / 92.2 t/s (the source's, at 0 / 32K / 131K: 41.99 / 39.98 / 34.66, prose 80.1 / 69.6 / 68.8, JSON 123.3 / 115.6 / 92.4). The one open difference so far is prompt reading (TTFT 12.4 s at 32K against 10.5 s; 83 s at 127K).

**Long needle** (step 6; k=6, util 0.95, MTP; journal step 6): 252,872 prompt tokens in 272.2 s (929 prompt t/s), needle recovered 3/3; the source: 249,450 tokens in 246.7 s (1,011 t/s). At 225,215 tokens: 223.5 s (1,008 t/s). The prompt-reading gap to the source is 8 % at this size, against 18–27 % at 0.5–32K.

**The recipe** (step 7; v0.32.0, NVFP4 KV, util 0.94, k=4): KV pool 463,166 tokens (source 474,032), needle recovered at 252,877 tokens in 279.3 s (905 t/s; the source's NVFP4 needle 912.7 t/s). Against fp8 here: steps/s −6 / −5 / −2 % at 1K / 32K / 127K, prose acceptance −7 % (source −7.5 %), decode at 1K prose 67.9 against 78.0 t/s, JSON 108.4 against 117.6; time to first token the same (13.17 s at 32K, 89.3 s at 127K).

**Two GPUs, other ways** (step 8; recipe, thinking off, 256 tokens; journal step 8, [`runs/8-parallel/summary.md`](runs/8-parallel/summary.md)). Total t/s with 1 / 2 / 4 requests at once, prose:

| | 1 | 2 | 4 | TTFT at 32K |
|---|--:|--:|--:|--:|
| TP=2, `max-num-seqs` 1 | 44.1 | 43.9 | 44.5 | 13.2 s |
| TP=2, `max-num-seqs` 2 (the recipe) | 44.2 | 68.6 | 70.0 | 13.1 s |
| TP=2, `max-num-seqs` 4 | 45.6 | 69.4 | 94.0 | 13.2 s |
| PP=2, `max-num-seqs` 4, 65,536 context | 31.3 | 53.2 | 86.0 | **9.3 s** |

Tensor parallelism is faster for one request (decode per stream 61 against 36 t/s) and holds the full 262K context; pipeline parallelism, which needed five starts (the last stage runs out of KV memory: its pool is 108K tokens against 463K), catches up with several requests (structured at 4: 129 against 120 t/s) and reads a 32K prompt 29 % faster. The 150 W cap changes nothing measurable in these runs (the cards draw 84–86 W on average and peak at 130–139 W): decode and totals are within the run-to-run spread, the 32K prompt is 1.4 % slower (13.32 against 13.14 s); the 250K needle, which peaked at 183 W, was not run capped.

**Phase B, driver 595.91.07** (steps 9–9g; same configs, same clients; journal 9a–9g, runs in [`runs/9-driver595/`](runs/9-driver595/)). The source's driver is the exact reproduction; the table puts the three side by side:

| | Source | Phase A (580) | Phase B (595) |
|---|---|---|---|
| KV pool fp8, MTP off / k=4 | 351,618 / 278,927 | 348,478 / 273,333 | **351,618 / 278,927** |
| KV pool, k=6 / NVFP4 recipe | 272,533 / 474,032 | 266,140 / 463,166 | **272,533 / 474,032** |
| Decode, MTP off | 41.99 | 42.05–42.15 | 42.25–42.35 |
| Decode k=4, thinking on, prose / structured / mixed | 78.31 / 124.89 / 81.61 | 69.81 / 81.10 / 73.97 | 67.82 / 85.76 / 78.25 |
| Decode k=4, thinking off, prose / JSON | | 62.75 / 117.02 | 64.76 / 116.44 |
| TTFT ~0.5K / 4K / 32K (MTP off) | 0.141 / 0.987 / 10.521 s | 0.187 / 1.314 / 12.417 s | 0.184 / 1.310 / 12.404 s |
| fp8 needle (k=6), tokens, time | 249,450, 246.7 s (1,011 t/s) | 252,872, 272.2 s (929 t/s) | 252,876, 271.95 s (930 t/s) |
| NVFP4 recipe needle, prompt t/s | 912.7 | 905 | 906 |

The new driver corrects vLLM's KV memory accounting (the pools now equal the source's to the token) and changes nothing else that was measured: not prompt reading, not decode (differences of 0.5–5 % follow the draft acceptance and sit inside the spread), not the needle. Strata's and Basalt's two-GPU MTP configurations (Flash-Next) are also unchanged within 0.4 % on prompts and inside their spread on decode (journal 9e).

**Where the prompt time goes** (journal 9d–9g). `NCCL_P2P_DISABLE` is not the cause: the driver reports no peer access between the cards (`CNS`), NCCL uses shared host memory (`SHM/direct`) with the flag at 0, at 1 and with `NCCL_P2P_LEVEL=SYS`, and TTFT and decode are identical to 0.1 %. An all-reduce benchmark between the two cards (nccl-tests, bf16) moves a 4,096-token prefill chunk (40 MiB) in 5.6–5.7 ms (7.4 GB/s; 5.57 ms inside the vLLM images' own NCCL), 10 µs for one decode token. A torch-profiler trace of one 4,220-token prefill (no MTP) puts **NCCL all-reduce at 56–57 % of the 1,299 ms**, the GEMMs at 24 %, flashinfer attention at 4.5 %; the full-chunk all-reduces average 5.35 ms in vLLM against 5.57 ms standalone, and NCCL's default protocol is already the fastest one (forced `LL` 3.0 and `LL128` 4.3 GB/s against 7.5). Decode is spared: its all-reduces take 5–8 % of a step.

**Basalt on the 27B** (step 10; `gpu-mode split`, QUASAR NVFP4 weights, DFlash 2 drafter, int8 KV, 262,144 context; not vLLM's checkpoint, drafter or KV type; journal 10, [`runs/10-basalt27b/`](runs/10-basalt27b/)):

| | vLLM TP=2, fp8, driver 595 | Basalt split |
|---|---|---|
| Decode, no drafting | 42.3 t/s | 24.95 t/s |
| Decode drafted, thinking off, prose / JSON | 64.76 / 116.44 (MTP k=4) | 46.57 / 127.99 |
| Decode drafted, thinking on, prose / structured / mixed | 67.82 / 85.76 / 78.25 | 51.40 / 80.03 / 57.58 |
| TTFT ~0.5K / 4K / 32K (drafter or MTP on) | 0.197 / 1.347 / 13.112 s | 0.186 / 0.986 / 5.613 s |
| 252K needle | 272.0 s (fp8, k=6), 279.1 s (NVFP4) | **114.5 s** (2,210 t/s), found 3/3 |
| Peak VRAM per card | 15.7–15.8 GB | 14.4 / 14.8 GB |

**Quality check** (step 11; ten prompts, greedy, two repetitions, reference vLLM fp8 without MTP; [`runs/11-quality/quality.md`](runs/11-quality/quality.md)): every prompt's check passes in all five settings (vLLM fp8 with and without MTP, vLLM NVFP4 recipe, Basalt split with and without drafter). The wording differs: vLLM with MTP gives another text than without on 3 of 10 prompts (4 with the NVFP4 KV), Basalt's drafted text equals its undrafted text on all ten, and Basalt's QUASAR pack gives vLLM's text on 8 of 10.

## Findings

### What reproduces

- **Decode without MTP, to the third digit:** 42.05–42.35 t/s against 41.99, for every kind of prompt; KV pools equal the source's to the token on the source's driver (595.91.07), 0.9–2.3 % short of them on 580.
- **MTP's step time and its scaling:** steps/s within 1–4 % of the source's for every k from 0 to 7 (25.3 ms + 2.64 ms × k against 25.4 + 2.37 × k), decode at depth 0 / 32K / 131K within 5 % at every cell measured (k=0 and k=4, prose and JSON), the NVFP4 recipe's cost (steps/s −2 to −6 %, prose acceptance −7 % against the source's −7.5 %), and its needle (906 t/s against 912.7).
- **MTP decode in kind, not in number:** as JSON the model decodes at 116–117 t/s (the source: 124.9, −6 %), as prose at 62–70 t/s (78.3, which sits at the top of our range). The source's prompts are not published; decode follows the draft acceptance length (2.3–2.5 for prose, 4.2 for JSON), and that is the same as the source's. The honest statement is that the engine side matches and the number depends on the prompt.

### What does not reproduce, and why

- **Prompt reading is 18–33 % slower than the source's at 0.5–32K tokens (TTFT 1.31 against 0.99 s at 4K, 12.4 against 10.5 s at 32K) and 8 % in prompt rate at 252K** (930 against 1,011 t/s; 272 against 247 s). It is not the driver (9b, 9c), not the power limit (8b), not the KV type, not `NCCL_P2P_DISABLE`, not the NCCL version or its protocol (9d, 9g). What it is: about half of every prefill is the tensor-parallel all-reduce through host memory, 128 transfers of 40 MiB per chunk at 7.4 GB/s (9f, 9g); a faster host path would shrink the gap, and the source's gap would need that part to be about 40 % cheaper. The source's CPU (9800X3D) and memory differ from here (i7-14700, DDR5-6000 CL40), but no run on this machine changes either, so that part is a hypothesis, not a finding.
- **The 150 W cap** the source used costs 0–3 % here: nothing measurable in decode (these runs draw 84–86 W on average, peaks of 130–139 W; step 8b), +0.4 % on the fp8 252K needle (it never draws 150 W), **+2.9 % on the NVFP4 recipe's needle** (287.2 s against 279.1 s; the one load that draws more, peaks of 183 W uncapped, 153 W capped; journal 14).

### The best two-GPU mode for a dense 27B model

It depends on what is read and what is written:

- **One chat, plain decode:** vLLM tensor parallelism: 42 t/s plain (24.95 for Basalt's split), and with MTP about 61 t/s per stream in step 8's runs.
- **Prompt-heavy work (long documents, agents re-reading context):** pipelined prompt reading. vLLM PP=2 reads 32K in 9.3 s against 13.2 s with TP=2 (at a 40 % cost on one stream, 65K context only); **Basalt split reads 32K in 5.6 s and 252K in 114.5 s**, 2.2–2.4× faster than vLLM TP=2, because it passes a chunk's rows once instead of all-reducing twice per layer.
- **Structured output:** Basalt's DFlash 2 drafter gets 128 t/s on JSON, 10 % above vLLM's MTP k=4, and loses on prose (−28 %).
- **Several requests at once:** vLLM's totals rise only up to its slot count (`--max-num-seqs` 4: 44 → 69 → 94 t/s at 1 / 2 / 4); Basalt's concurrency was not measured on the 27B.
- **Basalt peer mode** does not start on these cards with the published pack (journal 13): the main card must hold the whole 16 GiB model, so `split` is the only two-card mode of the 27B here.
- **Context:** vLLM TP=2 holds 262,144 tokens (NVFP4 KV: 463K-token pool); PP=2 only 65,536 with five starts to fit; Basalt split holds 262,144 with int8 KV.
- **Output stability:** Basalt's drafted text equals its undrafted text; vLLM's MTP changes the wording on 3 of 10 prompts (correctness checks pass everywhere).

### Not measured

Basalt with several requests, a no-drafter 27B pack (Basalt's `peer` mode, tried in step 13, refuses the published pack at every useful context: it needs 16.6 GiB on GPU 0 at the smallest context where a second card reads anything, 14.8 GiB are free), a matched checkpoint (vLLM and Basalt use different NVFP4 weights; the 252K comparison also differs in KV type), sequence-parallel or fp8 all-reduce in vLLM (nothing to switch on in these images), the profile with MTP or at 32K, the source's own KV-quality test (`kv-quality-test.md`), quality at distribution level (KL against BF16), and the 150 W cap with several requests at once.

