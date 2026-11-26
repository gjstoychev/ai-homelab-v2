# vLLM: Qwen3.8-27B on 2× RTX 5060 Ti with tensor parallelism, a reproduction

Status: **in progress** (started 2026-11-26; phase A, steps 3–8 done except the 150 W cap). Every step with the exact command, expected vs actual: [`journal.md`](journal.md).

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
| NVIDIA driver, phase A | 580.178.04 (installed now) | runs CUDA 13.0.2 images (minimum 580.95.05) |
| NVIDIA driver, phase B | **595.91.07**, the source's: Ubuntu's `nvidia-driver-595-server-open` 595.91.07-0ubuntu0.24.04.1 | the same package family as now (server, open kernel module) |

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

Tensor parallelism is faster for one request (decode per stream 61 against 36 t/s) and holds the full 262K context; pipeline parallelism, which needed five starts (the last stage runs out of KV memory: its pool is 108K tokens against 463K), catches up with several requests (structured at 4: 129 against 120 t/s) and reads a 32K prompt 29 % faster. The 150 W cap is not measured yet.

More steps to come.

## Findings

Pending.
